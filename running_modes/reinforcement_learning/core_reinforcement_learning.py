# -------------------------------------------------------------------------------------------------------------
# this file has been modified from https://github.com/MolecularAI/Reinvent for Augmented Memory implementation
# -------------------------------------------------------------------------------------------------------------

import time

import numpy as np
import torch
from reinvent_chemistry.utils import get_indices_of_unique_smiles
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.configurations import ReinforcementLearningConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.reinforcement_learning.margin_guard import MarginGuard
from running_modes.utils.general import to_tensor

from reinvent_chemistry.conversions import Conversions
from copy import deepcopy

import pandas as pd


class CoreReinforcementRunner(BaseRunningMode):

    def __init__(self, critic: GenerativeModelBase, actor: GenerativeModelBase,
                 configuration: ReinforcementLearningConfiguration,
                 scoring_function: BaseScoringFunction, diversity_filter: BaseDiversityFilter,
                 inception: Inception, logger: BaseReinforcementLogger):
        self._prior = critic
        self._agent = actor
        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self.config = configuration
        self._logger = logger
        self._inception = inception
        self._margin_guard = MarginGuard(self)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)

        # optimization algorithm
        self.optimization_algorithm = configuration.optimization_algorithm.lower()
        # specific algorithm parameters
        self.top_k = configuration.specific_algorithm_parameters.get("top_k", 0.5)
        self.alpha = configuration.specific_algorithm_parameters.get("alpha", 0.5)
        self.update_frequency = configuration.specific_algorithm_parameters.get("update_frequency", 5)
        # SMILES augmentation hyperparameters
        self.augmented_memory = configuration.augmented_memory
        self.augmentation_rounds = configuration.augmentation_rounds
        self.selective_memory_purge = configuration.selective_memory_purge
        # SMILES randomization functions from reinvent-chemistry
        self._chemistry = Conversions()

        # track the sampling as a function of oracle calls
        self.oracle_tracker = pd.DataFrame({'step': [],
                                            'oracle_calls': [],
                                            'total_score': [],
                                            'smiles': []})

    def run(self):
        self._logger.log_message("starting an RL run")
        start_time = time.time()
        self._disable_prior_gradients()

        if (self.optimization_algorithm == "augmented_memory") or (self.optimization_algorithm == "reinvent"):
            oracle_calls = 0
            for step in range(self.config.n_steps):
                seqs, smiles, agent_likelihood = self._sample_unique_sequences(self._agent, self.config.batch_size)
                # switch signs
                agent_likelihood = -agent_likelihood
                prior_likelihood = -self._prior.likelihood(seqs)
                score_summary: FinalSummary = self._scoring_function.get_final_score_for_step(smiles, step)
                score = self._diversity_filter.update_score(score_summary, step)

                if oracle_calls >= 5000:
                    print('----- Oracle Budget Reached -----\nEnding Run')
                    break

                # track oracle calls based on valid indices as invalid SMILES are discarded
                oracle_calls += len(score_summary.valid_idxs)

                self.update_oracle_tracker(step=step,
                                           oracle_calls=oracle_calls,
                                           total_score=score,
                                           smiles=smiles)

                augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(score)
                loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                # if augmented_memory is true, over-ride it here to not use it as we want to perform memory *after* sampling new SMILES in case of new "best-so-far" SMILES
                loss, agent_likelihood = self._inception_filter(self._agent, loss, agent_likelihood, prior_likelihood, smiles, score, self._prior, override=True)

                loss = loss.mean()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.augmented_memory:
                    if self.selective_memory_purge:
                        self._inception.selective_memory_purge(smiles, score)
                    for _ in range(self.augmentation_rounds):
                        # get randomized SMILES
                        randomized_smiles_list = self._chemistry.get_randomized_smiles(smiles, self._prior)
                        # get prior likelihood of randomized SMILES
                        prior_likelihood = -self._prior.likelihood_smiles(randomized_smiles_list)
                        # get agent likelihood of randomized SMILES
                        agent_likelihood = -self._agent.likelihood_smiles(randomized_smiles_list)
                        # compute augmented likelihood with the "new" prior likelihood using randomized SMILES
                        augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(score)
                        # compute loss
                        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                        # experience replay using randomized SMILES
                        loss, agent_likelihood = self._inception_filter(self._agent, loss, agent_likelihood, prior_likelihood, randomized_smiles_list, score, self._prior)
                        loss = loss.mean()
                        self._optimizer.zero_grad()
                        loss.backward()
                        self._optimizer.step()

                self._stats_and_chekpoint(score, start_time, step, smiles, score_summary,
                                          agent_likelihood, prior_likelihood,
                                          augmented_likelihood)

            self._logger.save_final_state(self._agent, self._diversity_filter)
            self._logger.log_out_input_configuration()
            self._logger.log_out_inception(self._inception)

            self.write_out_oracle_tracker()

        elif self.optimization_algorithm == "augmented_hill_climbing" or self.optimization_algorithm == "ahc":
            # original code-base: https://github.com/MorganCThomas/SMILES-RNN/blob/main/model/RL.py
            # original paper: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00646-z
            for step in range(self.config.n_steps):
                seqs, smiles, agent_likelihood = self._sample_unique_sequences(self._agent, self.config.batch_size)
                score_summary: FinalSummary = self._scoring_function.get_final_score_for_step(smiles, step)
                score = self._diversity_filter.update_score(score_summary, step)
                tensor_score = torch.tensor(score)
                sscore, sscore_idxs = tensor_score.sort(descending=True)

                # switch signs
                agent_likelihood = -agent_likelihood
                prior_likelihood = -self._prior.likelihood(seqs)

                augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(score)
                loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                # take the top_k
                loss = loss[sscore_idxs.data[:int(self.config.batch_size * self.top_k)]]
                # add experience replay
                loss, agent_likelihood = self._inception_filter(self._agent, loss, agent_likelihood, prior_likelihood,
                                                                smiles, score, self._prior, override=True)

                loss = loss.mean()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                self._stats_and_chekpoint(score, start_time, step, smiles, score_summary,
                                          agent_likelihood, prior_likelihood,
                                          augmented_likelihood)

            self._logger.save_final_state(self._agent, self._diversity_filter)
            self._logger.log_out_input_configuration()
            self._logger.log_out_inception(self._inception)

        elif self.optimization_algorithm == "best_agent_reminder" or self.optimization_algorithm == "bar":
            # built on https://github.com/MorganCThomas/SMILES-RNN/blob/main/model/RL.py
            # originally proposed for GraphINVENT from the following paper: https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00838
            # initialize the best Agent
            self.best_agent = deepcopy(self._agent)
            self.best_score_summary = None

            for step in range(self.config.n_steps):
                # sample batch from current Agent
                seqs, smiles, agent_likelihood = self._sample_unique_sequences(self._agent, self.config.batch_size)

                # sample batch from best Agent
                best_seqs, best_smiles, best_agent_likelihood = self._sample_unique_sequences(self.best_agent, self.config.batch_size)

                # score current Agent SMILES
                score_summary: FinalSummary = self._scoring_function.get_final_score_for_step(smiles, step)
                score = self._diversity_filter.update_score(score_summary, step)

                # score best Agent SMILES
                best_score_summary: FinalSummary = self._scoring_function.get_final_score_for_step(best_smiles, step)
                best_score = self._diversity_filter.update_score(best_score_summary, step)

                # compute loss between Prior and current Agent
                agent_likelihood = -agent_likelihood
                prior_likelihood = -self._prior.likelihood(seqs)
                augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(score)
                current_agent_loss = (1 - self.alpha) * torch.pow((augmented_likelihood - agent_likelihood), 2).mean()

                # compute loss between the best Agent and current Agent
                best_agent_likelihood = -best_agent_likelihood
                # this is the likelihood of the SMILES sampled by the *BAR Agent* as computed by the *current* Agent
                current_agent_likelihood = -self._agent.likelihood(best_seqs)
                best_augmented_likelihood = best_agent_likelihood + self.config.sigma * to_tensor(best_score)
                best_agent_loss = self.alpha * torch.pow((best_augmented_likelihood - current_agent_likelihood), 2)

                # add experience replay
                # pass the current Agent and current Agent's sampled SMILES because we want to update this Agent
                # the only thing we are passing that belongs to the best Agent is the best Agent loss
                best_agent_loss, best_agent_likelihood = self._inception_filter(self._agent, best_agent_loss, agent_likelihood, prior_likelihood,
                                                                                smiles, score, self._prior, override=True)

                # add experience replay before taking the mean
                best_agent_loss = best_agent_loss.mean()

                # compute the BAR loss
                BAR_loss = current_agent_loss + best_agent_loss

                self._optimizer.zero_grad()
                BAR_loss.backward()
                self._optimizer.step()

                self._stats_and_chekpoint(score, start_time, step, smiles, score_summary,
                                          agent_likelihood, prior_likelihood,
                                          augmented_likelihood)

                if step % self.update_frequency == 0:
                    if self.best_score_summary is not None:
                        penalized_best_average_score = np.mean(self.bar_scores_penalization(self.best_score_summary))
                        if np.mean(score) > penalized_best_average_score:
                            self.best_score_summary = score_summary
                            # new best Agent
                            self.best_agent = deepcopy(self._agent)
                    else:
                        self.best_score_summary = score_summary
                        # new best Agent
                        self.best_agent = deepcopy(self._agent)

            self._logger.save_final_state(self._agent, self._diversity_filter)
            self._logger.log_out_input_configuration()
            self._logger.log_out_inception(self._inception)

        else:
            raise ValueError("Optimization algorithm not available.")

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def _stats_and_chekpoint(self, score, start_time, step, smiles, score_summary: FinalSummary,
                             agent_likelihood, prior_likelihood, augmented_likelihood):
        self._margin_guard.adjust_margin(step)
        mean_score = np.mean(score)
        self._margin_guard.store_run_stats(agent_likelihood, prior_likelihood, augmented_likelihood, score)
        self._logger.timestep_report(start_time, self.config.n_steps, step, smiles,
                                     mean_score, score_summary, score,
                                     agent_likelihood, prior_likelihood, augmented_likelihood, self._diversity_filter)
        self._logger.save_checkpoint(step, self._diversity_filter, self._agent)

    def _sample_unique_sequences(self, agent, batch_size):
        seqs, smiles, agent_likelihood = agent.sample(batch_size)
        unique_idxs = get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood, smiles, score, prior, override=False):
        if self.augmented_memory and not override:
            if self._inception.configuration.augmented_memory_mode_collapse_guard:
                # if the below executes, Augmented Memory is effectively paused for this epoch
                self._inception.mode_collapse_guard()
            exp_smiles, exp_scores, exp_prior_likelihood = self._inception.augmented_memory_replay(prior)
        else:
            exp_smiles, exp_scores, exp_prior_likelihood = self._inception.sample()
        if len(exp_smiles) > 0:
            exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
            exp_augmented_likelihood = exp_prior_likelihood + self.config.sigma * exp_scores
            exp_loss = torch.pow((to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        self._inception.add(smiles, score, prior_likelihood)

        return loss, agent_likelihood

    def reset(self, reset_countdown=0):
        model_type_enum = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()
        actor_config = ModelConfiguration(model_type_enum.DEFAULT, model_regime.TRAINING,
                                          self.config.agent)
        self._agent = GenerativeModel(actor_config)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)
        self._logger.log_message("Resetting Agent")
        self._logger.log_message(f"Adjusting sigma to: {self.config.sigma}")
        return reset_countdown

    def bar_scores_penalization(self, score_summary: FinalSummary) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for idx in score_summary.valid_idxs:
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[idx])
            scaffold = self._chemistry.get_scaffold(smile)

            if scores[idx] >= self._diversity_filter.parameters.minscore:
                scores[idx] = self._diversity_filter._penalize_score(scaffold, scores[idx])

        return scores

    def update_oracle_tracker(self, step: int, oracle_calls: int, total_score: np.array, smiles: np.array):
        step = list(np.full_like(smiles, step))
        oracle_calls = list(np.full_like(smiles, oracle_calls))
        total_score = list(total_score)
        smiles = list(smiles)

        df = pd.DataFrame({"step": step, "oracle_calls": oracle_calls, "total_score": total_score, "smiles": smiles})
        self.oracle_tracker = pd.concat([self.oracle_tracker, df])

    def write_out_oracle_tracker(self):
        self.oracle_tracker.to_csv('oracle_tracker.csv')

