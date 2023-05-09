import numpy as np
import torch

from running_modes.automated_curriculum_learning.learning_strategy.base_single_query_learning_strategy import \
    BaseSingleQueryLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration

from reinvent_chemistry.conversions import Conversions

_chemistry = Conversions()


class DAPSingleQueryStrategy(BaseSingleQueryLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)

        self._sigma = self._configuration.parameters.get("sigma", 120)

    def _calculate_loss(self, smiles, sampled_sequences: np.ndarray, score, actor_nlls, inception, agent):
        critic_nlls = self.critic_model.likelihood(sampled_sequences)
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -actor_nlls
        augmented_nlls = negative_critic_nlls + self._sigma * self._to_tensor(score)
        loss = torch.pow((augmented_nlls - negative_actor_nlls), 2)
        loss, agent_likelihood = self._inception_filter(agent, loss, negative_actor_nlls, negative_critic_nlls,
                                                        self._sigma, smiles, score, inception, override=True)
        loss = loss.mean()

        return loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls

    def _calculate_augmented_loss(self, agent, score, smiles, inception, prior, augmented_memory):
        """implementation of Double Reinforcement Learning: https://arxiv.org/pdf/2210.12458.pdf"""
        # get randomized SMILES
        randomized_smiles_list = _chemistry.get_randomized_smiles(smiles, prior)
        # obtain critic (prior) likelihoods of randomized SMILES
        negative_critic_nlls = -self.critic_model.likelihood_smiles(randomized_smiles_list)
        # obtain actor (agent) likelihood of randomized SMILES
        negative_actor_nlls = -agent.likelihood_smiles(randomized_smiles_list)
        # compute augmented likelihood with the "new" prior likelihood using randomized SMILES
        augmented_nlls = negative_critic_nlls + self._sigma * self._to_tensor(score)
        # compute loss
        loss = torch.pow((augmented_nlls - negative_actor_nlls), 2)
        # experience replay using randomized SMILES
        loss, agent_likelihood = self._inception_filter(agent, loss, negative_actor_nlls, negative_critic_nlls,
                                                        self._sigma, randomized_smiles_list, score, inception, prior, augmented_memory)
        loss = loss.mean()

        return loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood, sigma, smiles, score, inception, prior=None, augmented_memory=False, override=False):
        if inception:
            if augmented_memory and not override:
                print('we are doing augmented memory')
                exp_smiles, exp_scores, exp_prior_likelihood = inception.augmented_memory_replay(prior)
            else:
                exp_smiles, exp_scores, exp_prior_likelihood = inception.sample()
            if len(exp_smiles) > 0:
                exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
                exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_scores
                exp_loss = torch.pow((self._to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            inception.add(smiles, score, prior_likelihood)
        return loss, agent_likelihood
