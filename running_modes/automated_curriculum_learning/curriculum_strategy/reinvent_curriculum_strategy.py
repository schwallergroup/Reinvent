import time
from typing import List, Tuple

import numpy as np
import torch
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.actions.reinvent_sample_model import ReinventSampleModel
from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.dto import SampledBatchDTO, CurriculumOutcomeDTO, TimestepDTO


import pandas as pd
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


class ReinventCurriculumStrategy(BaseCurriculumStrategy):

    def run(self) -> CurriculumOutcomeDTO:
        step_counter = 0
        self.disable_prior_gradients()

        for item_id, sf_configuration in enumerate(self._parameters.curriculum_objectives):
            if item_id == 1:
                self.inception.memory = pd.DataFrame({})
            start_time = time.time()
            scoring_function = self._setup_scoring_function(item_id)
            step_counter = self.promote_agent(agent=self._agent, scoring_function=scoring_function,
                                              step_counter=step_counter, start_time=start_time,
                                              merging_threshold=sf_configuration.score_threshold)
            self.save_and_flush_memory(agent=self._agent, memory_name=f"_merge_{item_id}")
        is_successful_curriculum = step_counter < self._parameters.max_num_iterations
        outcome_dto = CurriculumOutcomeDTO(self._agent, step_counter, successful_curriculum=is_successful_curriculum)

        return outcome_dto

    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step:int, start_time: float) -> float:
        # 1. Sampling
        sampled = self._sampling(agent)
        # 2. Scoring
        score, score_summary = self._scoring(scoring_function, sampled.smiles, step)
        # 3. Updating
        agent_likelihood, prior_likelihood, augmented_likelihood = self._updating(sampled, score, self.inception, agent)
        # 4. Augment SMILES and update Agent again
        if self.augmented_memory:
            # purge memory first
            if self.selective_memory_purge:
                self._selective_memory_purge(sampled.smiles, score)
            for _ in range(self.augmentation_rounds):
                agent_likelihood, prior_likelihood, augmented_likelihood = self._updating_augmented(agent, score, sampled.smiles, self.inception, self._prior, self.augmented_memory)
        # 5. Logging
        self._logging(agent=agent, start_time=start_time, step=step,
                      score_summary=score_summary, agent_likelihood=agent_likelihood,
                      prior_likelihood=prior_likelihood, augmented_likelihood=augmented_likelihood)

        score = score.mean()
        return score

    def _sampling(self, agent) -> SampledBatchDTO:
        sampling_action = ReinventSampleModel(agent, self._parameters.batch_size, self._logger)
        sampled_sequences = sampling_action.run()
        return sampled_sequences

    def _scoring(self, scoring_function, smiles: List[str], step) -> Tuple[np.ndarray, FinalSummary]:
        score_summary = scoring_function.get_final_score_for_step(smiles, step)
        dto = UpdateDiversityFilterDTO(score_summary, [], step)
        score = self._diversity_filter.update_score(dto)
        return score, score_summary

    def _updating(self, sampled, score, inception, agent):
        agent_likelihood, prior_likelihood, augmented_likelihood = \
            self.learning_strategy.run(sampled, score, inception, agent)
        return agent_likelihood, prior_likelihood, augmented_likelihood

    def _updating_augmented(self, agent, score, smiles, inception, prior, augmented_memory):
        agent_likelihood, prior_likelihood, augmented_likelihood = self.learning_strategy.run_augmented(agent, score, smiles, inception, prior, augmented_memory)
        return agent_likelihood, prior_likelihood, augmented_likelihood

    def _logging(self, agent: GenerativeModelBase, start_time: float, step: int, score_summary: FinalSummary,
                  agent_likelihood: torch.tensor, prior_likelihood: torch.tensor, augmented_likelihood: torch.tensor):
        report_dto = TimestepDTO(start_time, self._parameters.max_num_iterations, step, score_summary,
                                 agent_likelihood, prior_likelihood, augmented_likelihood)
        self._logger.timestep_report(report_dto, self._diversity_filter, agent)

    def save_and_flush_memory(self, agent, memory_name: str):
        self._logger.save_merging_state(agent, self._diversity_filter, name=memory_name)
        self._diversity_filter = DiversityFilter(self._parameters.diversity_filter)

    @staticmethod
    def _initialize_augmented_smiles_tracker(initial_smiles):
        augmented_smiles_tracker = {}
        for smiles in initial_smiles:
            augmented_smiles_tracker[smiles] = []

        return augmented_smiles_tracker

    def _selective_memory_purge(self, smiles, score):
        # TODO: move this to inception and implement it in CL
        zero_score_indices = np.where(score == 0.)[0]
        if len(zero_score_indices) > 0:
            smiles_to_purge = smiles[zero_score_indices]
            scaffolds_to_purge = [self.get_scaffold(smiles) for smiles in smiles_to_purge]
            purged_memory = deepcopy(self.inception.memory)
            purged_memory['scaffolds'] = purged_memory['smiles'].apply(self.get_scaffold)
            purged_memory = purged_memory.loc[~purged_memory['scaffolds'].isin(scaffolds_to_purge)]
            purged_memory.drop('scaffolds', axis=1, inplace=True)
            self.inception.memory = purged_memory
        else:
            return

    @staticmethod
    def get_scaffold(smiles):
        # TODO: this function exists - remove redundancy
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                scaffold = GetScaffoldForMol(mol)
                return Chem.MolToSmiles(scaffold)
            except Exception:
                return ''
        else:
            return ''

