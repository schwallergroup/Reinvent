from dataclasses import dataclass


@dataclass
class ReinforcementLearningConfiguration:
    prior: str
    agent: str
    n_steps: int = 300
    sigma: int = 128
    learning_rate: float = 0.0001
    batch_size: int = 128
    margin_threshold: int = 50
    optimization_algorithm: str = "augmented_memory"
    augmented_memory: bool = True
    augmentation_rounds: int = 2
    selective_memory_purge: bool = True


