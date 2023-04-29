from dataclasses import dataclass


@dataclass
class ReinforcementLearningConfiguration:
    prior: str
    agent: str
    n_steps: int = 3000
    sigma: int = 120
    learning_rate: float = 0.0001
    batch_size: int = 128
    margin_threshold: int = 50
    optimization_algorithm: str = "augmented_memory"
    double_loop_augment: bool = True
    augmented_memory: bool = False
    augmentation_rounds: int = 10
    selective_memory_purge: bool = True


