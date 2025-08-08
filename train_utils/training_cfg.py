import dataclasses

@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.3
    batch_size: int = 2
    learning_rate: float = 0.001
    num_workers: int = 1
    prefetch_factor: int = 1
