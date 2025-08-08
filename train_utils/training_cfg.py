import dataclasses

@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.3
    batch_size: int = 8
    learning_rate: float = 0.0001
    num_workers: int = 2
    prefetch_factor: int = 2
    num_epochs: int  = 20