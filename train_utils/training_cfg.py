import dataclasses

@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.25
    batch_size: int = 128
    learning_rate: float = 0.001
    num_workers: int = 2
    prefetch_factor: int = 2
    num_epochs: int  = 40