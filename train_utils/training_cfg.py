import dataclasses

@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.25
    batch_size: int = 64
    init_learning_rate: float = 0.001
    min_learning_rate: float = 0.00001
    lr_scheduler_epochs_skip: int = 4
    weight_decay = 0.25
    num_workers: int = 2
    prefetch_factor: int = 2
    num_epochs: int  = 30