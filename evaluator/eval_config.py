from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    seed: int = 53
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 64
    epochs: int = 1
    max_batches: int = 90
    validate_every: int = 10
    learning_rate: float = 0.001
    dataset_names = ["cifar10", "mnist"]
