from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    seed: int = 53
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 64
    epochs: int = 1
    max_batches: int = 3
    validate_every: int = 1
    learning_rate: float = 0.001
    dataset_names = ["shakespeare"]
    llm_validation_steps = 1
