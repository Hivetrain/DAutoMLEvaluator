from dataclasses import dataclass
import torch

SUPPORTED_DATASETS = [
    "mnist",
    "cifar10", 
    "cifar100",
    "imagenet",
    "shakespeare",
    "fineweb"
]

SUPPORTED_ARCHITECTURES = [
    "mlp",
    "cnn",
    "resnet",
    "vit", 
    "gpt",
    "transformer"
]

@dataclass
class TrainingConfig:
    seed: int = 53
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 1
    epochs: int = 1
    max_batches: int = 90
    validate_every: int = 10
    learning_rate: float = 0.001
    dataset_names = ["fineweb"]
    architectures = {
        "cifar10": ["cnn", "vit-small"],
        "cifar100": ["cnn", "resnet", "vit-small","vit-base" ],
        "imagenet": ["cnn", "resnet", "vit-small","vit-base" ],
    }
    llm_validation_steps = 50
