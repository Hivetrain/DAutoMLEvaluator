import torch
from evaluator.eval_config import TrainingConfig

class ModelFactory:
    @staticmethod
    def create_mnist_model(config: TrainingConfig) -> torch.nn.Module:
        """Create and return a simple MNIST model."""
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        ).to(config.device)
