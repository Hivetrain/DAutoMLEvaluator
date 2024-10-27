import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
from evaluator.eval_config import TrainingConfig

class DataManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and return train and validation dataloaders."""
        train_data = datasets.MNIST('../data', train=True, download=True, transform=self.transform)
        val_data = datasets.MNIST('../data', train=False, transform=self.transform)
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        return train_loader, val_loader
