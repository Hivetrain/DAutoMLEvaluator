import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
from evaluator.eval_config import TrainingConfig
from evaluator.data.data import load_datasets

class DataManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_data(self, dataset: str = "mnist") -> Tuple[DataLoader, DataLoader]:
        """Load and return train and validation dataloaders."""

        dataset_spec = load_datasets([dataset])[0]
            
        return dataset_spec.train_loader, dataset_spec.val_loader
