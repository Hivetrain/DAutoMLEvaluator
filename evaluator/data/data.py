from dataclasses import dataclass
import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Any, List, Tuple, Optional, Union
import os 

@dataclass
class DatasetSpec:
    name: str
    input_size: int
    output_size: int
    hidden_size: int = 128
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    training_iterations: int = 1
    weight: float = 1.0  

def get_mnist_loaders(
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        './data', 
        train=False,
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

def get_cifar10_loaders(
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        './data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    val_dataset = datasets.CIFAR10(
        './data', 
        train=False,
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

def get_cifar100_loaders(
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), 
            (0.2675, 0.2565, 0.2761)
        )
    ])
    
    train_dataset = datasets.CIFAR100(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.CIFAR100(
        './data',
        train=False,
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

def get_imagenet_1k_loaders(
    data_dir: str = './data/imagenet',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    ImageNet-1K data loaders with standard augmentation
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = datasets.ImageNet(
        data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageNet(
        data_dir,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

class ShakespeareDataset(Dataset):
    def __init__(
        self,
        text_path: str,
        seq_length: int = 256,
        train: bool = True
    ):
        text_url = "https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt"
        if not os.path.exists(text_path):
            response = requests.get(text_url)
            with open(text_path, "wb") as file:
                file.write(response.content)

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character level dictionary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        
        # Split into train/val (90/10)
        n = int(0.9 * len(data))
        self.data = data[:n] if train else data[n:]
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - 1
        
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    



def get_shakespeare_loaders(
    text_path: str = './data/shakespeare.txt',
    batch_size: int = 64,
    seq_length: int = 256,
    num_workers: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Shakespeare dataset loaders for character-level language modeling
    """
    train_dataset = ShakespeareDataset(
        text_path=text_path,
        seq_length=seq_length,
        train=True
    )
    
    val_dataset = ShakespeareDataset(
        text_path=text_path,
        seq_length=seq_length,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, train_dataset.vocab_size

class FinewebDataset(Dataset):
    def __init__(self, data_dir: str, max_tokens: int = 33_000_000, seq_length: int = 256, train: bool = True):
        """
        Args:
            data_dir: Directory containing the fineweb binary files
            max_tokens: Maximum number of tokens to load (default 33M)
            seq_length: Length of sequences to return
            train: If True, load training data, else load validation data
        """
        self.seq_length = seq_length
        
        # Load the binary data
        if train:
            filepath = os.path.join(data_dir, 'fineweb10B', 'fineweb_train_000001.bin')
        else:
            filepath = os.path.join(data_dir, 'fineweb10B', 'fineweb_val_000000.bin')
            
        # Load only the first max_tokens
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')[:max_tokens]
        
    def __len__(self):
        return len(self.data) - self.seq_length - 1
        
    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx:idx + self.seq_length + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_fineweb_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    seq_length: int = 256,
    max_tokens: int = 33_000_000,
    num_workers: int = 1
) -> tuple[DataLoader, DataLoader, int]:
    """
    Fineweb dataset loaders for language modeling
    """
    train_dataset = FinewebDataset(
        data_dir=data_dir,
        max_tokens=max_tokens,
        seq_length=seq_length,
        train=True
    )
    
    val_dataset = FinewebDataset(
        data_dir=data_dir,
        max_tokens=max_tokens // 10,  # Smaller validation set
        seq_length=seq_length,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, 50257  # GPT-2 vocab size


def load_datasets(dataset_names: Union[str, List[str]], batch_size: int = 32) -> List[DatasetSpec]:
    """
    Load specified datasets based on input names.
    
    Args:
        dataset_names: Single dataset name or list of dataset names
        batch_size: Batch size for data loaders
    
    Returns:
        List of DatasetSpec objects for requested datasets
    """
    # Convert single string to list for consistent processing
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Dictionary mapping dataset names to their loader functions and specs
    dataset_configs = {
        "mnist": {
            "loader": get_mnist_loaders,
            "input_size": 28*28,
            "output_size": 10,
            "weight": 1.0
        },
        "cifar10": {
            "loader": get_cifar10_loaders,
            "input_size": 32*32*3,
            "output_size": 10,
            "weight": 1.0
        },
        "cifar100": {
            "loader": get_cifar100_loaders,
            "input_size": 32*32*3,
            "output_size": 100,
            "weight": 1.0
        },
        "imagenet": {
            "loader": get_imagenet_1k_loaders,
            "input_size": (3, 224, 224),
            "output_size": 1000,
            "weight": 2.0,
            "learning_rate": 0.1
        },
        "shakespeare": {
            "loader": get_shakespeare_loaders,
            "input_size": 32,
            "output_size": None,  # Will be set after loading
            "hidden_size": 32,
            "learning_rate": 3e-4,
            "weight": 10.0
        },
        "fineweb": {
            "loader": get_fineweb_loaders,
            "input_size": 256,  # sequence length
            "output_size": 50257,  # GPT-2 vocab size
            "hidden_size": 256,
            "learning_rate": 3e-4,
            "weight": 10.0
        }
    }
    
    dataset_specs = []
    
    for name in dataset_names:
        if name not in dataset_configs:
            raise ValueError(f"Dataset '{name}' not recognized. Available datasets: {list(dataset_configs.keys())}")
            
        config = dataset_configs[name]
        
        if name in ["shakespeare", "fineweb"]:
            train_loader, val_loader, vocab_size = config["loader"](batch_size=batch_size)
            config["output_size"] = vocab_size
        else:
            train_loader, val_loader = config["loader"](batch_size=batch_size)
        
        spec = DatasetSpec(
            name=name,
            input_size=config["input_size"],
            output_size=config["output_size"],
            train_loader=train_loader,
            val_loader=val_loader,
            hidden_size=config.get("hidden_size", 128),
            learning_rate=config.get("learning_rate", 0.001),
            weight=config.get("weight", 1.0)
        )
        
        dataset_specs.append(spec)
    
    return dataset_specs