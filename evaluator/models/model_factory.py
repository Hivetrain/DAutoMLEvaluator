import torch
from evaluator.eval_config import TrainingConfig

import torch.nn as nn
import torch 

from transformers import GPT2Config, GPT2LMHeadModel
from evaluator.data.data import load_datasets
from typing import Union, Tuple

class BaselineNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class EvolvableNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, evolved_activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.evolved_activation = evolved_activation

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.evolved_activation(x)
        x = self.fc2(x)
        return x
    
class EvolvedLoss(torch.nn.Module):
    def __init__(self, genome, device = "cpu"):
        super().__init__()
        self.genome = genome
        self.device = device

    def forward(self, outputs, targets):
        #outputs = outputs.detach().float().requires_grad_()#.to(self.device)
        #targets = targets.detach().float().requires_grad_()#.to(self.device)
        
        memory = self.genome.memory
        memory.reset()
        
        memory[self.genome.input_addresses[0]] = outputs
        memory[self.genome.input_addresses[1]] = targets
        for i, op in enumerate(self.genome.gene):
            func = self.genome.function_decoder.decoding_map[op][0]
            input1 = memory[self.genome.input_gene[i]]#.to(self.device)
            input2 = memory[self.genome.input_gene_2[i]]#.to(self.device)
            constant = torch.tensor(self.genome.constants_gene[i], requires_grad=True)#.to(self.device)
            constant_2 = torch.tensor(self.genome.constants_gene_2[i], requires_grad=True)#.to(self.device)
            
            output = func(input1, input2, constant, constant_2, self.genome.row_fixed, self.genome.column_fixed)
            if output is not None:
                memory[self.genome.output_gene[i]] = output

        loss = memory[self.genome.output_addresses[0]]
        return loss
    
class GPTWrapper(nn.Module):
    def __init__(self, vocab_size=50257):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=256,     
            n_ctx=256,          
            n_embd=32,         
            n_layer=2,          
            n_head=2,           
            n_inner=32,        
            bos_token_id=50256,
            eos_token_id=50256,
        )
        self.model = GPT2LMHeadModel(config)
        
    def forward(self, x):
        # Return just the logits from the output
        return self.model(x).logits
    
class BabyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, num_heads=2, sequence_length=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        
        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_blocks[0],
            num_layers=len(transformer_blocks)
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        B, T = x.shape
        
        mask = self.generate_square_subsequent_mask(T).to(x.device)
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb
        
        x = x.transpose(0, 1)  # TransformerEncoder expects seq_len first
        x = self.transformer(x, mask=mask)  # 
        x = x.transpose(0, 1)  # Back to batch first
        
        logits = self.fc_out(x)
        return logits


def get_imagenet_model(
    num_classes: int = 1000,
    pretrained: bool = False
) -> nn.Module:
    """
    Returns a ResNet50 model suitable for ImageNet
    """
    if pretrained:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_shakespeare_model(
    embed_size: int = 384,
    num_heads: int = 6,
    num_layers: int = 6,
    **kwargs
) -> nn.Module:
    """
    Returns a small GPT model suitable for Shakespeare text
    """
    return BabyGPT(
        vocab_size=85,
        embedding_dim=embed_size,
        num_heads=num_heads,
        num_layers=num_layers
    )

def get_mlp(input_size: int, output_size: int, hidden_size: int = 128, dropout: float = 0.2) -> nn.Module:
    """Generic MLP that works with any dataset dimensions."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    )

def get_cnn(input_channels: int, output_size: int, base_channels: int = 32) -> nn.Module:
    """Generic CNN that works with any image dataset."""
    return nn.Sequential(
        nn.Conv2d(input_channels, base_channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(base_channels * 2 * 7 * 7, 128),  # This assumes 28x28 input, would need adjustment
        nn.ReLU(),
        nn.Linear(128, output_size)
    )

def get_baby_gpt(vocab_size: int, embed_size: int = 384, num_heads: int = 6, num_layers: int = 6) -> nn.Module:
    """Generic GPT model that works with any text dataset."""
    return BabyGPT(
        vocab_size=vocab_size,
        embedding_dim=embed_size,
        num_heads=num_heads,
        num_layers=num_layers
    )


def get_gpt2_model(
    vocab_size: int = 50257,
    **kwargs
) -> nn.Module:
    """
    Returns a small GPT-2 model suitable for Fineweb dataset
    """
    return GPTWrapper(vocab_size)

# Dictionary mapping dataset names to their model creators
# Map architecture names to their factory functions
ARCHITECTURE_MAP = {
    'mlp': get_mlp,
    'cnn': get_cnn,
    'gpt': get_baby_gpt,
    'resnet': get_imagenet_model  # Already generalized in original code
}

def get_model_for_dataset(dataset_name: str, architecture: str = 'mlp', dataset_spec = None, **kwargs) -> nn.Module:
    """Get the appropriate model for a given dataset and architecture.
    
    Args:
        dataset_name: Name of the dataset
        architecture: Name of the architecture ('mlp', 'cnn', 'gpt', 'resnet')
        dataset_spec: DatasetSpec object containing dataset parameters
        **kwargs: Additional arguments to pass to the model creator
        
    Returns:
        nn.Module: The initialized model
    """
    if architecture not in ARCHITECTURE_MAP:
        raise ValueError(f"Architecture {architecture} not recognized")
    
    if dataset_spec is None:
        # Get dataset spec if not provided
        dataset_spec = load_datasets([dataset_name])[0]
    
    # Get the appropriate model creator
    model_creator = ARCHITECTURE_MAP[architecture]
    
    # Configure architecture-specific parameters
    if architecture == 'mlp':
        return model_creator(
            input_size=dataset_spec.input_size,
            output_size=dataset_spec.output_size,
            hidden_size=dataset_spec.hidden_size,
            **kwargs
        )
    elif architecture == 'cnn':
        # Assume image data with channels
        if isinstance(dataset_spec.input_size, tuple):
            input_channels = dataset_spec.input_size[0]
        else:
            input_channels = 1  # Default to single channel
        return model_creator(
            input_channels=input_channels,
            output_size=dataset_spec.output_size,
            **kwargs
        )
    elif architecture == 'gpt':
        return model_creator(
            vocab_size=dataset_spec.output_size,
            embed_size=dataset_spec.hidden_size,
            **kwargs
        )
    else:  # resnet or other architectures
        return model_creator(
            num_classes=dataset_spec.output_size,
            **kwargs
        )