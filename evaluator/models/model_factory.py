import torch
from evaluator.eval_config import TrainingConfig

import torch.nn as nn
import torch 

from transformers import GPT2Config, GPT2LMHeadModel
from evaluator.data.data import load_datasets
from typing import Tuple, Union

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

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # MLP head (use [CLS] token)
        x = self.head(x[:, 0])
        return x

def get_vit(
    input_size: Union[Tuple[int, int, int], int],
    output_size: int,
    variant: str = 'base',
    **kwargs
) -> nn.Module:
    """Create a Vision Transformer model.
    
    Args:
        input_size: Either tuple of (channels, height, width) or flat size
        output_size: Number of classes
        variant: 'small', 'base', or 'large'
    """
    # Handle input size formatting
    if isinstance(input_size, int):
        # Infer image dimensions based on flat size
        if input_size == 28 * 28:  # MNIST
            in_chans, height, width = 1, 28, 28
        elif input_size == 32 * 32 * 3:  # CIFAR
            in_chans, height, width = 3, 32, 32
        else:
            raise ValueError(f"Cannot infer image dimensions from flat size {input_size}")
    else:
        in_chans, height, width = input_size

    # ViT configurations following paper specifications
    configs = {
        'small': {
            'patch_size': 4 if height <= 32 else 16,  # Smaller patch size for small images
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
        },
        'base': {
            'patch_size': 4 if height <= 32 else 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
        },
        'large': {
            'patch_size': 4 if height <= 32 else 16,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
        }
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown ViT variant: {variant}")
    
    config = configs[variant]
    
    # Ensure image size is compatible with patch size
    assert height == width, "Image must be square"
    assert height % config['patch_size'] == 0, f"Image size {height} must be divisible by patch size {config['patch_size']}"
    
    return ViT(
        img_size=height,
        patch_size=config['patch_size'],
        in_chans=in_chans,
        num_classes=output_size,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        **kwargs
    )



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

def get_cnn(input_size: Tuple[int, int, int], output_size: int, base_channels: int = 32) -> nn.Module:
    """Generic CNN that works with any image dataset.
    
    Args:
        input_size: Tuple of (channels, height, width)
        output_size: Number of classes
        base_channels: Number of base channels (will be doubled in deeper layers)
    """
    in_channels, height, width = input_size
    
    def calc_conv_output(size, kernel=3, stride=1, padding=1):
        return ((size + 2 * padding - kernel) // stride) + 1
    
    def calc_pool_output(size, kernel=2, stride=2):
        return size // stride
    
    # Calculate size after first conv + pool
    conv1_size = calc_conv_output(height)
    pool1_size = calc_pool_output(conv1_size)
    
    # Calculate size after second conv + pool
    conv2_size = calc_conv_output(pool1_size)
    pool2_size = calc_pool_output(conv2_size)
    
    # Calculate flattened size
    flat_size = (base_channels * 2) * pool2_size * pool2_size

    return nn.Sequential(
        nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(flat_size, 128),
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
    'resnet': get_imagenet_model,  # Already generalized in original code
    'vit-small': lambda *args, **kwargs: get_vit(*args, variant='small', **kwargs),
    'vit-base': lambda *args, **kwargs: get_vit(*args, variant='base', **kwargs),
    'vit-large': lambda *args, **kwargs: get_vit(*args, variant='large', **kwargs)
    }

def get_model_for_dataset(dataset_name: str, architecture: str = 'mlp', dataset_spec = None, **kwargs) -> nn.Module:
    """Get the appropriate model for a given dataset and architecture."""
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
    elif architecture in ['cnn', 'vit-small', 'vit-base', 'vit-large']:
        # Handle case where input_size is a single number
        if isinstance(dataset_spec.input_size, int):
            if dataset_name == 'mnist':
                input_size = (1, 28, 28)  # MNIST is single channel
            elif dataset_name in ['cifar10', 'cifar100']:
                input_size = (3, 32, 32)  # CIFAR is RGB
            else:
                raise ValueError(f"Cannot determine image dimensions for dataset {dataset_name}")
        else:
            input_size = dataset_spec.input_size
            
        return model_creator(
            input_size=input_size,
            output_size=dataset_spec.output_size,
            **kwargs
        )
    else:  # gpt, resnet, or other architectures
        return model_creator(
            num_classes=dataset_spec.output_size,
            **kwargs
        )