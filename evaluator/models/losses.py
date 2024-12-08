import torch
device = torch.device("cpu")
import random

one = torch.tensor(1.0, device=device)
half = torch.tensor(0.5, device=device)
zero = torch.tensor(0.0, device=device)
e = torch.tensor(2.718281828459045, device=device)

def safe_div(x, y):
    epsilon = 1e-8
    return x / (y + epsilon)

def safe_add(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x + y

def safe_sub(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x - y

def safe_mul(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x * y


def safe_sigmoid(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.sigmoid(x)

def safe_relu(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.relu(x)

def safe_tanh(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.tanh(x)

def safe_log(x):
    return torch.log(torch.abs(x) + 1e-8)

def safe_sqrt(x):
    return torch.sqrt(torch.abs(x))

def safe_exp(x):
    return torch.exp(torch.clamp(x, -100, 100))

def generate_random():
    return torch.tensor(random.uniform(-1, 1), device=device)

sin = torch.sin
sigmoid = torch.sigmoid
relu = torch.relu
std = torch.std
tanh = torch.tanh
cos = torch.cos
square = torch.square
pi = torch.tensor(3.141592653589793, device=device)
round = torch.round

def square(x):
    return torch.pow(x, 2)

def cube(x):
    return torch.pow(x, 3)

def loss(x, y):
    return safe_sub(safe_add(safe_exp(safe_sub(safe_add(safe_exp(one), sin(safe_mul(y, safe_sub(safe_log(y), sigmoid(safe_mul(safe_mul(half, x), safe_mul(safe_exp(sin(half)), x))))))), safe_exp(safe_exp(one)))), safe_add(safe_exp(safe_sub(safe_add(safe_exp(safe_div(safe_mul(safe_div(half, one), safe_exp(half)), one)), relu(std(safe_sub(safe_sub(safe_exp(one), safe_sub(safe_log(y), sigmoid(safe_mul(safe_mul(half, x), safe_mul(safe_exp(sin(half)), x))))), safe_sub(safe_log(y), safe_sub(safe_mul(safe_add(y, one), safe_log(y)), cos(safe_sub(safe_log(y), safe_div(x, safe_exp(sin(half))))))))))), safe_exp(safe_exp(one)))), safe_sub(safe_add(safe_mul(square(y), square(pi)), tanh(std(safe_sub(safe_sub(x, safe_sub(safe_log(y), sigmoid(safe_mul(safe_mul(half, x), safe_mul(safe_mul(safe_add(y, one), safe_log(y)), x))))), safe_sub(safe_log(y), safe_sub(safe_log(y), cos(safe_sub(safe_log(y), safe_div(x, safe_exp(sin(half))))))))))), safe_div(e, cos(y))))), cube(safe_add(y, round(safe_mul(safe_log(y), pi)))))

def batch_loss(x, y, batch_size=85):

    total_loss = 0
    
    # Process in batches
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        
        # Calculate loss for this batch
        batch_loss = loss(batch_x, batch_y)
        
        # Normalize the loss by number of batches to maintain scale
        batch_loss = batch_loss / ((len(x) + batch_size - 1) // batch_size)
        
        # Accumulate gradients
        batch_loss.backward()
        
        total_loss += batch_loss.item()
    
    
    return total_loss