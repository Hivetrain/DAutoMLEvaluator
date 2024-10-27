import torch
from typing import Callable
import logging

class MetricsEvaluator:
    @staticmethod
    def safe_evaluate(func: Callable, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Safely evaluate a loss function."""
        try:
            loss = func(outputs, labels)
            if loss is None or not torch.is_tensor(loss) or not torch.isfinite(loss).all():
                return torch.tensor(float('inf'), device=outputs.device)
            return loss.mean() if loss.ndim > 0 else loss
        except Exception as e:
            logging.error(f"Error in loss calculation: {str(e)}")
            return torch.tensor(float('inf'), device=outputs.device)
