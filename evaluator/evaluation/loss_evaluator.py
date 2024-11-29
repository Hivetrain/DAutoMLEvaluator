import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Callable
from tqdm import tqdm
import logging
from evaluator.eval_config import TrainingConfig
from evaluator.evaluation.metrics import MetricsEvaluator

class LossEvaluator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_evaluator = MetricsEvaluator()

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

    def train_and_evaluate(
        self,
        model: torch.nn.Module,
        loss_function: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader, 
        num_classes: int = 10,
        metric_type: str = "loss"
    ) -> Dict[str, List[float]]:
        """Train the model and evaluate its performance."""
        metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'batch_numbers': []
        }
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        batch_counter = 0
        
        model.train()
        for epoch in range(self.config.epochs):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
                for inputs, targets in train_loader:
                    batch_counter += 1
                    if batch_counter > self.config.max_batches:
                        break
                    
                    metrics = self._training_step(
                        model, loss_function, optimizer,
                        inputs, targets, metrics, batch_counter, num_classes
                    )
                    
                    if batch_counter % self.config.validate_every == 0:
                        metrics = self._validation_step(
                            model, val_loader, metrics, batch_counter, metric_type
                        )
                        pbar.set_postfix({
                            'Loss': f"{metrics['train_loss'][-1]:.4f}",
                            'Val Acc': f"{metrics['val_accuracy'][-1]:.4f}"
                        })
                    
                    pbar.update(1)
        
        return metrics
    
    def _training_step(
        self,
        model: torch.nn.Module,
        loss_function: Callable,
        optimizer: torch.optim.Optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        metrics: Dict[str, List[float]],
        batch_counter: int,
        num_classes: int = 10
    ) -> Dict[str, List[float]]:
        """Perform a single training step."""
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        try:
            loss = self.safe_evaluate(loss_function, outputs, targets_one_hot)
        except:
            breakpoint()
        loss.backward()
        optimizer.step()
        
        metrics['train_loss'].append(loss.item())
        return metrics

    def _validation_step(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        metrics: Dict[str, List[float]],
        batch_counter: int,
        metric_type: str = 'loss'  # New parameter to choose metric type
    ) -> Dict[str, List[float]]:
        """Perform validation and update metrics."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for idx, (val_inputs, val_targets) in tqdm(enumerate(val_loader)):
                val_inputs = val_inputs.to(self.config.device)
                val_targets = val_targets.to(self.config.device)
                val_outputs = model(val_inputs)
                if len(val_outputs.shape) == 3:
                    if idx > self.config.llm_validation_steps: #max validation config
                        break
                    _, predicted = val_outputs.max(dim=-1)
                    total += val_targets.numel()  # Count all elements
                    correct += predicted.eq(val_targets).sum().item()

                    loss = F.cross_entropy(
                    val_outputs.view(-1, val_outputs.size(-1)), 
                    val_targets.view(-1)
                    )
                    total_loss += loss.item()
                else:
                    _, predicted = val_outputs.max(1)
                    total += val_targets.size(0)
                    correct += predicted.eq(val_targets).sum().item()

                    loss = F.cross_entropy(val_outputs, val_targets)
                    total_loss += loss.item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        if metric_type == 'accuracy':
            metrics['val_accuracy'].append(accuracy)
        else:
            metrics['val_accuracy'].append(loss) #FIXME Hack 
        metrics['batch_numbers'].append(batch_counter)
        model.train()
        return metrics
