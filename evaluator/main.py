from datetime import datetime
import logging

from typing import Dict, Any, List
import os
import torch
from torch.utils.data import DataLoader

from evaluator.eval_config import TrainingConfig
from evaluator.data.data_manager import DataManager
from evaluator.evaluation.loss_evaluator import LossEvaluator
from evaluator.visualization.visualizer import Visualizer
from evaluator.utils.deap_utils import DeapToolboxFactory
from evaluator.utils.results_handler import ResultsHandler
from dml.gene_io import load_individual_from_json
from evaluator.models.model_factory import get_model_for_dataset

from tqdm import tqdm 

class LossFunctionEvaluator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.evaluator = LossEvaluator(config)
        self.visualizer = Visualizer()
        self.toolbox, self.pset = DeapToolboxFactory.create_toolbox()
        self.results_handler = ResultsHandler()  # Add this line

    def evaluate_loss_functions(self, json_folder: str) -> None:
        """Evaluate multiple loss functions from JSON files."""
        for dataset in self.config.dataset_names:
            train_loader, val_loader = self.data_manager.load_data(dataset)

            for filename in os.listdir(json_folder):
                if filename.endswith('.json'):
                    self._evaluate_single_loss(
                        os.path.join(json_folder, filename),
                        train_loader,
                        val_loader,
                        dataset
                    )

            self._evaluate_baseline_losses(train_loader, val_loader, dataset)
        
            # Update visualization call:
            
            self.visualizer.create_plots(
                [{"filename": result.name, "validation_accuracy": result.accuracy_progression} 
                for result in self.results_handler.results]
            )
            
            # Add JSON output generation:
            output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.results_handler.generate_json_output(output_file)

    def _evaluate_single_loss(
        self,
        file_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset: str
    ) -> Dict[str, Any]:
        """Evaluate a single loss function from a JSON file."""
        try:
            torch.manual_seed(self.config.seed)
            individual, loss_function = load_individual_from_json(
                filename=file_path,
                pset=self.pset,
                toolbox=self.toolbox
            )
            
            model = get_model_for_dataset(dataset)
            model.to(self.config.device)
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader
            )
            
            # Add results processing:
            self.results_handler.process_evaluation_metrics(
                name=os.path.basename(file_path),
                metrics=metrics,
                function_str=str(individual),
                total_batches=len(train_loader),
                epochs=self.config.epochs
            )
        except:
            metrics = {
                'train_loss': [99],
                'val_accuracy': [0.0],
                'batch_numbers': [1]
            }
            self.results_handler.process_evaluation_metrics(
                name=os.path.basename(file_path),
                metrics=metrics,
                function_str=None,
                total_batches=len(train_loader),
                epochs=self.config.epochs
            )


    def _evaluate_baseline_losses(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset: str
    ) -> List[Dict[str, Any]]:
        """Evaluate baseline loss functions (MSE and Cross-Entropy)."""
        baseline_losses = {
            #'MSE': torch.nn.MSELoss(),
            'CrossEntropy': torch.nn.CrossEntropyLoss()
        }

        for loss_name, loss_fn in baseline_losses.items():
            torch.manual_seed(self.config.seed)
            model = get_model_for_dataset(dataset)
            model.to(self.config.device)
            
            try:
                if loss_name == 'MSE':
                    loss_function = lambda outputs, targets: loss_fn(outputs, targets)
                else:  # CrossEntropy
                    loss_function = lambda outputs, targets: loss_fn(outputs, torch.argmax(targets, dim=1))
                
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader
                )
                
                # Add results processing:
                self.results_handler.process_evaluation_metrics(
                    name=f"{loss_name} (Baseline)",
                    metrics=metrics,
                    total_batches=len(train_loader),
                    epochs=self.config.epochs
                )
            except:
                continue


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = TrainingConfig()
    evaluator = LossFunctionEvaluator(config)
    evaluator.evaluate_loss_functions('subnet_1')

if __name__ == "__main__":
    main()
