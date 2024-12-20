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
from deap import base, creator, gp, tools
from dml.gene_io import load_individual_from_json
from dml.ops import create_pset, create_pset_validator
from evaluator.models.model_factory import get_model_for_dataset
from evaluator.models.losses import batch_loss

from tqdm import tqdm 

class LossFunctionEvaluator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.evaluator = LossEvaluator(config)
        self.visualizer = Visualizer()
        self.results_handler = ResultsHandler()  # Add this line
        self.initialize_deap()

    def initialize_deap(self):
        self.toolbox = base.Toolbox()
        self.pset = create_pset_validator()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        #self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )

    def evaluate_loss_functions(self, json_folder: str) -> None:
        """Evaluate multiple loss functions from JSON files."""
        for dataset in self.config.architectures.keys():
            train_loader, val_loader = self.data_manager.load_data(dataset)
            for architecture in self.config.architectures[dataset]:

                self.results_handler = ResultsHandler()  

                for filename in tqdm(os.listdir(json_folder)):
                    if filename.endswith('.json'):
                        self._evaluate_single_loss(
                            os.path.join(json_folder, filename),
                            train_loader,
                            val_loader,
                            dataset, 
                            architecture
                        )

                self._evaluate_baseline_losses(train_loader, val_loader, dataset,architecture)
            
                # Update visualization call:
                
                self.visualizer.create_plots(
                    [{"filename": result.name, "validation_accuracy": result.accuracy_progression} 
                    for result in self.results_handler.results]
                )
                
                # Add JSON output generation:
                output_file = f"evaluation_results_{dataset}_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.results_handler.generate_json_output(output_file)

    def _evaluate_single_loss(
        self,
        file_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset: str,
        arch: str
    ) -> Dict[str, Any]:
        """Evaluate a single loss function from a JSON file."""
        #try
        torch.manual_seed(self.config.seed)
        individual, loss_function, loss_str, _ = load_individual_from_json(
            filename=file_path,
            pset=self.pset,
            toolbox=self.toolbox
        )
        
        model = get_model_for_dataset(dataset, arch)
        model.to(self.config.device)
        if dataset == "shakespeare":
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, num_classes=85, metric_type="loss"
            )
        elif dataset == "fineweb":
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, num_classes=50257, metric_type="loss"
            )
        elif dataset == "cifar100":
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, num_classes=100, metric_type="accuracy"
            )            
        else:
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, metric_type="accuracy"
            )
        
        # Add results processing:
        self.results_handler.process_evaluation_metrics(
            name=os.path.basename(file_path),
            metrics=metrics,
            function_str=str(individual),
            total_batches=len(train_loader),
            epochs=self.config.epochs
        )
        # except Exception as e:
        #     print(e)
        #     metrics = {
        #         'train_loss': [99],
        #         'val_accuracy': [0.0],
        #         'batch_numbers': [1]
        #     }
        #     self.results_handler.process_evaluation_metrics(
        #         name=os.path.basename(file_path),
        #         metrics=metrics,
        #         function_str=None,
        #         total_batches=len(train_loader),
        #         epochs=self.config.epochs
        #     )


    def _evaluate_baseline_losses(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset: str,
        arch: str
    ) -> List[Dict[str, Any]]:
        """Evaluate baseline loss functions (MSE and Cross-Entropy)."""
        baseline_losses = {
            'MSE': torch.nn.MSELoss(),
            'CrossEntropy': torch.nn.CrossEntropyLoss(),
            "evolved": batch_loss
        }

        for loss_name, loss_fn in baseline_losses.items():
            torch.manual_seed(self.config.seed)
            model = get_model_for_dataset(dataset, arch)
            model.to(self.config.device)
            
            #try:
            if loss_name == 'MSE':
                loss_function = lambda outputs, targets: loss_fn(outputs, targets)
            else:  # CrossEntropy
                loss_function = torch.nn.CrossEntropyLoss()

            if dataset == "shakespeare":
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, num_classes=85, metric_type="loss"
                )
            elif dataset == "fineweb":
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, num_classes=50257, metric_type="loss"
                )
            elif dataset == "cifar100":
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, num_classes=100, metric_type="accuracy"
                )
            else:
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, metric_type="accuracy"
                )

            
            # Add results processing:
            self.results_handler.process_evaluation_metrics(
                name=f"{loss_name} (Baseline)",
                metrics=metrics,
                total_batches=len(train_loader),
                epochs=self.config.epochs
            )
            # except:
            #     continue



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
