from datetime import datetime
from typing import List, Dict, Any
import json
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ComplexityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

@dataclass
class LossFunctionResult:
    name: str
    mnist_accuracy: float
    training_time: float
    complexity: ComplexityLevel
    accuracy_progression: List[float]
    cifar10_accuracy: float = None  # Optional for now


class ResultsHandler:
    def __init__(self):
        self.results: List[LossFunctionResult] = []
    
    def add_result(self, result: LossFunctionResult):
        """Add a new result to the collection."""
        self.results.append(result)

    def _determine_complexity(self, function_str: str) -> ComplexityLevel:
        """Determine complexity based on function structure."""
        # Simple heuristic based on function length and number of operations
        if len(function_str) < 50:
            return ComplexityLevel.LOW
        elif len(function_str) < 150:
            return ComplexityLevel.MEDIUM
        return ComplexityLevel.HIGH

    def process_evaluation_metrics(self, 
                                 name: str,
                                 metrics: Dict[str, List[float]],
                                 function_str: str = None,
                                 total_batches: int = None,
                                 epochs: int = None) -> LossFunctionResult:
        """Process raw evaluation metrics into a structured result."""
        # Get the final accuracy and create progression list
        accuracy_values = metrics['val_accuracy']
        final_accuracy = accuracy_values[-1] * 100  # Convert to percentage
        
        # Calculate training time as proportion of total possible iterations
        if total_batches and epochs:
            training_time = len(metrics['train_loss']) / (total_batches * epochs)
        else:
            training_time = 1.0  # Default to 1.0 if not specified
        
        # Determine complexity
        complexity = (self._determine_complexity(function_str) 
                     if function_str else ComplexityLevel.LOW)
        
        # Create result object
        result = LossFunctionResult(
            name=name,
            mnist_accuracy=final_accuracy,
            training_time=training_time,
            complexity=complexity,
            accuracy_progression=[acc * 100 for acc in accuracy_values]  # Convert to percentages
        )
        
        self.add_result(result)
        return result

    def generate_json_output(self, output_file: str = None) -> Dict[str, Any]:
        """Generate JSON output in the specified format."""
        output = {
            "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sota": [
                {
                    "name": result.name,
                    "mnist_accuracy": round(result.mnist_accuracy, 1),
                    "cifar10_accuracy": round(result.cifar10_accuracy, 1) if result.cifar10_accuracy else None,
                    "training_time": round(result.training_time, 2),
                    "complexity": result.complexity.value
                }
                for result in self.results
            ],
            "performance": [
                {
                    "name": result.name,
                    "values": [round(v, 2) for v in result.accuracy_progression]
                }
                for result in self.results
            ]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)
        
        return output