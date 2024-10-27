import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any

class Visualizer:
    @staticmethod
    def create_plots(metrics_list: List[Dict[str, Any]], output_dir: str = '.'):
        """Create and save performance plots."""
        # # Training Loss Plot
        # plt.figure(figsize=(12, 8))
        # for metric in metrics_list:
        #     plt.plot(metric['training_loss'], label=metric["filename"])
        # plt.title('Training Loss')
        # plt.xlabel('Batch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig(os.path.join(output_dir, 'train_loss_function_performance.png'))
        # plt.close()

        # Validation Accuracy Plot
        plt.figure(figsize=(12, 8))
        for metric in metrics_list:
            plt.plot(metric['validation_accuracy'], label=metric["filename"])
        plt.title('Validation Accuracy')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_loss_function_performance.png'))
        plt.close()