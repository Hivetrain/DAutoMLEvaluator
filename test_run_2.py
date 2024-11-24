from evaluator.eval_config import TrainingConfig
from evaluator.main import LossFunctionEvaluator

if __name__ == "__main__":
    config = TrainingConfig(max_batches=3)
    evaluator = LossFunctionEvaluator(config)
    evaluator.evaluate_loss_functions('./my_checkpoints')
