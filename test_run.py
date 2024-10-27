from evaluator.eval_config import TrainingConfig
from evaluator.main import LossFunctionEvaluator

def main():
    config = TrainingConfig(
    )
    evaluator = LossFunctionEvaluator(config)
    evaluator.evaluate_loss_functions('/home/mekaneeky/repos/automl_evaluator/checkpoints')

if __name__ == "__main__":
    main()