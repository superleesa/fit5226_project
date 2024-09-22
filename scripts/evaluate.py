from fire import Fire

from fit5226_project.evaluation import Evaluation


def evaluate(model_path: str):
    evaluation = Evaluation(n=4)
    evaluation.run_dqn_train()
    evaluation.load_trained_dqn(model_path)
    
    average_score = evaluation.dqn_performance_test()
    print(f"Average performance score (1 is the best): {average_score:.4f}")
    
    evaluation.visualize_dqn()


if __name__ == "__main__":
    Fire(evaluate)
