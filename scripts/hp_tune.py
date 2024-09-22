from fit5226_project.tuning import Tuning


if __name__ == "__main__":
    tuning = Tuning()
    tuning.run_hyperparameter_tuning()
    tuning.hyperparameter_tuning_visualization()