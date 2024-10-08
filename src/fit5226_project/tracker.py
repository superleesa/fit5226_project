import mlflow


# FIXME: this code too verbose...
class MLFlowManager:
    def __init__(self, experiment_name: str | None = None, run_name: str | None = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.experiment_has_start = False

    def start_experiment_if_not_already(self) -> None:
        if not self.experiment_has_start:
            if self.experiment_name is not None:
                mlflow.set_experiment(self.experiment_name)
            if self.run_name is not None:
                mlflow.start_run(run_name=self.run_name)
            else:
                mlflow.start_run()
            self.experiment_has_start = True

    def log_loss(self, loss: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("loss", loss, step=step)

    def log_reward(self, reward: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("reward", reward, step=step)

    def log_episode_wise_reward(self, reward: float, episode_idx: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("episode_wise_reward", reward, step=episode_idx)

    def log_avg_predicted_qval(self, avg_qval: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("avg_predicted_qval", avg_qval, step=step)

    def log_avg_target_qval(self, target_qval: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("avg_target_qval", target_qval, step=step)

    def log_max_target_qval(self, max_qval: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("max_target_qval", max_qval, step=step)

    def log_max_predicted_qval(self, max_qval: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("max_predicted_qval", max_qval, step=step)

    def log_validation_score(self, validation_score: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("validation_score", validation_score, step=step)

    def log_num_failed_validation_episodes(self, num_failed_episodes: float, step: int):
        self.start_experiment_if_not_already()
        mlflow.log_metric("failed_validation_episodes", num_failed_episodes, step=step)

    def end_run(self):
        mlflow.end_run()


mlflow_manager = MLFlowManager()
