import ast
import itertools
import pickle

import optuna

from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.train import Trainer


# FIXME: calculaing this in every trial is not efficient
def generate_reward_combinations(
    goal_state_rewards: list[int | float],
    item_state_rewards: list[int | float],
    goal_no_item_penalty: list[int | float],
) -> list[tuple[int | float, int | float, int | float]]:
    options = list(itertools.product(goal_state_rewards, item_state_rewards, goal_no_item_penalty))
    meaningful_options = []
    for option in options:
        if option[0] > option[1] > option[2]:
            meaningful_options.append(option)
    return meaningful_options


def objective(trial: optuna.Trial) -> tuple[float, float, float]:
    alpha = trial.suggest_categorical("alpha", [1e-4, 5e-4, 1e-3, 1e-2])
    tau = trial.suggest_float("tau", 0.03, 0.8)
    discount_rate = trial.suggest_float("discount_rate", 0.7, 0.975)
    epsilon = trial.suggest_float("epsilon", 0.2, 0.95)
    replay_memory_size = trial.suggest_int("replay_memory_size", 1000, 5000)
    min_replay_memory_size = trial.suggest_int("min_replay_memory_size", 100, 3000)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.8, 0.99999)
    epsilon_min = trial.suggest_float("epsilon_min", 0.001, 0.1)
    # batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    num_episodes = trial.suggest_int("num_episodes", 500, 3000)
    update_target_interval = trial.suggest_int("update_target_interval", 1, 50)

    reward_combinations = generate_reward_combinations(
        goal_state_rewards=[5, 10, 20], item_state_rewards=[2.5, 5, 10], goal_no_item_penalty=[-10, -5, -1]
    )
    # we stringify this suggestion becuase optuna-dashboard doesn't support tuples as suggestion
    reward_combination_tags = [str(option) for option in reward_combinations]
    reward_combination_tag = trial.suggest_categorical("reward_combination", reward_combination_tags)
    goal_state_reward, item_state_reward, goal_no_item_penalty = ast.literal_eval(reward_combination_tag)

    tune_env = Assignment2Environment(
        n=4,  # Grid size
        time_penalty=-1,
        goal_no_item_penalty=goal_no_item_penalty,
        item_state_reward=item_state_reward,
        goal_state_reward=goal_state_reward,
        with_animation=False,
    )

    tune_agent = DQNAgent(
        alpha=alpha,
        tau=tau,
        discount_rate=discount_rate,
        epsilon=epsilon,
        replay_memory_size=replay_memory_size,
        min_replay_memory_size=min_replay_memory_size,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        batch_size=128,
        with_log=False,
    )

    tune_trainer = Trainer(
        tune_agent,
        tune_env,
        update_target_interval=update_target_interval,
        num_episodes=num_episodes,
        with_log=False,
        with_validation=False,
        with_visualization=False,
        save_checkpoints=False,
    )
    tune_trainer.train()

    tune_trainer.save_agent()
    trial.set_user_attr("checkpoint_id", tune_trainer.training_unique_id)
    
    # # Prune the trial early if it's performing poorly
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()

    # considers all possible env states
    average_path_length_score, goal_reached_percentage, average_reward = tune_trainer.validate(is_eval=True)
    return average_path_length_score, goal_reached_percentage, average_reward


def tune(study_name: str) -> None:
    NUM_TRIALS = 500

    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],
        storage="sqlite:///tuning_result.db",
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)

    with open("best_trials.pickle", "wb") as file:
        pickle.dump(study.best_trials, file)
