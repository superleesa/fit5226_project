import ast
import itertools

import optuna
import yaml

from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.train import Trainer


# FIXME: calculaing this in every trail is not efficient
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


def objective(trial: optuna.Trial) -> float:
    alpha = trial.suggest_categorical("alpha", [1e-4, 5e-4, 1e-3, 1e-2])
    tau = trial.suggest_float("tau", 0.03, 0.8)
    discount_rate = trial.suggest_float("discount_rate", 0.7, 0.975)
    epsilon = trial.suggest_categorical("epsilon", [0.2, 0.35, 0.45, 0.6])
    replay_memory_size = trial.suggest_int("replay_memory_size", 1000, 5000)
    # batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    time_penalty = trial.suggest_int("time_penalty", -10, -1)
    num_episodes = trial.suggest_int("num_episodes", 100, 1000)

    reward_combinations = generate_reward_combinations(
        goal_state_rewards=[5, 10, 20], item_state_rewards=[2.5, 5, 10], goal_no_item_penalty=[-10, -5, -1]
    )
    # we stringify this suggestion becuase optuna-dashboard doesn't support tuples as suggestion
    reward_combination_tags = [str(option) for option in reward_combinations]
    reward_combination_tag = trial.suggest_categorical(
            "reward_combination", reward_combination_tags
        )
    goal_state_reward, item_state_reward, goal_no_item_penalty = ast.literal_eval(reward_combination_tag)

    tune_env = Assignment2Environment(
        n=4,  # Grid size
        time_penalty=time_penalty,
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
        batch_size=128,
        with_log=False,
    )

    tune_trainer = Trainer(
        tune_agent,
        tune_env,
        num_validation_episodes=20,
        with_log=False,
        with_validation=False,
        with_visualization=False,
    )
    tune_trainer.train(num_episodes=num_episodes)

    # # Prune the trial early if it's performing poorly
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()

    return tune_trainer.validate()[0]


def tune(study_name: str) -> None:
    study = optuna.create_study(direction="maximize", storage="sqlite:///tuning_result.db", study_name=study_name)
    num_trials = 200

    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
    
    print("Best Hyperparameters: ", study.best_params)
    print("Best Value: ", study.best_value)
    with open("config.yml", "w") as file:
        yaml.dump(study.best_params, file, default_flow_style=False)

    optuna.visualization.plot_optimization_history(study)
