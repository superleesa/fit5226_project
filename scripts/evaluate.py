import yaml
from fire import Fire

from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.train import Trainer
from fit5226_project.utils import ensure_key_exists


def evaluate(model_path: str, config_path: str, num_visualizations: int = 3):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    ensure_key_exists(config)
    
    agent = DQNAgent(**config["agent"])
    agent.load_state(model_path)
    env = Assignment2Environment(n=4, **config["env"])
    trainer = Trainer(agent, env, **config["trainer"])

    score, goal_reached_percentage, average_reward = trainer.validate(is_eval=True)
    print("score: ", score)
    print("goal_reached_percentage: ", goal_reached_percentage)
    print("average_reward: ", average_reward)
    trainer.visualize_sample_episode(num_visualizations)


if __name__ == "__main__":
    Fire(evaluate)
