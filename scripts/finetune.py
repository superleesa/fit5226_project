import yaml
from fire import Fire

from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.train import Trainer
from fit5226_project.utils import ensure_key_exists


def train(config_path: str, base_model_path: str | None = None):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    ensure_key_exists(config)

    dqn_env = Assignment2Environment(n=4, **config["env"])
    dqn_agent = DQNAgent(**config["agent"])
    if base_model_path:
        dqn_agent.load_state(base_model_path)

    trainer = Trainer(agent=dqn_agent, environment=dqn_env, **config["trainer"])
    trainer.train()


if __name__ == "__main__":
    Fire(train)
