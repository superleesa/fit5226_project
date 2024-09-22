from typing import Any

from fire import Fire

from fit5226_project.train import Trainer
from fit5226_project.env import Assignment2Environment
from fit5226_project.agent import DQNAgent


def train(base_model_path: str | None = None, **agent_params: dict[str, Any]):
    dqn_envs = Assignment2Environment(n=4, with_animation=False)
    dqn_agent = DQNAgent(with_log=True, **agent_params)
    if base_model_path:
        dqn_agent.load_state(base_model_path)
    
    trainer = Trainer(
        agent=dqn_agent,
        environment=dqn_envs,
        with_log=True,
    )
    trainer.train(num_episodes=250)


if __name__ == "__main__":
    Fire(train)
