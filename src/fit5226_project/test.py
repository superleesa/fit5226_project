# Import necessary libraries and classes
import numpy as np
import torch
import random
from fit5226_project.actions import Action
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import Assignment2State
from fit5226_project.agent import DQNAgent
from fit5226_project.train import Trainer

# Initialize the environment
env = Assignment2Environment(
    n=4,  # Grid size
    time_penalty=-1,
    item_state_reward=200,
    goal_state_reward=300,
    direction_reward_multiplier=1,
    with_animation=True  # Enable animation for visualization
)

# Initialize the agent
agent = DQNAgent(
    statespace_size=11,  # Size of the state space (based on Assignment2State attributes)
    action_space_size=len(Action),  # Number of possible actions
    alpha=0.997,
    discount_rate=0.95,
    epsilon=1.0,
    epsilon_decay=0.997,
    epsilon_min=0.1,
    replay_memory_size=1000,
    batch_size=200,
    update_target_steps=500
)

# Initialize the trainer with the agent and environment
trainer = Trainer(agent, env)

# Train for 5 episodes
trainer.train(num_episodes=5)
