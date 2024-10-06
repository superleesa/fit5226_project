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
    time_penalty=-2,
    item_state_reward=200,
    goal_state_reward=300,
    direction_reward_multiplier=1,
    with_animation=False,  # Enable animation for visualization
)

# Initialize the agent with updated parameters
agent = DQNAgent(
    statespace_size=11,  # Size of the state space (based on Assignment2State attributes)
    action_space_size=len(Action),  # Number of possible actions
    alpha=0.997,
    discount_rate=0.95,
    epsilon=1.0,
    epsilon_decay=0.997,
    epsilon_min=0.1,
    replay_memory_size=1000,  # Updated memory size
    batch_size=200,  # Updated batch size
    update_target_steps=500,
)

# Load the pre-trained state (weights and hyperparameters)
agent.load_state("dqn_agent_state_truncated.pth")  # Replace with the path to your saved state file

# Initialize the trainer with the agent and environment
trainer = Trainer(agent, env)

# Save the trained model weights before training (optional)
# agent.save_model("dqn_agent_weights_before_training.pth")

# Train for 1000 episodes
trainer.train(num_episodes=1000)  # Updated number of episodes

# Save the new state (weights and hyperparameters) after further training
# agent.save_state("dqn_agent_full_state.pth")
# agent.save_state("dqn_agent_state_final.pth")
agent.save_state("dqn_agent_state_truncated.pth")

# Evaluate the trained model
# trainer.evaluate(num_episodes=5)
