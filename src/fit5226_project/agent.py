import random
from typing import List
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from fit5226_project.actions import Action


# import mlflow
from fit5226_project.tracker import mlflow_manager
from fit5226_project.replay_buffers import PrioritizedExperienceBuffer

class DQNAgent:
    def __init__(
        self,
        statespace_size: int = 11,
        action_space_size: int = len(Action),
        alpha: float = 0.0005,
        discount_rate: float = 0.99,
        epsilon: float = 1,
        epsilon_decay: float = 0.99997,
        epsilon_min: float = 0.007,
        replay_memory_size: int = 10000,
        batch_size: int = 128,
        min_replay_memory_size: int = 1000,
        tau: float = 0.05,
        with_log: bool = False,
        loss_log_interval: int = 100,
    ) -> None:
        """
        Initialize the DQN Agent
        """
        self.alpha = alpha  # learning rate for optimizer
        self.discount_rate = discount_rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # rate at which exploration rate decays
        self.epsilon_min = epsilon_min
        
        self.batch_size = batch_size
        self.action_space_size = action_space_size
        self.min_replay_memory_size = min_replay_memory_size
        
        self.replay_buffer = PrioritizedExperienceBuffer(max_size=replay_memory_size)
        
        # Initialize DQN models
        self.model = self.prepare_torch(statespace_size)  # prediction model
        self.target_model = deepcopy(self.model)  # target model

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.alpha, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.steps = 0
        
        self.tau = tau  # for soft update of target parameters
        
        self.with_log = with_log
        self.loss_log_interval = loss_log_interval

    def prepare_torch(self, statespace_size: int):
        """
        Prepare the PyTorch model for DQL.
        """
        l1 = statespace_size
        l2 = 150
        l3 = 100
        l4 = self.action_space_size
        model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4)
        )
        return model

    def update_target_network(self) -> None:
        """
        Copy weights from the prediction network to the target network.
        """
        # self.target_model = deepcopy(self.model)
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_net_state_dict)

    def select_action(self, state: np.ndarray, available_actions: List[Action], is_test: bool = False) -> tuple[Action, bool, np.ndarray]:
        """
        Select an action using an ε-greedy policy.
        
        second return val for is_greedy
        chosen_action, is_greedy, q values for all actions
        """
        qvals = self.get_qvals(state)
        if not is_test and random.random() < self.epsilon:
            return random.choice(available_actions), False, qvals
        else:
            # Filter Q-values to only consider available actions
            valid_qvals = [qvals[action.value] for action in available_actions]
            return available_actions[np.argmax(valid_qvals)], True, qvals
        
    def get_qvals(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a given state from the prediction network.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Convert to tensor
        with torch.no_grad():
            qvals_tensor = self.model(state_tensor)
        return qvals_tensor.detach().numpy()[0]

    def get_maxQ(self, state: np.ndarray) -> float:
        """
        Get the maximum Q-value for a given state from the target network.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Convert to tensor
        with torch.no_grad():
            max_qval_tensor = torch.max(self.target_model(state_tensor))
        return max_qval_tensor.item()
    
    def get_double_q(self, state: np.ndarray) -> float:
        """
        Calculate the Double DQN target value for a given state.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            best_action_index = torch.argmax(self.model(state_tensor)[0]).item()
            max_qval = self.target_model(state_tensor)[0][best_action_index].item()
            
        return max_qval

    def train_one_step(self, states: List[np.ndarray], actions: List[int], targets: List[float]) -> float:
        """
        Perform a single training step on the prediction network.
        """
        # Convert states, actions, and targets to tensors
        state_tensors = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in states])
        action_tensors = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        target_tensors = torch.tensor(targets, dtype=torch.float)
        
        self.optimizer.zero_grad()
        qvals = self.model(state_tensors).gather(1, action_tensors).squeeze()
        losses = self.loss_fn(qvals, target_tensors)
        loss = losses.mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        if self.with_log and self.steps % self.loss_log_interval == 0:
            with torch.no_grad():
                mlflow_manager.log_avg_predicted_qval(qvals.mean().item(), step=self.steps)
                mlflow_manager.log_avg_target_qval(target_tensors.mean().item(), step=self.steps)
                mlflow_manager.log_max_predicted_qval(qvals.max().item(), step=self.steps)
                mlflow_manager.log_max_target_qval(target_tensors.max().item(), step=self.steps)
        
        with torch.no_grad():
            self.replay_buffer.update_priorities(losses.detach().numpy())  # TODO: maybe using l1 better (at least original paper uses l1)
            
        return loss.item()

    def replay(self) -> None:
        """
        Train the model using experience replay.
        """
        if len(self.replay_buffer.buffer) < self.min_replay_memory_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

        # Compute targets
        targets = []
        for i in range(self.batch_size):
            if dones[i]:
                targets.append(rewards[i])
            else:
                # max_future_q = self.get_maxQ(next_states[i])
                max_future_q = self.get_double_q(next_states[i])
                targets.append(rewards[i] + self.discount_rate * max_future_q)

        self.steps += 1
        # Train the model
        loss = self.train_one_step(states, actions, targets)
        
        if self.with_log and self.steps % self.loss_log_interval == 0:
            mlflow_manager.log_loss(loss, step=self.steps)

        # TODO: plot loss

    def save_state(self, filepath: Path | str):
        """Save the entire agent state, including model weights and hyperparameters."""
        torch.save({
            'model_state_dict': self.model.state_dict(),  # Model weights
            'target_model_state_dict': self.target_model.state_dict(),  # Target model weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'epsilon': self.epsilon,  # Epsilon value
            'epsilon_decay': self.epsilon_decay,  # Epsilon decay rate
            'epsilon_min': self.epsilon_min,  # Minimum epsilon
            'discount_rate': self.discount_rate,  # Discount factor
            'replay_buffer': self.replay_buffer,  # replay_buffer
            'steps': self.steps,  # Steps to update target network
            'random_state': random.getstate(),  # Python random state
            'numpy_random_state': np.random.get_state(),  # Numpy random state
        }, filepath)

    def load_state(self, filepath, load_seeds=False):
        """Load the entire agent state, including model weights and hyperparameters."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])  # Load target model weights
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Restore optimizer state
        self.epsilon = checkpoint['epsilon']  # Restore epsilon value
        self.epsilon_decay = checkpoint['epsilon_decay']  # Restore epsilon decay rate
        self.epsilon_min = checkpoint['epsilon_min']  # Restore minimum epsilon
        self.discount_rate = checkpoint['discount_rate']  # Restore discount factor
        self.replay_buffer = checkpoint['replay_buffer']  # Restore replay_buffer
        self.steps = checkpoint['steps']  # Restore steps
        if load_seeds:
            random.setstate(checkpoint['random_state'])  # Restore Python random state
            np.random.set_state(checkpoint['numpy_random_state'])  # Restore Numpy random state
        # If using a learning rate scheduler:
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Restore scheduler state
