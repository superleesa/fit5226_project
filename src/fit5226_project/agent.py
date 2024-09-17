import random
import numpy as np
from fit5226_project.actions import Action
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import State, Assignment2State
import torch
import copy
from typing import List, Tuple

class DQNAgent:
    def __init__(
        self,
        statespace_size: int = 11,
        action_space_size: int = len(Action),
        alpha: float = 0.0005,
        discount_rate: float = 0.95,
        epsilon: float = 0.4,
        epsilon_decay: float = 0.97,
        epsilon_min: float = 0.1,
        replay_memory_size: int = 10000,
        batch_size: int = 30,
        update_target_steps: int = 25,
    ) -> None:
        """
        Initialize the DQN Agent
        """
        self.alpha = alpha  # learning rate for optimizer
        self.discount_rate = discount_rate  # discount factor for future rewards
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # rate at which exploration rate decays
        self.epsilon_min = epsilon_min  # minimum exploration rate
        self.replay_memory = []  # experience replay memory
        self.replay_memory_size = replay_memory_size  # maximum size of replay memory
        self.batch_size = batch_size  # batch size for experience replay
        self.update_target_steps = update_target_steps  # steps after which to update target network
        self.action_space_size = action_space_size  # number of possible actions
        
        # Initialize DQN models
        self.model = self.prepare_torch(statespace_size)  # prediction model
        self.target_model = copy.deepcopy(self.model)  # target model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss()
        self.steps = 0  # to track when to update target network

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
        self.target_model.load_state_dict(self.model.state_dict())

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

    def train_one_step(self, states: List[np.ndarray], actions: List[int], targets: List[float]) -> float:
        """
        Perform a single training step on the prediction network.
        """
        # Convert states, actions, and targets to tensors
        state_tensors = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in states])
        action_tensors = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        target_tensors = torch.tensor(targets, dtype=torch.float)

        # Calculate current Q values
        qvals = self.model(state_tensors).gather(1, action_tensors).squeeze()

        # Compute loss
        loss = self.loss_fn(qvals, target_tensors)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def remember(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        """
        Store an experience in the replay memory.
        """
        if len(self.replay_memory) >= self.replay_memory_size:
            self.replay_memory.pop(0)  # Remove the oldest experience
        self.replay_memory.append(experience)

    def replay(self) -> None:
        """
        Train the model using experience replay.
        """
        if len(self.replay_memory) < self.batch_size:
            return  # Not enough experiences to sample from

        # Sample a minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Compute targets
        targets = []
        for i in range(self.batch_size):
            if dones[i]:
                targets.append(rewards[i])
            else:
                max_future_q = self.get_maxQ(next_states[i])
                targets.append(rewards[i] + self.discount_rate * max_future_q)

        # Train the model
        loss = self.train_one_step(states, actions, targets)

        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_steps == 0:
            self.update_target_network()

    def save_state(self, filepath):
        """Save the entire agent state, including model weights and hyperparameters."""
        torch.save({
            'model_state_dict': self.model.state_dict(),  # Model weights
            'target_model_state_dict': self.target_model.state_dict(),  # Target model weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'epsilon': self.epsilon,  # Epsilon value
            'epsilon_decay': self.epsilon_decay,  # Epsilon decay rate
            'epsilon_min': self.epsilon_min,  # Minimum epsilon
            'discount_rate': self.discount_rate,  # Discount factor
            'replay_memory': self.replay_memory,  # Replay memory
            'steps': self.steps,  # Steps to update target network
            'random_state': random.getstate(),  # Python random state
            'numpy_random_state': np.random.get_state(),  # Numpy random state
        }, filepath)

    def load_state(self, filepath):
        """Load the entire agent state, including model weights and hyperparameters."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])  # Load target model weights
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Restore optimizer state
        self.epsilon = checkpoint['epsilon']  # Restore epsilon value
        self.epsilon_decay = checkpoint['epsilon_decay']  # Restore epsilon decay rate
        self.epsilon_min = checkpoint['epsilon_min']  # Restore minimum epsilon
        self.discount_rate = checkpoint['discount_rate']  # Restore discount factor
        self.replay_memory = checkpoint['replay_memory']  # Restore replay memory
        self.steps = checkpoint['steps']  # Restore steps
        random.setstate(checkpoint['random_state'])  # Restore Python random state
        np.random.set_state(checkpoint['numpy_random_state'])  # Restore Numpy random state
        # If using a learning rate scheduler:
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Restore scheduler state
