from abc import ABC, abstractmethod
from collections import deque
import random

import numpy as np

Experience = tuple[np.ndarray, int, float, np.ndarray, bool]


class BaseReplayBuffer(ABC):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: deque[Experience] = deque([])

    def remember(self, experience: Experience) -> None:
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()  # Remove the oldest experience
        self.buffer.append(experience)

    @abstractmethod
    def sample_batch(self, batch_size: int) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray], list[bool]]:
        raise NotImplementedError


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self, max_size: int):
        self.max_size = max_size

    def sample_batch(self, batch_size: int) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray], list[bool]]:
        """
        Sample a batch of experiences from the replay memory.
        """
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        )


class PrioritizedExperienceBuffer(BaseReplayBuffer):
    def __init__(self, max_size: int, alpha: float = 0.6):
        """
        alpha for prioritization
        """
        super().__init__(max_size)
        self.alpha = alpha
        self.priorities = np.ones(max_size, dtype=np.float32)
        
        self.sampled_indices: np.ndarray[int] | None = None

    def remember(self, experience: Experience) -> None:
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()
            self.priorities[:-1] = self.priorities[1:]
            
        self.buffer.append(experience)
        max_priority = self.priorities.max() if self.buffer else 1.0  # max to ensure newly added experience has the highest priority
        self.priorities[len(self.buffer) - 1] = max_priority

    def sample_batch(self, batch_size: int) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray], list[bool]]:
        """
        Sample a batch of experiences from the replay memory.
        """
        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        self.sampled_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in self.sampled_indices]
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*samples)
        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        )

    def update_priorities(self, priorities: np.ndarray) -> None:
        if self.sampled_indices is None:
            raise ValueError("You need to sample a batch before updating priorities")
        for idx, priority in zip(self.sampled_indices, priorities):
            self.priorities[idx] = priority
        self.sampled_indices = None