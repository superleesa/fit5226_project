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

