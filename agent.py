import numpy as np

from env import Environment, Item
from state import State


class Agent:
    def __init__(
        self,
        alpha: float = 0.1,
        discount_rate: float = 0.9,
        epsilon: float = 0.1,
        num_episode_per_intermediate_item: int = 100,
    ) -> None:
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.discount_rate = discount_rate
        self.num_episode_per_intermediate_item = num_episode_per_intermediate_item

    def train_one_intermediate_item(self, item: Item | None = None):
        env = Environment(n=5, item=item)
        env.initialize_state()

        qval_matrix = np.zeros((env.n, env.n, 4))  # 4 for 4 actions

        for episode in range(self.num_episode_per_intermediate_item):
            while True:
                current_state = env.get_state()
                
                action = self.choose_action(qval_matrix, current_state)
                next_state = env.get_next_state_from_action(
                    current_state, action
                )  # TODO: implement this method
                reward = env.get_reward(next_state)
                self.update(current_state, next_state, reward, action, qval_matrix)
                
                if env.is_goal_state(next_state):
                    break
                current_state = next_state
    
    def update(self, current_state: State, next_state: State, reward: float, action: int, qval_matrix: np.ndarray) -> None:
        qval_difference = self.alpha * (
            reward
            + self.discount_rate * np.max(qval_matrix[*next_state.agent_location])
            - qval_matrix[*current_state.agent_location, action]
        )
        qval_matrix[
            *current_state.agent_location, action
        ] += qval_difference

    def choose_action(self, qval_matrix, state: State, is_training: bool = True) -> int:
        """
        Epislon greedy method to choose action
        """
        agent_location_x, agent_location_y = state.agent_location
        if is_training and np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(qval_matrix[agent_location_x, agent_location_y])
