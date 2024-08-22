import numpy as np
import pickle
import random

from env import Environment, ItemObject
from state import State


def generate_grid_location_list(max_x: int, max_y) -> list[tuple[int, int]]:
    return [(i, j) for i in range(max_x) for j in range(max_y)]


def save_trained_qval_matrix(trained_qval_matrix: np.ndarray, item: ItemObject) -> None:
    if item.location is None:
        raise ValueError("Item location is None")
    with open(f'qval_matrix{item.location[0]}_{item.location[1]}.pickle', "wb") as f:
        pickle.dump(trained_qval_matrix, f)


class Agent:
    def __init__(
        self,
        alpha: float = 0.3,
        discount_rate: float = 0.9,
        epsilon: float = 0.1,
        num_episode_per_intermediate_item: int = 1000,
        grid_size: tuple[int, int] = (5, 5),
        save_weights: bool = False,
    ) -> None:
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.discount_rate = discount_rate
        self.num_episode_per_intermediate_item = num_episode_per_intermediate_item
        self.grid_size = grid_size
        self.save_weights = save_weights
        
        self.trained_qval_matrices: list[np.ndarray] = []
    
    def train(self) -> None:
        """
        We are training for all "goal location" in the grid; so indivisual state consists of x, y, goal_x, goal_y, technically speaking.
        However, to ensure that the agent samples from all possible goal locations fairly, we will separately train for all possible goal locations.
        """
        item_grid_locations = generate_grid_location_list(self.grid_size[0], self.grid_size[1])
        all_items = [ItemObject(grid_location) for grid_location in item_grid_locations]
        for item in all_items:
            qval_matrix = self.train_one_intermediate_item(item)
            self.trained_qval_matrices.append(qval_matrix)
            if self.save_weights:
                save_trained_qval_matrix(qval_matrix, item)

    def train_one_intermediate_item(self, item: ItemObject | None = None) -> np.ndarray:
        env = Environment(n=5, item=item)

        qval_matrix = np.zeros((env.n, env.n, 4))  # 4 for 4 actions

        for episode in range(self.num_episode_per_intermediate_item):
            env.initialize_for_new_episode()
            current_state = env.get_state()
            while True:
                if env.is_goal_state(current_state):
                    break
                
                # TODO: i think we don't need to keep control of the current state (env already has within it)
                possible_actions = env.get_available_actions()
                action = self.choose_action(possible_actions, current_state, qval_matrix)
                next_state = env.get_next_state(action)
                reward = env.get_reward(next_state)
                # TODO: call env.step(action) instead of calling get_next_state and get_reward separately
                self.update(current_state, next_state, reward, action, qval_matrix)
                
                if env.is_goal_state(next_state):
                    break
                current_state = next_state
        
        return qval_matrix
    
    def update(self, current_state: State, next_state: State, reward: float, action: int, qval_matrix: np.ndarray) -> None:
        qval_difference = self.alpha * (
            reward
            + self.discount_rate * np.max(qval_matrix[*next_state.agent_location])
            - qval_matrix[*current_state.agent_location, action]
        )
        qval_matrix[
            *current_state.agent_location, action
        ] += qval_difference

    def choose_action(self, possible_actions: list[int], state: State, qval_matrix: np.ndarray, is_training: bool = True) -> int:
        """
        Epislon greedy method to choose action
        """
        agent_location_x, agent_location_y = state.agent_location
        if not is_training and random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            action_to_qval = list(zip(possible_actions, qval_matrix[agent_location_x, agent_location_y, possible_actions]))
            return max(action_to_qval, key=lambda x: x[1])[0]
