import numpy as np
import pickle
from agent import Agent
from env import InferenceEnvironment, ItemObject


def load_qval_matrix(item_location: tuple[int, int]) -> np.ndarray:
    with open(f'qval_matrix{item_location[0]}_{item_location[1]}.pickle', "rb") as f:
        return pickle.load(f)

def inference():
    # Initialize the inference environment
    env = InferenceEnvironment(n=5)
    env.item.set_location_randomly(env.n, env.n)
    item_location = env.item.location
    env.agent.set_location_randomly(env.n, env.n, disallowed_locations=[item_location])
    
    # Load the pre-trained Q-value matrix for the random item location
    qval_matrix = load_qval_matrix(item_location)
    
    # Start from the initial state with random agent and item locations
    current_state = env.get_state()
    
    # Move towards the item
    while current_state.agent_location != current_state.item_location:
        possible_actions = env.get_available_actions()
        action = env.agent.choose_action(possible_actions, current_state, qval_matrix, is_training=False)
        _, next_state = env.step(action)
        current_state = next_state
    env.item.location = None  # Removing the item to simulate it being picked up
    env.goal_location = (4, 4)  # Set the actual goal location (n-1, n-1)
    
    while current_state.agent_location != env.goal_location:
        possible_actions = env.get_available_actions()
        action = env.agent.choose_action(possible_actions, current_state, qval_matrix, is_training=False)
        _, next_state = env.step(action)
        current_state = next_state
    
    print("Goal Reached!")

if __name__ == "__main__":
    inference()
