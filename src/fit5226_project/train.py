from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import State, Assignment2State
from fit5226_project.actions import Action
import numpy as np


class Trainer:
    def __init__(self, agent: DQNAgent, environment: Assignment2Environment) -> None:
        """
        Initialize the Trainer with the DQN agent and environment.
        """
        self.agent = agent
        self.environment = environment

    def train_one_episode(self) -> None:
        """
        Conducts training for a single episode.
        """
        # Initialize the environment for a new episode
        self.environment.initialize_for_new_episode()
        current_state = self.environment.get_state()
        done = False

        while not done:
            # Convert the current state to a numpy array for input to the neural network
            state_array = self.state_to_array(current_state)

            # Select an action using the agent's ε-greedy policy
            action = self.agent.select_action(state_array)

            # Execute the action in the environment, receive reward and next state
            reward, next_state = self.environment.step(action)

            # Convert the next state to a numpy array
            next_state_array = self.state_to_array(next_state)

            # Check if the next state is a goal state
            done = self.environment.is_goal_state(next_state)

            # Store experience in the agent's replay memory
            self.agent.remember((state_array, action.value, reward, next_state_array, done))

            # Learn from experiences using experience replay
            self.agent.replay()

            # Move to the next state
            current_state = next_state

    def state_to_array(self, state: Assignment2State) -> np.ndarray:
        """
        Converts a State object into a numpy array suitable for input to the DQN.
        """
        # Assuming state contains agent_location, item_location, has_item, goal_location, goal_direction, item_direction
        state_array = np.array([
            *state.agent_location,  # Agent's (x, y) location
            *state.item_location,   # Item's (x, y) location
            float(state.has_item),  # 1 if agent has item, 0 otherwise
            *state.goal_location,   # Goal's (x, y) location
            *state.goal_direction,  # Direction to goal (dx, dy)
            *state.item_direction   # Direction to item (dx, dy)
        ])
        return state_array

    def train(self, num_episodes: int) -> None:
        """
        Train the agent across multiple episodes.
        """
        for episode in range(num_episodes):
            print(f"Starting Episode {episode + 1}")
            self.train_one_episode()
            print(f"Episode {episode + 1} completed. Epsilon: {self.agent.epsilon:.4f}")

    def evaluate(self, num_episodes: int) -> None:
        """
        Evaluate the agent's performance over a specified number of episodes.
        """
        success_count = 0

        for episode in range(num_episodes):
            print(f"Starting Evaluation Episode {episode + 1}")
            self.environment.initialize_for_new_episode()
            current_state = self.environment.get_state()
            done = False

            while not done:
                # Convert the current state to a numpy array for input to the neural network
                state_array = self.state_to_array(current_state)

                # Select the best action (exploitation only, no exploration)
                qvals = self.agent.get_qvals(state_array)
                action = Action(np.argmax(qvals))

                # Execute the action in the environment
                reward, next_state = self.environment.step(action)

                # Check if the next state is a goal state
                done = self.environment.is_goal_state(next_state)
                current_state = next_state

            # Check if the episode was successful (reached the goal)
            if done:
                success_count += 1
            print(f"Evaluation Episode {episode + 1} completed.")

        success_rate = (success_count / num_episodes) * 100
        print(f"Success Rate: {success_rate:.2f}% over {num_episodes} episodes.")
