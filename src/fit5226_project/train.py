from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import Assignment2State,State
from fit5226_project.actions import Action
import numpy as np

class Trainer:
    def __init__(self, agent: DQNAgent, environment: Assignment2Environment) -> None:
        """
        Initialize the Trainer with the DQN agent and environment.
        """
        self.agent = agent
        self.environment = environment
        self.current_sub_environment = None  # Track the current environment

    def train_one_episode(self) -> None:
        """
        Conducts training for a single episode.
        """
        # Initialize the environment for a new episode
        self.environment.initialize_for_new_episode()
        # Store the current environment reference to use throughout the episode
        self.current_sub_environment = self.environment.current_sub_environment

        current_state = self.current_sub_environment.get_state()  # Get state from current sub-environment
        done = False

        while not done:
            # Convert the current state to a numpy array for input to the neural network
            state_array = self.state_to_array(current_state)

            # Retrieve available actions from the current sub-environment
            available_actions = self.current_sub_environment.get_available_actions(current_state)

            # Print debug information: agent location, item location, available actions, and has item
            print(f"Agent Location: {current_state.agent_location}")
            print(f"Item Location: {current_state.item_location}")
            print(f"Available Actions: {available_actions}")
            print(f"Has Item: {current_state.has_item}")

            # Select an action using the agent's ε-greedy policy
            action = self.agent.select_action(state_array, available_actions)

            # Print the selected action
            print(f"Selected Action: {action}")

            # Execute the action in the current sub-environment, receive reward and next state
            reward, next_state = self.current_sub_environment.step(action)

            # Print the reward received after taking the action
            print(f"Reward: {reward}")

            # Convert the next state to a numpy array
            next_state_array = self.state_to_array(next_state)

            # Check if the next state is a goal state
            done = self.current_sub_environment.is_goal_state(next_state)

            # Store experience in the agent's replay memory
            self.agent.remember((state_array, action.value, reward, next_state_array, done))

            # Learn from experiences using experience replay
            self.agent.replay()

            # Move to the next state
            current_state = next_state

    def state_to_array(self, state: State) -> np.ndarray:
        """
        Converts a State object into a numpy array suitable for input to the DQN.
        """
        # Check if the state is an instance of Assignment2State
        if isinstance(state, Assignment2State):
            # Convert Assignment2State to array
            state_array = np.array([
                *state.agent_location,  # Agent's (x, y) location
                *state.item_location,   # Item's (x, y) location
                float(state.has_item),  # 1 if agent has item, 0 otherwise
                *state.goal_location,   # Goal's (x, y) location
                *state.goal_direction,  # Direction to goal (dx, dy)
                *state.item_direction   # Direction to item (dx, dy)
            ])
        else:
            # Convert basic State to array (without goal-related information)
            state_array = np.array([
                *state.agent_location,  # Agent's (x, y) location
                *state.item_location,   # Item's (x, y) location
                float(state.has_item)   # 1 if agent has item, 0 otherwise
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
            # Ensure reference is updated to the new environment for each episode
            self.current_sub_environment = self.environment.current_sub_environment

            current_state = self.current_sub_environment.get_state()  # Use current sub-environment's state
            done = False

            while not done:
                # Convert the current state to a numpy array for input to the neural network
                state_array = self.state_to_array(current_state)

                # Select the best action (exploitation only, no exploration)
                qvals = self.agent.get_qvals(state_array)
                action = Action(np.argmax(qvals))

                # Execute the action in the environment
                reward, next_state = self.current_sub_environment.step(action)

                # Check if the next state is a goal state
                done = self.current_sub_environment.is_goal_state(next_state)
                current_state = next_state

            # Check if the episode was successful (reached the goal)
            if done:
                success_count += 1
            print(f"Evaluation Episode {episode + 1} completed.")

        success_rate = (success_count / num_episodes) * 100
        print(f"Success Rate: {success_rate:.2f}% over {num_episodes} episodes.")
