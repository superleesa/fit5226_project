from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import Assignment2State,State
from fit5226_project.actions import Action
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, agent: DQNAgent, environment: Assignment2Environment) -> None:
        """
        Initialize the Trainer with the DQN agent and environment.
        """
        self.agent = agent
        self.environment = environment
        self.current_sub_environment = None  # Track the current environment
        self.episode_rewards = []  # List to store total rewards for each episode
        self.episode_steps = []  # List to store the number of steps per episode


    def train_one_episode(self) -> None:
        """
        Conducts training for a single episode.
        """
        # Initialize the environment for a new episode
        self.environment.initialize_for_new_episode()
        # Store the current environment reference to use throughout the episode
        self.current_sub_environment = self.environment.current_sub_environment

        current_state = self.environment.get_state()  # Get state from current sub-environment
        done = False
        total_reward = 0  # Track total reward for the episode
        step_count = 0  # Initialize step counter

        while not done and step_count < 40:  # Truncate episode after 40 steps
            # Convert the current state to a numpy array for input to the neural network
            state_array = self.state_to_array(current_state)

            # Retrieve available actions from the current sub-environment
            available_actions = self.environment.get_available_actions(current_state)

            # Print debug information: agent location, item location, available actions, and has item
            print(f"Agent Location: {current_state.agent_location}")
            print(f"Item Location: {current_state.item_location}")
            print(f"Goal Location: {current_state.goal_location}")
            print(f"Available Actions: {available_actions}")
            print(f"Has Item: {current_state.has_item}")

            # Select an action using the agent's ε-greedy policy
            action = self.agent.select_action(state_array, available_actions)

            # Print the selected action
            print(f"Selected Action: {action}")

            # Execute the action in the current sub-environment, receive reward and next state
            reward, next_state = self.environment.step(action)

            # Print the reward received after taking the action
            print(f"Reward: {reward}")

            # Add the reward to the total reward for this episode
            total_reward += reward

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

            # Increment the step counter
            step_count += 1

        # Store total reward of the episode
        self.episode_rewards.append(total_reward)

        # Store the number of steps taken in the episode
        self.episode_steps.append(step_count)  # New line added here

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

        # Ensure the state array matches the input size of the neural network
        if len(state_array) != 11:
            print(f"Warning: State array length mismatch. Expected 11, got {len(state_array)}. Padding with zeros.")
            state_array = np.pad(state_array, (0, 11 - len(state_array)), 'constant')
        return state_array


    def train(self, num_episodes: int) -> None:
        """
        Train the agent across multiple episodes.
        """
        for episode in range(num_episodes):
            print(f"Starting Episode {episode + 1}")
            self.train_one_episode()
            print(f"Episode {episode + 1} completed. Epsilon: {self.agent.epsilon:.4f}")

        # Plot and save the rewards and epsilon decay after training is complete
        self.plot_rewards(save=True, filename='reward_plot.png')
        self.plot_epsilon_decay(num_episodes, save=True, filename='epsilon_decay_plot.png')
        self.plot_training_loss(save=True, filename='training_loss_plot.png')
        self.plot_steps_per_episode(save=True,filename='steps_per_episode_plot.png')

    def plot_training_loss(self, save: bool = False, filename: str = None) -> None:
        """
        Plot the training loss over episodes.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.agent.loss_history, label='Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.legend()
        if save and filename:
            plt.savefig(filename)
            print(f"Training loss plot saved to {filename}")
        else:
            plt.show()


    def plot_rewards(self, save: bool = False, filename: str = None) -> None:
        """
        Plot the total reward earned per episode.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label='Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward Earned per Episode')
        plt.legend()
        if save and filename:
            plt.savefig(filename)
            print(f"Reward plot saved to {filename}")
        else:
            plt.show()

    def plot_epsilon_decay(self, num_episodes: int, save: bool = False, filename: str = None) -> None:
        """
        Plot the epsilon decay over episodes.
        """
        epsilons = [max(self.agent.epsilon_min, self.agent.epsilon * (self.agent.epsilon_decay ** i)) for i in range(num_episodes)]
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_episodes), epsilons, label='Epsilon Decay')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay over Episodes')
        plt.legend()
        if save and filename:
            plt.savefig(filename)
            print(f"Epsilon decay plot saved to {filename}")
        else:
            plt.show()

    def plot_steps_per_episode(self, save: bool = False, filename: str = None) -> None:
        """
        Plot the number of steps taken per episode.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_steps, label='Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Steps')
        plt.title('Number of Steps Taken per Episode During Training')
        plt.legend()
        if save and filename:
            plt.savefig(filename)
            print(f"Steps per episode plot saved to {filename}")
        else:
            plt.show()

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
