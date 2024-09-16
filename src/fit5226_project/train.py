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
        
        self.episode_rewards: list[float] = []


    def train_one_episode(self) -> None:
        """
        Conducts training for a single episode.
        """
        self.environment.initialize_for_new_episode()

        current_state = self.environment.get_state()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            state_array = self.state_to_array(current_state)
            available_actions = self.environment.get_available_actions(current_state)
            action, is_greedy, all_qvals = self.agent.select_action(state_array, available_actions)
            reward, next_state = self.environment.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals)
            # print(f"S_t={current_state}, A={action.name}, R={reward}, S_t+1={next_state}")
            print(f"R={reward}")
            print("========================")
            next_state_array = self.state_to_array(next_state)
            done = self.environment.is_goal_state(next_state)
            total_reward += reward
            
            self.agent.remember((state_array, action.value, reward, next_state_array, done))
            self.agent.replay()  # maybe train inside
            
            current_state = next_state
            step_count += 1

        # decrease exploration over time
        self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * self.agent.epsilon_decay)
        self.episode_rewards.append(total_reward)

    def state_to_array(self, state: Assignment2State) -> np.ndarray:
        """
        Converts a State object into a numpy array suitable for input to the DQN.
        """
        # Convert Assignment2State to array
        return np.array([
            *state.agent_location,  # Agent's (x, y) location
            *state.item_location,   # Item's (x, y) location
            float(state.has_item),  # 1 if agent has item, 0 otherwise
            *state.goal_location,   # Goal's (x, y) location
            *state.goal_direction,  # Direction to goal (dx, dy)
            *state.item_direction   # Direction to item (dx, dy)
        ])

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
