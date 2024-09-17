import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from fit5226_project.metrics import calculate_metrics_score
from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import Assignment2State
from fit5226_project.plotter import Plotter

class Trainer:
    def __init__(
        self,
        agent: DQNAgent, 
        environment: Assignment2Environment, 
        with_log: bool = False, 
        log_step: int = 100, 
        update_target_episodes: int = 20, 
        num_validation_episodes: int = 30, 
        save_checkpoint_interval: int = 50,  # in episodes
    ) -> None:
        """
        Initialize the Trainer with the DQN agent and environment.
        """
        self.agent = agent
        self.environment = environment
        
        self.update_target_episodes = update_target_episodes
        
        self.episode_rewards: list[float] = []
        self.validation_scores: list[float] = []
        
        self.with_log = with_log
        self.global_step = 0
        self.log_step = log_step
        self.num_validation_episodes = num_validation_episodes
        
        self.save_checkpoint_interval = save_checkpoint_interval

        # Initialize the Plotter
        self.plotter = Plotter(save_dir="./plots")


    def train_one_episode(self, epoch_idx: int) -> None:
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
            next_state_array = self.state_to_array(next_state)
            done = self.environment.is_goal_state(next_state)
            total_reward += reward
            
            self.agent.replay_buffer.remember((state_array, action.value, reward, next_state_array, done))
            self.agent.replay()  # maybe train inside
            
            current_state = next_state
            step_count += 1
            self.global_step += 1

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
        num_nn_passes = 0
        current_best_validation_score = -float('inf')

        for episode in range(1, num_episodes+1):
            print(f"Starting Episode {episode + 1}")
            self.train_one_episode(episode)

            if episode % 10 == 0:
                # print(self.agent.logged_data)
                self.plotter.update_plot(self.agent.logged_data)

            if episode % self.update_target_episodes == 0:
                self.agent.update_target_network()
                if self.with_log:
                    print("Target network updated")
            print(f"Episode {episode + 1} completed. Epsilon: {self.agent.epsilon:.4f}")
            if self.agent.steps != num_nn_passes:
                validation_score = self.validate(episode)
                self.validation_scores.append(validation_score)
                if validation_score > current_best_validation_score:
                    print(f"New best validation score: {validation_score}")
                    current_best_validation_score = validation_score
                    self.save_agent(episode)
                # self.visualize_sample_episode()
                num_nn_passes = self.agent.steps
            if episode % self.save_checkpoint_interval == 0:
                self.save_agent(episode)
                
        
        # Plot and save the rewards and epsilon decay after training is complete
        self.plot_rewards(save=True, filename='reward_plot.png')
        self.plot_epsilon_decay(num_episodes, save=True, filename='epsilon_decay_plot.png')
        self.plot_validation_scores(save=True, filename='validation_score_plot.png')  # Plot validation scores
        self.plotter.update_plot(self.agent.logged_data)

    def visualize_sample_episode(self) -> None:
        sample_env = Assignment2Environment(n=4, with_animation=True)
        sample_env.initialize_for_new_episode()
        current_state = sample_env.get_state()
        start_time = time.time()
        done = False
        
        prev_state = None
        
        while not done and time.time() - start_time < 1*20:
            state_array = self.state_to_array(current_state)
            available_actions = sample_env.get_available_actions(current_state)
            action, is_greedy, all_qvals = self.agent.select_action(state_array, available_actions, is_test=True)
            reward, next_state = sample_env.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals)
            done = sample_env.is_goal_state(next_state)
            
            # check for three-step cycle and stop early
            if next_state == prev_state:
                print("cycle detected... breaking")
                break
            prev_state = current_state
            current_state = next_state
        
        plt.close('all')

    def validate(self, current_episode_index: int):
        calulated_scores = []
        for _ in range(self.num_validation_episodes):
            sample_env = Assignment2Environment(n=4, with_animation=False)
            sample_env.initialize_for_new_episode()
            sample_env.current_sub_environment.agent.has_item = False # metric assumes that agent starts without item
            current_state = sample_env.get_state()
            start_time = time.time()
            done = False
            start_location = sample_env.current_sub_environment.agent.get_location()
            item_location = sample_env.current_sub_environment.item.get_location()
            goal_location = sample_env.current_sub_environment.goal_location
            
            prev_state = None
            predicted_steps = 0
            while not done:
                if time.time() - start_time > 20:
                    predicted_steps = 0
                    break
                state_array = self.state_to_array(current_state)
                available_actions = sample_env.get_available_actions(current_state)
                action, is_greedy, all_qvals = self.agent.select_action(state_array, available_actions, is_test=True)
                reward, next_state = sample_env.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals)
                done = sample_env.is_goal_state(next_state)
                
                # check for three-step cycle and stop early
                if next_state == prev_state:
                    predicted_steps = 0
                    break
                prev_state = current_state
                current_state = next_state
                predicted_steps += 1
            calulated_scores.append(calculate_metrics_score(predicted_steps, start_location, item_location, goal_location))
        
        result = sum(calulated_scores) / self.num_validation_episodes


        return result

    def save_agent(self, episode_index: int) -> None:
        save_path = Path(f"checkpoints/episode_{episode_index}.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_state(save_path)

    def plot_rewards(self, save: bool = False, filename: str | None = None) -> None:
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

    def plot_validation_scores(self, save: bool = False, filename: str | None = None) -> None:
        """
        Plot the validation scores over episodes.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.validation_scores, label='Validation Score per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Validation Score')
        plt.title('Validation Score Over Episodes')
        plt.legend()
        if save and filename:
            plt.savefig(filename)
            print(f"Validation score plot saved to {filename}")
        else:
            plt.show()

    def plot_epsilon_decay(self, num_episodes: int, save: bool = False, filename: str | None = None) -> None:
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
