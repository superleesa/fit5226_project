import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fit5226_project.metrics import calculate_metrics_score
from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import Assignment2State
from fit5226_project.tracker import mlflow_manager

class Trainer:
    def __init__(
        self,
        agent: DQNAgent, 
        environment: Assignment2Environment, 
        with_log: bool = True, 
        log_step: int = 100, 
        update_target_episodes: int = 20, 
        num_validation_episodes: int = 30, 
        save_checkpoint_interval: int = 50,  # in episodes
        validation_interval: int = 5,  # in episodes
        with_visualization: bool = True,
    ) -> None:
        """
        Initialize the Trainer with the DQN agent and environment.
        """
        self.agent = agent
        self.environment = environment
        
        self.update_target_episodes = update_target_episodes
        
        self.episode_rewards: list[float] = []
        
        self.with_log = with_log
        self.global_step = 0
        self.log_step = log_step
        self.num_validation_episodes = num_validation_episodes
        
        self.save_checkpoint_interval = save_checkpoint_interval
        
        self.validation_interval = validation_interval
        self.with_visualization = with_visualization


    def train_one_episode(self, epoch_idx: int) -> None:
        """
        Conducts training for a single episode.
        """
        self.environment.initialize_for_new_episode()

        current_state = self.environment.get_state()
        done = False
        total_reward = 0.0
        step_count = 0
        current_log_cycle_reward_list = []

        while not done:
            state_array = self.state_to_array(current_state)
            available_actions = self.environment.get_available_actions(current_state)
            action, is_greedy, all_qvals = self.agent.select_action(state_array, available_actions)
            reward, next_state = self.environment.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals)
            next_state_array = self.state_to_array(next_state)
            done = self.environment.is_goal_state(next_state)
            self.agent.replay_buffer.remember((state_array, action.value, reward, next_state_array, done))
            self.agent.replay()  # maybe train inside
            
            total_reward += reward
            current_log_cycle_reward_list.append(reward)
            step_count += 1
            self.global_step += 1
            
            # print(f"S_t={current_state}, A={action.name}, R={reward}, S_t+1={next_state}")
            if self.with_log and self.global_step % self.log_step == 0:
                # print(f"R={reward}")
                # print("========================")
                running_reward = sum(current_log_cycle_reward_list) / len(current_log_cycle_reward_list)
                # mlflow.log_metric("reward", running_reward, step=self.global_step)
                mlflow_manager.log_reward(running_reward, step=self.global_step)
                current_log_cycle_reward_list.clear()
            
            current_state = next_state

        # decrease exploration over time
        self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * self.agent.epsilon_decay)
        self.episode_rewards.append(total_reward)
        mlflow_manager.log_episode_wise_reward(total_reward/step_count, episode_idx=epoch_idx)

    def state_to_array(self, state: Assignment2State) -> torch.Tensor:
        """
        Converts a State object into a numpy array suitable for input to the DQN.
        """
        # Convert Assignment2State to array
        return torch.tensor(np.array([
            *state.agent_location,  # Agent's (x, y) location
            *state.item_location,   # Item's (x, y) location
            float(state.has_item),  # 1 if agent has item, 0 otherwise
            *state.goal_location,   # Goal's (x, y) location
            *state.goal_direction,  # Direction to goal (dx, dy)
            *state.item_direction   # Direction to item (dx, dy)
        ])).float()

    def train(self, num_episodes: int) -> None:
        """
        Train the agent across multiple episodes.
        """
        
        current_best_validation_score = -float('inf')
        for episode in range(1, num_episodes+1):
            print(f"Starting Episode {episode + 1}")
            self.train_one_episode(episode)
            if episode % self.update_target_episodes == 0:
                self.agent.update_target_network()
                if self.with_log:
                    print("Target network updated")
            print(f"Episode {episode + 1} completed. Epsilon: {self.agent.epsilon:.4f}")
            if self.agent.steps % self.validation_interval == 0:
                validation_score, num_failed_episodes = self.validate(episode)
                if validation_score > current_best_validation_score:
                    print(f"New best validation score: {validation_score}")
                    current_best_validation_score = validation_score
                    self.save_agent(episode)
                if self.with_visualization:
                    self.visualize_sample_episode()
                
            if episode % self.save_checkpoint_interval == 0:
                self.save_agent(episode)
                
        
        # Plot and save the rewards and epsilon decay after training is complete
        self.plot_rewards(save=True, filename='reward_plot.png')
        self.plot_epsilon_decay(num_episodes, save=True, filename='epsilon_decay_plot.png')

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

    def validate(self, current_episode_index: int) -> tuple[float, float]:
        """
        Don't use this method when animating because we kill each episode after 0.01 seconds.
        """
        KILL_EPISODE_AFTER = 0.01
        
        calulated_scores = []
        num_failed_episodes = 0
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
            is_failed = False
            while not done:
                # we can't always detect cycles (unless we track the whole path and use set)
                # which is expensive so we just kill the episode after a certain time
                if time.time() - start_time > KILL_EPISODE_AFTER:
                    predicted_steps = 0
                    is_failed = True
                    break
                state_array = self.state_to_array(current_state)
                available_actions = sample_env.get_available_actions(current_state)
                action, is_greedy, all_qvals = self.agent.select_action(state_array, available_actions, is_test=True)
                reward, next_state = sample_env.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals)
                done = sample_env.is_goal_state(next_state)
                
                # check for three-step cycle and stop early
                if next_state == prev_state:
                    predicted_steps = 0
                    is_failed = True
                    break
                prev_state = current_state
                current_state = next_state
                predicted_steps += 1
            calulated_scores.append(calculate_metrics_score(predicted_steps, start_location, item_location, goal_location))
            if is_failed:
                num_failed_episodes += 1
        
        result = sum(calulated_scores) / self.num_validation_episodes
        if self.with_log:
            mlflow_manager.log_validation_score(result, step=current_episode_index)
            mlflow_manager.log_num_failed_validation_episodes(num_failed_episodes, step=current_episode_index)
        return result, num_failed_episodes

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