import itertools
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fit5226_project.agent import DQNAgent
from fit5226_project.env import Assignment2Environment
from fit5226_project.metrics import calculate_metrics_score
from fit5226_project.state import Assignment2State
from fit5226_project.tracker import mlflow_manager
from fit5226_project.utils import generate_grid_location_list, generate_unique_id


class Trainer:
    def __init__(
        self,
        agent: DQNAgent,
        environment: Assignment2Environment,
        update_target_interval: int = 20,
        num_episodes: int = 500,
        with_log: bool = True,
        log_reward_step: int = 100,
        num_validation_episodes: int = 30,
        save_checkpoint_interval: int = 50,  # in episodes
        validation_interval: int = 5,  # in episodes
        with_validation: bool = True,
        with_visualization: bool = True,
        save_checkpoints: bool = True,
    ) -> None:
        self.training_unique_id = generate_unique_id()

        self.agent = agent
        self.environment = environment

        self.update_target_interval = update_target_interval
        self.num_episodes = num_episodes

        self.episode_rewards: list[float] = []

        self.with_log = with_log
        self.agent.with_log = self.with_log

        self.global_step = 0
        self.log_reward_step = log_reward_step
        self.num_validation_episodes = num_validation_episodes

        self.save_checkpoint_interval = save_checkpoint_interval

        self.validation_interval = validation_interval
        self.with_validation = with_validation
        self.with_visualization = with_visualization
        self.save_checkpoints = save_checkpoints

    def train_one_episode(self, epoch_idx: int) -> None:
        """
        Conducts training for a single episode.
        """
        KILL_EPISODE_AFTER = 1

        self.environment.initialize_for_new_episode()

        current_state = self.environment.get_state()
        done = False
        total_reward = 0.0
        step_count = 0
        current_log_cycle_reward_list = []
        prev_state = None

        start_time = time.time()
        while not done:
            if time.time() - start_time > KILL_EPISODE_AFTER:
                break
            state_array = self.state_to_array(current_state)
            available_actions = self.environment.get_available_actions(current_state)
            action, is_greedy, all_qvals = self.agent.select_action(state_array, available_actions)
            reward, next_state = self.environment.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals)
            next_state_array = self.state_to_array(next_state)
            done = self.environment.is_goal_state(next_state)

            # detection of three-step cycle
            if next_state == prev_state:
                break

            self.agent.replay_buffer.remember((state_array, action.value, reward, next_state_array, done))
            self.agent.replay()  # maybe train inside

            total_reward += reward
            current_log_cycle_reward_list.append(reward)
            step_count += 1
            self.global_step += 1

            if self.with_log and self.global_step % self.log_reward_step == 0:
                running_reward = sum(current_log_cycle_reward_list) / len(current_log_cycle_reward_list)
                mlflow_manager.log_reward(running_reward, step=self.global_step)
                current_log_cycle_reward_list.clear()

            prev_state = current_state
            current_state = next_state

        # decrease exploration over time
        self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * self.agent.epsilon_decay)
        self.episode_rewards.append(total_reward)

        if self.with_log:
            mlflow_manager.log_episode_wise_reward(total_reward / step_count, episode_idx=epoch_idx)

    def state_to_array(self, state: Assignment2State) -> torch.Tensor:
        # Convert Assignment2State to array
        return torch.tensor(
            np.array(
                [
                    *state.agent_location,  # Agent's (x, y) location
                    *state.item_location,  # Item's (x, y) location
                    float(state.has_item),  # 1 if agent has item, 0 otherwise
                    *state.goal_location,  # Goal's (x, y) location
                    *state.goal_direction,  # Direction to goal (dx, dy)
                    *state.item_direction,  # Direction to item (dx, dy)
                ]
            )
        ).float()

    def train(self) -> None:
        """
        Train the agent across multiple episodes.
        """
        current_best_validation_score = -float("inf")
        for episode in range(1, self.num_episodes + 1):
            if self.with_log:
                print(f"Starting Episode {episode + 1}")
            self.train_one_episode(episode)
            if episode % self.update_target_interval == 0:
                self.agent.update_target_network()
                if self.with_log:
                    print("Target network updated")
            if self.with_log:
                print(f"Episode {episode + 1} completed. Epsilon: {self.agent.epsilon:.4f}")
            if self.with_validation and episode % self.validation_interval == 0:
                validation_score, _, _ = self.validate(episode)
                if validation_score > current_best_validation_score:
                    if self.with_log:
                        print(f"New best validation score: {validation_score}")
                    current_best_validation_score = validation_score
                    self.save_agent(episode)
                if self.with_visualization:
                    self.visualize_sample_episode(num_visualizations=1)

            if self.save_checkpoints and episode % self.save_checkpoint_interval == 0:
                self.save_agent(episode)

    def visualize_sample_episode(self, num_visualizations: int = 3) -> None:
        for _ in range(num_visualizations):
            sample_env = Assignment2Environment(n=4, with_animation=True)
            sample_env.initialize_for_new_episode()
            current_state = sample_env.get_state()
            start_time = time.time()
            done = False

            prev_state = None

            while not done and time.time() - start_time < 1 * 10:
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

            plt.close(sample_env.current_sub_environment.fig)

    def validate(
        self,
        current_episode_index: int | None = None,
        episode_samples: list[tuple[tuple[int, int], int]] | None = None,
        is_eval: bool = False,
    ) -> tuple[float, float, float]:
        """
        Sample a number of episodes and calculate the average score.
        Don't use this method when animating because we kill each episode after 0.01 seconds.

        Args:
            current_episode_index: The current episode index. (for logging purpose only)
            episode_samples: A list of tuples containing the agent location and environment index for each episode to be sampled.
            is_eval: If True, the method will use all possible combinations of agent location and environment index to calculate the metric.

        Returns: average_path_length_score, goal_reached_percentage, average_reward
        """
        KILL_EPISODE_AFTER = 0.01

        # is is_eval, we will use all possible combinations of agent, item, and goal locations i.e. consider all possible states
        episode_samples = (
            episode_samples
            if episode_samples is not None
            else list(
                itertools.product(
                    generate_grid_location_list(self.environment.n, self.environment.n),
                    range(len(self.environment.environments)),
                )
            )
            if is_eval
            else None
        )
        num_episodes = len(episode_samples) if episode_samples is not None else self.num_validation_episodes
        calulated_scores = []
        num_failed_episodes = total_predicted_steps = 0
        total_rewards = 0.0

        # we use the same environment as trainer to ensure that we use the same env parameters in validation
        sample_env = deepcopy(self.environment)
        sample_env.set_with_animation(False)
        for sample_episode_idx in range(num_episodes):
            if episode_samples is not None:
                sample_env.initialize_for_new_episode(
                    agent_location=episode_samples[sample_episode_idx][0],
                    agent_has_item=False,  # metric assumes that agent starts without item
                    env_index=episode_samples[sample_episode_idx][1],
                )
            else:
                sample_env.initialize_for_new_episode()

            sample_env.current_sub_environment.agent.has_item = False  # metric assumes that agent starts without item
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
                total_predicted_steps += 1
                total_rewards += reward

            calulated_scores.append(
                calculate_metrics_score(predicted_steps, start_location, item_location, goal_location)
            )

            if is_failed:
                num_failed_episodes += 1

        average_path_length_score = sum(calulated_scores) / num_episodes if num_episodes != 0 else 0
        goal_reached_percentage = (num_episodes - num_failed_episodes) / num_episodes if num_episodes != 0 else 0
        average_reward = total_rewards / total_predicted_steps if total_predicted_steps != 0 else 0

        if self.with_log and current_episode_index is not None:
            mlflow_manager.log_validation_score(average_path_length_score, step=current_episode_index)
            mlflow_manager.log_num_failed_validation_episodes(num_failed_episodes, step=current_episode_index)

        return average_path_length_score, goal_reached_percentage, average_reward

    def save_agent(self, episode_index: int | None = None) -> None:
        checkpoint_name = f"episode_{episode_index}" if episode_index else "checkpoint"
        save_path = Path(f"checkpoints/{self.training_unique_id}/{checkpoint_name}.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_state(save_path)
