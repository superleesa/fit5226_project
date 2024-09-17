import random
import numpy as np
from tqdm import tqdm
import time
import os

from fit5226_project.metrics import calculate_metrics_score
from fit5226_project.agent import DQNAgent
from fit5226_project.train import Trainer
from fit5226_project.env import Assignment2Environment
from fit5226_project.state import Assignment2State, State
from fit5226_project.actions import Action

class Evaluation:
    def __init__(self, n=4) -> None:
        self.n = n
        self.dqn_envs = Assignment2Environment(n=4, with_animation=False)
        self.dqn_agent = DQNAgent(with_log=True)

    def run_dqn_train(self):
        """
        Trains DQN agent in the environment and save the states.
        """
        trainer = Trainer(self.dqn_agent, self.dqn_envs, with_log=True)
        trainer.train(num_episodes=110)
        self.dqn_agent.save_state("trained_dqn.pth")

    def load_trained_dqn(self, path: str):
        """
        Load the saved DQN
        """
        self.dqn_agent.load_state(path)
    
    @staticmethod
    def generate_grid_location_list(max_x: int, max_y) -> list[tuple[int, int]]:
        """
        Generate the grid location list for all possible cases
        """
        return [(i, j) for i in range(max_x) for j in range(max_y)]
    
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
    
    def dqn_performance_test(self):
        """
        Conducts a performance test for DQN. The maximum of the score is 1. 
        """
        num_episodes = 0
        total_score = 0

        # Loop over all environments in DQN environment
        for i, env in tqdm(enumerate(self.dqn_envs.environments)):
            env.set_with_animation(False)
            for agent_location in tqdm(self.generate_grid_location_list(self.n, self.n)):
                self.dqn_envs.current_sub_environment.agent.has_item = False # metric assumes that agent starts without item
                # Ensure agent location is not same place with item and goal
                if agent_location == env.item.location or agent_location == env.goal_location:
                    continue

                # Initialize episode with a given agent location
                self.dqn_envs.initialize_for_new_episode(agent_location=agent_location, index=i)

                # Get start, item, and goal location to calcurate distance
                start_location = self.dqn_envs.current_sub_environment.agent.get_location()
                item_location = self.dqn_envs.current_sub_environment.item.get_location()
                goal_location = self.dqn_envs.current_sub_environment.goal_location
                
                current_state = self.dqn_envs.get_state() # get current state
                start_time = time.time() # to keeps track time
                predicted_steps = 0 # count the number of actual steps taken
                done = False # for one environment
                is_break = False # to keep track the break

                while not done:
                    # Break if it takes more than 20 seconds
                    if time.time() - start_time > 20:
                        is_break = True
                        break
                    state_array = self.state_to_array(current_state) # get the states in array format
                    available_actions = self.dqn_envs.get_available_actions(current_state) # get available actions
                    action, is_greedy, all_qvals = self.dqn_agent.select_action(state_array, available_actions, is_test=True)
                    reward, next_state = self.dqn_envs.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals) # get next state
                    done = self.dqn_envs.is_goal_state(next_state) # Check if it is goal position
                    current_state = next_state # update current state
                    predicted_steps += 1

                if not is_break:
                    # calculate the metrics score
                    total_score += calculate_metrics_score(predicted_steps, start_location, item_location, goal_location)
                    num_episodes += 1 # increase the episode
            
            # Return the average score across all possible tests
            return total_score / num_episodes
    
    def visualize_dqn(self, num_of_vis: int = 5) -> None:
        """
        Visualize the path after trained for given times
        """
        for _ in (0, num_of_vis):
            self.dqn_envs.set_with_animation(True) # Set the animation True
            self.dqn_envs.initialize_for_new_episode()
            self.dqn_envs.current_sub_environment.agent.has_item = False # Assumes that agent starts without item

            current_state = self.dqn_envs.get_state()
            start_time = time.time() # Keep track time
            done = False
            
            while not done:
                # If it takes more than 20 seconds to reach the goal, break the loop
                if time.time() - start_time > 20:
                    break
                state_array = self.state_to_array(current_state) # get the states in array format
                available_actions = self.dqn_envs.get_available_actions(current_state) # get available actions
                action, is_greedy, all_qvals = self.dqn_agent.select_action(state_array, available_actions, is_test=True)
                reward, next_state = self.dqn_envs.step(action=action, is_greedy=is_greedy, all_qvals=all_qvals) # get next state
                done = self.dqn_envs.is_goal_state(next_state) # Check if it is goal position
                current_state = next_state # update current state

if __name__ == "__main__":
    # DQN
    evl = Evaluation()

    # Training DQN
    # evl.run_dqn_train()

    # Load DQN model 
    current_path = os.getcwd() # get current path
    saved_path = current_path+'/trained_dqn.pth'
    evl.load_trained_dqn(saved_path)

    # Conduct the performance test
    average_score = evl.dqn_performance_test()
    print(f"Average performance score (1 is the best): {average_score:.4f}")

    # Visualize randomly the environments and show the steps of the agent
    evl.visualize_dqn()