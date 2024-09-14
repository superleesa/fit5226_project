import random

from tqdm import tqdm

from fit5226_project.agent import Agent, Trainer, ItemObject, generate_grid_location_list, DQNAgent
from fit5226_project.env import Environment, Assignment2Environment


class Evaluation:
    def __init__(self, n=5) -> None:
        self.n = n
        self.agent = Agent(num_episode_per_intermediate_item=1000)
        item_grid_locations = generate_grid_location_list(self.n, self.n)
        all_items = [ItemObject(grid_location) for grid_location in item_grid_locations]
        self.q_learning_envs = [Environment(item = item, with_animation=False) for item in all_items]
        self.dqn_envs = Assignment2Environment(with_animation=False)

    
    def run_train(self) -> None:
        """
        Trains the agent in the environment and returns the trained agent.
        """
        trainer = Trainer(self.agent, self.q_learning_envs)
        trainer.train()

    @staticmethod
    def calculate_manhattan_distance(start_location: tuple[int, int], goal_location: tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.
        """
        start_x, start_y = start_location
        goal_x, goal_y = goal_location
        return abs(start_x - goal_x) + abs(start_y - goal_y)

    @staticmethod
    def calculate_metrics_score(shortest_distance: int, distance: int) -> float:
        """
        Calculates the proportion of the Q-learning distance to the shortest distance.
        """
        return shortest_distance / distance

    def visualize(self, num_of_vis: int = 5) -> None:
        """
        Visualize the path after trained
        """
        env_indices = random.sample(range(1, self.n*self.n), num_of_vis)
        for index in env_indices:
            env = self.q_learning_envs[index]
            env.set_with_animation(True)
            env.initialize_for_new_episode()

            # Run the agent in the environment
            current_state = env.get_state()
            while not env.is_goal_state(current_state):
                possible_actions = env.get_available_actions()
                action = self.agent.choose_action(possible_actions, current_state, self.agent.trained_qval_matrices[index], is_training=False)
                _, next_state = env.step(action)
                current_state = next_state
    
    def q_learning_performance_test(self):
        """
        Conducts a performance test for q learning
        """
        num_episodes = 0
        total_score = 0
        for env in tqdm(self.q_learning_envs):
            env.set_with_animation(False)
            for agent_locaiton in tqdm(generate_grid_location_list(self.n, self.n)):
                if agent_locaiton == env.item.location or agent_locaiton == env.goal_location:
                    continue

                env.initialize_for_new_episode(agent_location=agent_locaiton)
                start_location = env.agent.location  # Get the start location of the agent
                item_location = env.item.location  # Get intermediate location of the item

                # Calculate shortest distance from start to item to goal
                shortest_distance = (
                    self.calculate_manhattan_distance(start_location, item_location)
                    + 1 
                    + self.calculate_manhattan_distance(item_location, env.goal_location)
                )

                current_state = env.get_state()
                num_steps = 0
                while not env.is_goal_state(current_state):
                    possible_actions = env.get_available_actions()
                    item_x, item_y = current_state.item_location
                    action = self.agent.choose_action(possible_actions, current_state, self.agent.trained_qval_matrices[self.n*item_x+item_y], is_training=False)
                    _, next_state = env.step(action)
                    current_state = next_state
                    num_steps += 1

                # Calculate and accumulate the score
                total_score += self.calculate_metrics_score(shortest_distance, num_steps)
                
                num_episodes += 1

        # Return the average score across all tests
        return total_score / num_episodes
    
    def dqn_performance_test(self):
        """
        Conducts a performance test for DQN.
        """
        num_episodes = 0
        total_score = 0

        # Loop over all environments in DQN environment
        for env in tqdm(self.dqn_envs.environments):
            env.set_with_animation(False)
            for agent_location in tqdm(generate_grid_location_list(self.n, self.n)):
                if agent_location == env.item.location or agent_location == env.goal_location:
                    continue

                # Initialize episode with a given agent location
                env.initialize_for_new_episode(agent_location=agent_location)
                start_location = env.agent.location
                item_location = env.item.location

                # Calculate the shortest distance from start to item to goal
                shortest_distance = (
                    self.calculate_manhattan_distance(start_location, item_location)
                    + 1  # For picking up the item
                    + self.calculate_manhattan_distance(item_location, env.goal_location)
                )

                # Start testing the agent
                current_state = env.get_state()
                num_steps = 0
                while not env.is_goal_state(current_state):
                    action = self.agent.select_action(current_state)
                    # Take a step in the environment and observe the next state and if goal is reached
                    _, next_state = env.step(action)
                    current_state = next_state
                    num_steps += 1

                # Calculate and accumulate the score
                total_score += self.calculate_metrics_score(shortest_distance, num_steps)
                num_episodes += 1

        # Return the average score across all tests
        return total_score / num_episodes
    
    def visualize_dqn(self, num_of_vis: int = 5) -> None:
        """
        Visualize the path after trained
        """
        for _ in (0, num_of_vis):
            self.dqn_envs.initialize_for_new_episode()

            # Run the agent in the environment
            current_state = self.dqn_envs.current_sub_environment.get_state()
            while not self.dqn_envs.current_sub_environment.is_goal_state(current_state):
                # possible_actions = self.dqn_envs.current_sub_environment.get_available_actions()
                action = self.agent.select_action(current_state, is_training=False)
                _, next_state = self.dqn_envs.current_sub_environment.step(action)
                current_state = next_state
    
if __name__ == "__main__":
    # # Q Learning
    # evl = Evaluation()
    # evl.run_train()

    # # Conduct the performance test
    # average_score = evl.q_learning_performance_test()
    # print(f"Average performance score (1 is the best): {average_score:.4f}")

    # # visualize randomly the environments and show the steps of the agent
    # evl.visualize()

    # DQN
    evl = Evaluation()
    evl.run_train()

    # Conduct the performance test
    average_score = evl.dqn_performance_test()
    print(f"Average performance score (1 is the best): {average_score:.4f}")

    # visualize randomly the environments and show the steps of the agent
    evl.visualize()



