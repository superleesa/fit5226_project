from env import Environment
from agent import Agent, Trainer

class Evaluation:
    def __init__(self, n=5, with_animation=False) -> None:
        self.n = n
        self.with_animation = with_animation

    def run_train(self) -> None:
        environment = Environment(n=self.n, with_animation=self.with_animation) 
        agent = Agent()
        trainer = Trainer(agent, environment)
        trainer.train_for_all_items()

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
        return distance / shortest_distance
    
    def performance_test(self, num_of_tests: int = 10) -> float:
        """
        Conducts a performance test for a given number of tests and returns the average score.
        """
        total_score = 0.0

        for _ in range(num_of_tests):
            env = Environment(n=self.n)  # Create the environment
            agent = Agent()
            env.initialize_for_new_episode()

            start_location = env.agent.location  # Get the start location of the agent
            item_location = env.item.location  # Get the intermediate location of the item
            goal_location = (self.n - 1, self.n - 1)  # Set the goal location

            # Calculate the shortest distance from start to item to goal
            shortest_distance = (
                self.calculate_manhattan_distance(start_location, item_location)
                + self.calculate_manhattan_distance(item_location, goal_location)
            )

            # Run the agent in the environment
            env.initialize_for_new_episode()
            current_state = env.get_state()
            while not env.is_goal_state(current_state):
                possible_actions = env.get_available_actions()
                action = agent.choose_action(possible_actions, current_state, agent.trained_qval_matrices[0], is_training=False)
                _, next_state = env.step(action)
                current_state = next_state

            # Calculate and accumulate the score
            total_score += self.calculate_metrics_score(shortest_distance, env.num_steps)

        # Return the average score across all tests
        return total_score / num_of_tests
    
    # def performance_test(self, num_of_test = 10, n = 5):
    #     # it conducs performance test for the num_of_test times, and returns the average of the proportion

    #     # Import Agent here to avoid circular import at the top level
    #     from agent import Agent
    #     total_distance = 0
    #     for _ in num_of_test:
    #         env = Environment(n) # Create the environment
    #         agent = Agent()
    #         env.initialize_for_new_episode()
    #         start_location = env.agent.location # get the start location for agent
    #         item_location = env.item.location # get intermidate location for item
    #         goal_location = (n - 1, n - 1) # get goal locaiton
    #         # calculate shortest distance for the given agent starting locaiton, item locaiton, and goal location
    #         shortest_distance = self.calculate_manhattan_distance(start_location, item_location) + self.calculate_manhattan_distance(item_location, goal_location)
    #         total_distance += self.calculate_metrics_score(shortest_distance, env.num_steps)
    #     return total_distance/num_of_test

if __name__ == "__main__":
    # test_train_viz()
    # performance_test()
    evl = Evaluation()
    average_score = evl.performance_test(num_of_tests=10)
    print(f"Average performance score: {average_score:.4f}")