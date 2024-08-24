from env import Environment
from agent import Agent, Trainer

class Evaluation:
    def __init__(self, n=5, with_animation=False) -> None:
        self.n = n
        self.with_animation = with_animation

    def run_train(self) -> tuple[Environment, Agent]:
        """
        Trains the agent in the environment and returns the trained agent.
        """
        environment = Environment(n=self.n, with_animation=self.with_animation) 
        agent = Agent()
        trainer = Trainer(agent, environment)
        trainer.train_for_all_items()
        return environment, agent

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

    def performance_test(self, agent: Agent, env: Environment, num_of_tests: int = 10) -> float:
        """
        Conducts a performance test for a given number of tests and returns the average score.
        """
        total_score = 0.0

        for _ in range(num_of_tests):
            env.initialize_for_new_episode()

            start_location = env.agent.location  # Get the start location of the agent
            item_location = env.item.location  # Get intermediate location of the item
            goal_location = (self.n - 1, self.n - 1)  # Set the goal location

            # Calculate shortest distance from start to item to goal
            shortest_distance = (
                self.calculate_manhattan_distance(start_location, item_location)
                + self.calculate_manhattan_distance(item_location, goal_location)
            )

            # Run the agent in the environment
            current_state = env.get_state()
            while not env.is_goal_state(current_state):
                possible_actions = env.get_available_actions()
                print(agent.trained_qval_matrices)
                x, y = current_state.agent_location
                action = agent.choose_action(possible_actions, current_state, agent.trained_qval_matrices[x+y], is_training=False)
                _, next_state = env.step(action)
                current_state = next_state

            # Calculate and accumulate the score
            total_score += self.calculate_metrics_score(shortest_distance, env.num_steps)

        # Return the average score across all tests
        return total_score / num_of_tests

if __name__ == "__main__":
    evl = Evaluation()

    # Train the agent first
    trained_environment, trained_agent = evl.run_train()

    # Conduct the performance test
    average_score = evl.performance_test(agent=trained_agent, env=trained_environment, num_of_tests=10)
    print(f"Average performance score: {average_score:.4f}")
