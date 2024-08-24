import numpy as np
from env import Environment

# Import Agent inside the function where it's used
def test_train_viz():
    # Import Agent here to avoid circular import at the top level
    from agent import Agent

    # Create the environment
    env = Environment(n=5)

    # Initialize the agent
    agent = Agent()
    agent.qval_matrix = np.zeros((5, 5, 4))  # Dummy matrix for the example

    # Example of using the step function with animation
    env.initialize_for_new_episode()
    for _ in range(10):  # Assuming a maximum of 10 steps for demonstration
        possible_actions = env.get_available_actions()
        action = agent.choose_action(possible_actions, env.get_state(), agent.qval_matrix, is_training=True)
        env.step(action)  # Each step will now trigger an animation update

def calculate_manhattan_distance(start_location: tuple[int, int], goal_location: tuple[int, int]):
    # calculate shortest distance from between the locations using manhattan distance
    start_x, start_y = start_location
    goal_x, goal_y = goal_location
    distance = abs(start_x - goal_x) + abs(start_y - goal_y)
    return distance

def calculate_metrics_score(shortest_distance, distance):
    # calculate the proportion the distance (q-learning / shortest distance)
    return distance/shortest_distance

def performance_test(num_of_test = 100, n = 5):
    # it conducs performance test for the num_of_test times, and returns the average of the proportion
    total_distance = 0
    for _ in num_of_test:
        env = Environment(n) # Create the environment
        start_location = env.agent.location # get the start location for agent
        item_location = env.item.location # get intermidate location for item
        goal_location = (n - 1, n - 1) # get goal locaiton
        # calculate shortest distance for the given agent starting locaiton, item locaiton, and goal location
        shortest_distance = calculate_manhattan_distance(start_location, item_location) + calculate_manhattan_distance(item_location, goal_location)
        total_distance += calculate_metrics_score(shortest_distance, )
    return total_distance/num_of_test


if __name__ == "__main__":
    test_train_viz()
    # performance_test() 