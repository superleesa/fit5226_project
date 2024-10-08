import numpy as np
from fit5226_project.env import Environment

# Import Agent inside the function where it's used
def test_train_viz():
    # Import Agent here to avoid circular import at the top level
    from fit5226_project.agent import Agent

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


if __name__ == "__main__":
    # test_train_viz()
    from fit5226_project.agent import Agent, Trainer, ItemObject, generate_grid_location_list

    agent = Agent()
    item_grid_locations = generate_grid_location_list(5, 5)
    all_items = [ItemObject(grid_location) for grid_location in item_grid_locations]
    envs = [Environment(item = item, with_animation=True) for item in all_items]
    trainer = Trainer(agent, envs)
    trainer.train()
