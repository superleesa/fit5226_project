from env import Environment

# Import Agent inside the function where it's used
def main():
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

if __name__ == "__main__":
    main()