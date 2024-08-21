import numpy as np
from env import Environment
from agent import Agent
from state import State

def execute():
    
    # train
    env = Environment(n=5)
    agent = Agent(alpha=0.1, discount_rate=0.9, epsilon=0.1, num_episode_per_intermediate_item=100)
    print("Starting training...")
    agent.train_one_intermediate_item()  # This will use the alpha, discount_rate, and epsilon defined in the Agent class
    print("Training completed.")

    # for inference
    env.initialize_state()
    state = env.get_state()
    total_steps = 0
    qval_matrix = np.zeros((env.n, env.n, 4))

    # Run inference
    print("Starting inference...")
    while not env.is_goal_state(state):
        action = agent.choose_action(qval_matrix, state, is_training=False)
        next_state = env.get_next_state_from_action(state, action) 
        state = next_state
        total_steps += 1
        print(f"Step: {total_steps}, Agent Location: {state.agent_location}, Item Location: {state.item_location}")

    print("Goal reached!")

if __name__ == "__main__":
    execute()

#Things before 2am, will check changes later
