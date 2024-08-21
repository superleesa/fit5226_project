from env import Environment

env = Environment(n=5)
env.initialize_state()
print(env.get_state())