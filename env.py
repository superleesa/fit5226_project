import numpy as np

class Environment:
    def __init__(self, n=5):
        self.n = n
        self.goal_location = (n-1, n-1)
        self.goal = Goal(self)
        self.agent = Agent(self)
        # self.item = Item(self)

    def initialize_state(self):
        self.grid = np.zeros((self.n, self.n)) 
        r = self.item.place_randomly()
        self.agent.place_randomly(r)
        # self.grid[self.goal.location] = 3

    def get_available_actions(self):
        # logic to determine available actions
        pass

    def animate(self):
        pass

    def step(self):
        self.animate()

class Goal:
    def __inint__(self, environment):
        self.location = environment.goal_location

class MapObject: # abstract
    pass

class Agent:
    def __init__(self, environment) -> None:
        self.environment = environment
        self.location = self.place_rondomly()

    def place_rondomly(self):
        start_location = None
        # check if the start location is not on the goal location
        while start_location != None and start_location != self.environment.goal_location: 
            start_location = np.random.randint(0, self.environment.n, size=2)
        return start_location

class Item:
    def __init__(self, environment):
        self.environment = environment
        self.location = np.array([0, 0])

    def place_rondomly(self):
        pass
#         self.location = np.random.randint(0, self.environment.n, size=2)
#         while np.array_equal(self.location, self.environment.goal.location):
#             self.location = np.random.randint(0, self.environment.size, size=2)


# test
env = Environment(n=5)
# state = env.reset()
# done = False
# while not done:
#     action = np.random.choice(env.get_available_actions())
#     state, reward, done = env.step(action)
#     print(f"State: {state}, Reward: {reward}, Done: {done}")

