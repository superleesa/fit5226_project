import numpy as np
from abc import ABC, abstractmethod

class Environment:
    def __init__(self, n=5):
        self.n = n
        self.goal_location = np.array([n-1, n-1])
        self.goal = Goal(self)
        self.agent = Agent(self)
        self.item = Item(self)

    def initialize_state(self):
        self.grid = np.zeros((self.n, self.n))
        self.agent.location = self.agent.place_randomly()
        self.item.location  = self.item.place_randomly(self.agent.location)
        self.goal.location = np.array([self.n-1, self.n-1])

    def get_state(self):
        return {
            'Agent Location: ': self.agent.location,
            'Item Location: ': self.item.location,
            'Goal Location: ': self.goal.location
        }

    def get_available_actions(self, action):
        # logic to determine available actions
        # if action == 'up':
        # elif action == 'down':
        # elif action == 'right':
        # elif action == 'left':
        pass

    def animate(self):
        pass

    def step(self):
        self.animate()

class Goal:
    def __init__(self, environment):
        self.location = environment.goal_location

class GridEntity(ABC):
    def __init__(self, environment):
        self.environment = environment
        self.icon = None
        self.location = None
    
    def place_randomly(self, another_entity=None):
        location = None
        if another_entity is None:
            while location is None or np.array_equal(location, self.environment.goal_location):
                location = np.random.randint(0, self.environment.n, size=2)
        else:
            while location is None or np.array_equal(location, self.environment.goal_location) or np.array_equal(location, another_entity):
                location = np.random.randint(0, self.environment.n, size=2)
        return location

class Agent(GridEntity):
    def __init__(self, environment) -> None:
        super().__init__(environment)
        self.icon = 'A'

class Item(GridEntity):
    def __init__(self, environment):
        super().__init__(environment)
        self.icon = 'I'

# test
env = Environment(n=5)
env.initialize_state()
print(env.get_state())

