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

    def get_available_actions(self):
        # logic to determine available actions
        actions = []
        x, y = self.agent.location
        if x > 0:
            actions.append(np.array([x, y+1]))
        if x < self.n - 1:
            actions.append(np.array([x, y-1]))
        if y > 0:
            actions.append(np.array([x-1, y]))
        if y < self.n - 1:
            actions.append(np.array([x+1, y]))
        return actions

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
        # The start, item, goal location must be different position
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

    def move(self, action):
        pass

class Item(GridEntity):
    def __init__(self, environment):
        super().__init__(environment)
        self.icon = 'I'

# test
env = Environment(n=5)
env.initialize_state()
print(env.get_state())