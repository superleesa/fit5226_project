import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abc import ABC, abstractmethod

class Environment:
    def __init__(self, n=5):
        self.n = n
        self.goal_location = np.array([n-1, n-1])
        self.goal = Goal(self)
        self.agent = Agent(self)
        self.item = Item(self)
        self.is_goal = False
        self.has_item = False

    def initialize_state(self):
        self.grid = np.zeros((self.n, self.n))
        self.agent.location = self.agent.place_randomly()
        self.item.location  = self.item.place_randomly(self.agent.location)
        self.goal.location = np.array([self.n-1, self.n-1])
        self.is_goal = False
        self.has_item = False

    def get_state(self):
        return {
            'Agent Location: ': self.agent.location,
            'Item Location: ': self.item.location if not self.has_item else 'Picked up',
            'Goal Location: ': self.goal.location
        }

    def is_goal_state(self, x, y):
        return np.array_equal([x, y], self.goal.location)

    def animate(self):
        fig, ax = plt.subplots(figsize = (8,8))

        def update(i):
            ax.clear()
            ax.set_xlim(-0.5, self.n - 0.5)
            ax.set_ylim(-0.5, self.n - 0.5)
            ax.set_xticks(np.arange(-0.5, self.n, 1))
            ax.set_yticks(np.arange(-0.5, self.n, 1))
            ax.grid(True)

            # Plotting the agent and goal
            ax.text(*self.agent.location[::-1], 'A', ha='center', va='center', fontsize=16, color='blue')
            if not self.has_item:
                ax.text(*self.item.location[::-1], 'I', ha='center', va='center', fontsize=16, color='green')
            ax.text(*self.goal.location[::-1], 'G', ha='center', va='center', fontsize=16, color='red')

            # Agent picks up the item if it's at the same location
            if not self.has_item and np.array_equal(self.agent.location, self.item.location):
                self.has_item = True

            # Agent moves towards the goal after picking up the item
            if not self.has_item:
                self.agent.move_towards(self.item.location)
            else:
                self.agent.move_towards(self.goal.location)

            
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Person (A)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Item (I)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Goal (G)')
            ]
            ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))


            return ax,

        plt.subplots_adjust(right=0.75, left=0.1)  # Adjust the subplot to make space for the legend
        ani = animation.FuncAnimation(fig, update, frames=range(20), interval=500, repeat=False)
        plt.show()

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

    def move_towards(self, target_location):
        x, y = self.location
        target_x, target_y = target_location
        if x < target_x:
            x += 1
        elif x > target_x:
            x -= 1
        elif y < target_y:
            y += 1
        elif y > target_y:
            y -= 1
        self.location = np.array([x, y])

class Item(GridEntity):
    def __init__(self, environment):
        super().__init__(environment)
        self.icon = 'I'

# Create the environment
env = Environment(n=5)
env.initialize_state()

# Visualize the agent's movement
env.animate()
