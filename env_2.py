import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
        self.fig, self.ax = plt.subplots(figsize=(8, 8))  # Create figure and axis once

    def initialize_state(self):
        self.grid = np.zeros((self.n, self.n))
        self.agent.location = self.agent.place_randomly()
        self.item.location  = self.item.place_randomly(self.agent.location)
        self.goal.location = np.array([self.n-1, self.n-1])
        self.is_goal = False
        self.has_item = False
        self.setup_buttons()  # NOTE: Setup buttons only once, CAN BE REMOVED OR COMMENTED
        self.animate()  # Initial drawing of the grid

    '''NOTE: This part can be removed and movement can directly be called using step()
             when ready with training.
    '''
    def setup_buttons(self):
        # Buttons for movement (added once)
        ax_up = plt.axes([0.45, 0.02, 0.1, 0.075])
        ax_down = plt.axes([0.45, 0.12, 0.1, 0.075])
        ax_left = plt.axes([0.35, 0.07, 0.1, 0.075])
        ax_right = plt.axes([0.55, 0.07, 0.1, 0.075])

        self.btn_up = Button(ax_up, 'Up')
        self.btn_down = Button(ax_down, 'Down')
        self.btn_left = Button(ax_left, 'Left')
        self.btn_right = Button(ax_right, 'Right')

        self.btn_up.on_clicked(lambda event: self.step('up'))
        self.btn_down.on_clicked(lambda event: self.step('down'))
        self.btn_left.on_clicked(lambda event: self.step('left'))
        self.btn_right.on_clicked(lambda event: self.step('right'))

    def step(self, direction):
        # Update the agent's location based on the direction
        self.agent.move(direction)
        self.animate()  # Update the grid with new agent position

    def animate(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.n)  # Set x-axis limits from 0 to n (0 to 5)
        self.ax.set_ylim(self.n, 0)  # Invert y-axis: start from n (top) to 0 (bottom)
        self.ax.set_xticks(np.arange(0, self.n + 1, 1))  # Set x-ticks from 0 to n (0 to 5)
        self.ax.set_yticks(np.arange(0, self.n + 1, 1))  # Set y-ticks from 0 to n (0 to 5))
        self.ax.grid(True)

        # Plotting the agent, item, and goal, centered in the grid cells
        self.ax.text(self.agent.location[1] + 0.5, self.agent.location[0] + 0.5, 'A',
                    ha='center', va='center', fontsize=16, color='blue')
        if not self.has_item:
            self.ax.text(self.item.location[1] + 0.5, self.item.location[0] + 0.5, 'I',
                        ha='center', va='center', fontsize=16, color='green')
        self.ax.text(self.goal.location[1] + 0.5, self.goal.location[0] + 0.5, 'G',
                    ha='center', va='center', fontsize=16, color='red')

        # Add legends to explain the symbols outside the grid
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Person (A)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Item (I)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Goal (G)')
        ]
        self.ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(right=0.75, left=0.1)  # Adjust the subplot to make space for the legend

        self.fig.canvas.draw_idle()  # Redraw the current figure

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

    def move(self, direction):
        x, y = self.location
        if direction == 'up' and x > 0:
            x -= 1
        elif direction == 'down' and x < self.environment.n - 1:
            x += 1
        elif direction == 'left' and y > 0:
            y -= 1
        elif direction == 'right' and y < self.environment.n - 1:
            y += 1
        self.location = np.array([x, y])

class Item(GridEntity):
    def __init__(self, environment):
        super().__init__(environment)
        self.icon = 'I'

# Create the environment
env = Environment(n=5)
env.initialize_state()

# Start the event loop
plt.show()
