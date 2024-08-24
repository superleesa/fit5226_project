from __future__ import annotations
from abc import ABC
from random import randint

import numpy as np
import matplotlib.pyplot as plt

from actions import Action
from state import State


DEFAULT_TIME_PENALTY = -1
GOAL_STATE_REWARD = 100


class Environment:
    # NOTE: currently action is an integer, but we might want to change it to enum
    def __init__(self, n: int = 5, item: ItemObject | None = None, time_penalty: int | float = DEFAULT_TIME_PENALTY, goal_state_reward: int | float = GOAL_STATE_REWARD) -> None:
        self.n = n
        self.time_penalty = time_penalty
        self.goal_state_reward = goal_state_reward

        self.item = ItemObject() if item is None else item
        self.agent = AgentObject()

        if self.item.location is None:
            self.item.set_location_randomly(self.n, self.n)

        self.state: State
        # TODO: possibly implmeent this if there are multiple GridObjects to check for
        # initialize grid and put grid objects on the grid
        # x_agent, y_agent = self.agent.location
        # x_item, y_item = self.item.location
        # self.grid = np.zeros((self.n, self.n))
        # self.grid[x_agent, y_agent] = self.agent
        # self.grid[x_item, y_item] = self.item

        # Setup for animation
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def initialize_for_new_episode(self) -> None:
        self.agent.set_location_randomly(self.n, self.n, [self.item.location])
        self.state = State(self.agent.location, self.item.location)
        self.animate()  # Initial drawing of the grid

    def get_state(self) -> State:
        return self.state

    def get_available_actions(self) -> list[Action]:
        """
        Assumes that the current state is not the goal state
        """
        # logic to determine available actions
        actions = []
        current_state = self.get_state()
        x, y = current_state.agent_location
        
        if current_state.agent_location == current_state.item_location:
            # item reached state
            if x > 0:
                actions.append(Action.GOT_ITEM_LEFT)  # left
            if x < self.n - 1:
                actions.append(Action.GOT_ITEM_RIGHT)  # right
            if y > 0:
                actions.append(Action.GOT_ITEM_DOWN)  # down
            if y < self.n - 1:
                actions.append(Action.GOT_ITEM_UP)  # up
        else:
            if x > 0:
                actions.append(Action.LEFT)  # left
            if x < self.n - 1:
                actions.append(Action.RIGHT)  # right
            if y > 0:
                actions.append(Action.DOWN)  # down
            if y < self.n - 1:
                actions.append(Action.UP)  # up
        
        return actions

    def get_reward(self, state: State):
        # TODO: technically, i think it should accept (prev state, action, next state)
        return self.goal_state_reward if self.is_goal_state(state) else self.time_penalty

    def get_next_state(self, action: Action) -> State:
        self.agent.move(action)
        self.state = State(self.agent.location, self.item.location)
        return self.state

    def is_goal_state(self, state: State) -> bool:
        return (
            self.item.location == state.agent_location
        )  # we treat the item location as the goal location

    def animate(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(self.n, 0)
        self.ax.set_xticks(np.arange(0, self.n + 1, 1))
        self.ax.set_yticks(np.arange(0, self.n + 1, 1))
        self.ax.grid(True)

        # Plotting the agent, item, and goal
        self.ax.text(self.agent.location[1] + 0.5, self.agent.location[0] + 0.5, 'A',
            ha='center', va='center', fontsize=16, color='blue')
        self.ax.text(self.item.location[1] + 0.5, self.item.location[0] + 0.5, 'G',
            ha='center', va='center', fontsize=16, color='green')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Agent (A)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Dummy Goal (G)'),
        ]
        self.ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(right=0.75, left=0.1)
        self.fig.canvas.draw_idle()
        plt.pause(0.5)  # Pause to allow visualization of the movement
         
    def step(self, action: Action) -> tuple[float, State]:
        next_state = self.get_next_state(action)
        self.animate()
        reward = self.get_reward(next_state)
        return reward, next_state


class InferenceEnvironment(Environment):
    """
    environment used during inference, that represent the environement of the actual problem world
    """
    def __init__(self, n: int = 5, item: ItemObject | None = None):
        super().__init__(n, item, DEFAULT_TIME_PENALTY, GOAL_STATE_REWARD)  # note: during inference, we don't use rewards
        self.goal_location = (self.n - 1, self.n - 1)  # Set the goal state location to (n-1, n-1)

    def animate(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(self.n, 0)
        self.ax.set_xticks(np.arange(0, self.n + 1, 1))
        self.ax.set_yticks(np.arange(0, self.n + 1, 1))
        self.ax.grid(True)

        # Plotting the agent, item, and goal
        self.ax.text(self.agent.location[1] + 0.5, self.agent.location[0] + 0.5, 'A',
            ha='center', va='center', fontsize=16, color='blue')
        self.ax.text(self.item.location[1] + 0.5, self.item.location[0] + 0.5, 'I',
            ha='center', va='center', fontsize=16, color='green')
        self.ax.text(self.goal_location[1] + 0.5, self.goal_location[0] + 0.5, 'G',
            ha='center', va='center', fontsize=16, color='red')

        handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Agent (A)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Item (I)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Goal (G)')
        ]
        self.ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(right=0.75, left=0.1)
        self.fig.canvas.draw_idle()
        plt.pause(0.5)  # Pause to allow visualization of the movement


class GridObject(ABC):
    def __init__(self, location: tuple[int, int] | None = None) -> None:
        self.icon: str
        self.location = location  # NOTE: location is a tuple of (x, y) where x and y are coordinates on the grid (not indices)

    def set_location_randomly(
        self, max_x: int, max_y: int, disallowed_locations: list[tuple[int, int]] = []
    ) -> tuple[int, int]:
        """
        Note: max_x and max_y are exclusive

        disallowed_locations: list of locations that are not allowed to be placed
        (e.g. agent and item location should not be initialized to the same place)
        """
        # The start, item, goal location must be different position
        location = None
        while location is None or location in disallowed_locations:
            location = (randint(0, max_x - 1), randint(0, max_y - 1))

        self.location = location
        return location
    
    def get_location(self) -> tuple[int, int]:
        if self.location is None:
            raise ValueError("Location is not set")
        return self.location


class AgentObject(GridObject):
    def __init__(self, location: tuple[int, int] | None = None) -> None:
        super().__init__(location)
        self.icon = "A"
        self.has_item = False  # TODO: has_item of AgentObject and State must be synched somehow

    def move(self, action: Action) -> None:
        # NOTE: assumes that action is valid (i.e. agent is not at the edge of the grid)
        if self.location is None:
            raise ValueError("Agent location is not set")
        
        x, y = self.location
        if action == Action.LEFT:
            self.location = (x - 1, y)  # left
        elif action == Action.RIGHT:
            self.location = (x + 1, y)  # right
        elif action == Action.DOWN:
            self.location = (x, y - 1)  # down
        elif action == Action.UP:
            self.location = (x, y + 1)  # up
        elif action == Action.GOT_ITEM_LEFT:
            self.has_item = True
            self.location = (x - 1, y)
        elif action == Action.GOT_ITEM_RIGHT:
            self.has_item = True
            self.location = (x + 1, y)
        elif action == Action.GOT_ITEM_DOWN:
            self.has_item = True
            self.location = (x, y - 1)
        elif action == Action.GOT_ITEM_UP:
            self.has_item = True
            self.location = (x, y + 1)
        else:
            raise ValueError(f"Invalid action: {action}")


class ItemObject(GridObject):
    def __init__(self, location: tuple[int, int] | None = None):
        super().__init__(location)
        self.icon = "I"
