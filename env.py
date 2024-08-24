from __future__ import annotations
from abc import ABC
from random import randint

import numpy as np
import matplotlib.pyplot as plt

from actions import Action
from state import State


DEFAULT_TIME_PENALTY = -1
GOAL_STATE_REWARD = 200
DEFAULT_ITEM_REWARD = 100


class Environment:
    def __init__(
        self,
        n: int = 5,
        item: ItemObject | None = None,
        goal_location: tuple[int, int] = (4, 4),
        time_penalty: int | float = DEFAULT_TIME_PENALTY,
        item_state_reward: int | float = DEFAULT_ITEM_REWARD,
        goal_state_reward: int | float = GOAL_STATE_REWARD,
    ) -> None:
        self.n = n
        self.goal_location = goal_location
        self.time_penalty = time_penalty
        self.item_state_reward = item_state_reward
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
        self.agent.set_location_randomly(self.n, self.n, [self.item.get_location()])
        self.state = State(
            agent_location=self.agent.get_location(),
            item_location=self.item.get_location(),
            has_item=self.agent.has_item,
        )
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
            actions.append(Action.COLLECT)
        
        # note: technically speaking we know that whenever agent is at the item location, the only available (or, the most optimal) action is to collect the item
        # however, according to the CE, we must ensure that 
        # "the agent is supposed to learn (rather than being told) that
        # once it has picked up the load it needs to move to the delivery point to complete its mission. ", 
        # implyging that agent must be able to learn to "collect" instead of being told to collect (so add all possible actions)
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
        if state.agent_location == state.item_location:
            return self.item_state_reward
        elif self.is_goal_state(state):
            return self.goal_state_reward
        else:
            return self.time_penalty

    def get_next_state(self, action: Action) -> State:
        self.agent.move(action)
        self.state = State(
            agent_location=self.agent.get_location(),
            item_location=self.item.get_location(),
            has_item=self.agent.has_item,
        )
        return self.state

    def is_goal_state(self, state: State) -> bool:
        return self.goal_location == state.agent_location

    def animate(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(self.n, 0)
        self.ax.set_xticks(np.arange(0, self.n + 1, 1))
        self.ax.set_yticks(np.arange(0, self.n + 1, 1))
        self.ax.grid(True)

        # Plotting the agent, item, and goal
        self.ax.text(
            self.agent.location[1] + 0.5,
            self.agent.location[0] + 0.5,
            "A",
            ha="center",
            va="center",
            fontsize=16,
            color="blue" if not self.agent.has_item else "purple",
        )
        self.ax.text(
            self.item.location[1] + 0.5,
            self.item.location[0] + 0.5,
            "I",
            ha="center",
            va="center",
            fontsize=16,
            color="green",
        )
        self.ax.text(
            self.goal_location[1] + 0.5,
            self.goal_location[0] + 0.5,
            "G",
            ha="center",
            va="center",
            fontsize=16,
            color="red",
        )

        # TODO: add a message saying "item collected" if the agent has collected the item
        # or else there is a single frame where the agent is at the same location twice,
        # so it looks like the agent is not moving
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="Agent (A)") if not self.agent.has_item else plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="purple", markersize=8, label="Agent (A) with item"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Item (I)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Goal (G)"),
        ]
        self.ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(right=0.75, left=0.1)
        self.fig.canvas.draw_idle()
        plt.pause(0.5)  # Pause to allow visualization of the movement

    def step(self, action: Action) -> tuple[float, State]:
        next_state = self.get_next_state(action)
        self.animate()
        reward = self.get_reward(next_state)
        return reward, next_state


# TODO: we might not need this actually
class InferenceEnvironment(Environment):
    """
    environment used during inference, that represent the environement of the actual problem world
    """

    def __init__(self, n: int = 5, item: ItemObject | None = None, goal_location: tuple[int, int] = (4, 4)) -> None:
        super().__init__(
            n,
            item,
            goal_location=goal_location,
        )  # note: during inference, we don't use rewards
        self.goal_location = (self.n - 1, self.n - 1)  # Set the goal state location to (n-1, n-1)

    def animate(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(self.n, 0)
        self.ax.set_xticks(np.arange(0, self.n + 1, 1))
        self.ax.set_yticks(np.arange(0, self.n + 1, 1))
        self.ax.grid(True)

        # Plotting the agent, item, and goal
        self.ax.text(
            self.agent.location[1] + 0.5,
            self.agent.location[0] + 0.5,
            "A",
            ha="center",
            va="center",
            fontsize=16,
            color="blue",
        )
        self.ax.text(
            self.item.location[1] + 0.5,
            self.item.location[0] + 0.5,
            "I",
            ha="center",
            va="center",
            fontsize=16,
            color="green",
        )
        self.ax.text(
            self.goal_location[1] + 0.5,
            self.goal_location[0] + 0.5,
            "G",
            ha="center",
            va="center",
            fontsize=16,
            color="red",
        )

        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="Agent (A)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Item (I)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Goal (G)"),
        ]
        self.ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(right=0.75, left=0.1)
        self.fig.canvas.draw_idle()
        plt.pause(0.5)  # Pause to allow visualization of the movement


class GridObject(ABC):
    def __init__(self, location: tuple[int, int] | None = None) -> None:
        self.icon: str
        self.location = (
            location  # NOTE: location is a tuple of (x, y) where x and y are coordinates on the grid (not indices)
        )

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
        elif action == Action.COLLECT:
            self.has_item = True


class ItemObject(GridObject):
    def __init__(self, location: tuple[int, int] | None = None):
        super().__init__(location)
        self.icon = "I"
