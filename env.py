from abc import ABC
from random import randint

import numpy as np

from state import State


class Environment:
    def __init__(self, n=5, item = None):
        self.n = n
        
        self.item = ItemObject() if item is None else item
        self.agent = AgentObject()
        
        if self.item.location is None:
            self.item.set_location_randomly(self.n, self.n)
        
        # TODO: possibly implmeent this if there are multiple GridObjects to check for
        # initialize grid and put grid objects on the grid
        # x_agent, y_agent = self.agent.location
        # x_item, y_item = self.item.location
        # self.grid = np.zeros((self.n, self.n))
        # self.grid[x_agent, y_agent] = self.agent
        # self.grid[x_item, y_item] = self.item
    
    def initialize_for_new_episode(self):
        self.agent.set_location_randomly(self.n, self.n, [self.item.location])

    def get_state(self):
        return State(self.agent.location, self.item.location)

    def get_available_actions(self) -> list[int]:
        """
        Assumes that the current state is not the goal state
        """
        # logic to determine available actions
        actions = []
        x, y = self.agent.location
        if self.is_goal_state():
            self.is_goal = True
        else:    
        if x > 0:
            actions.append(0)  # left
        if x < self.n - 1:
            actions.append(1)  # right
        if y > 0:
            actions.append(2)  # down
        if y < self.n - 1:
            actions.append(3)  # up
        return actions
    
    def get_reward(self, state: State):
        # TODO: technically, speaking i think it should accept (prev state, action, next state)
        
        DEFAULT_TIME_PENALTY = -1  # TODO: parametrize this
        reward = DEFAULT_TIME_PENALTY
        if self.is_goal_state(state):
            reward = 10
        return reward
    
    def get_next_state(self, action: int) -> State:
        self.agent.move(action)
        return State(self.agent.location, self.item.location)
    
    def is_goal_state(self, state: State):
        return self.item.location == state.agent_location  # we treat the item location as the goal location

    def animate(self):
        pass

    def step(self):
        self.animate()


class GridObject(ABC):
    def __init__(self, location: tuple[int, int] | None = None) -> None:
        self.icon = None
        self.location = location
    
    def set_location_randomly(self, max_x: int, max_y: int, disallowed_locations: list[tuple[int, int]] = []) -> tuple[int, int]:
        """
        Note: max_x and max_y are exclusive
        
        disallowed_locations: list of locations that are not allowed to be placed
        (e.g. agent and item location should not be initialized to the same place)
        """
        if self.location is not None:
            return self.location
        
        # The start, item, goal location must be different position
        location = None
        
        while location is None or location in disallowed_locations:
            location = (randint(0, max_x-1), randint(0, max_y-1))
        
        self.location = location
        return location


class AgentObject(GridObject):
    def __init__(self, location: tuple[int, int] | None = None) -> None:
        super().__init__(location)
        self.icon = 'A'
    
    def move(self, action: int) -> None:
        # NOTE: assumes that action is valid (i.e. agent is not at the edge of the grid)
        if self.location is None:
            raise ValueError("Agent location is not set")
        x, y = self.location
        if action == 0:
            self.location = (x-1, y)  # left
        elif action == 1:
            self.location = (x+1, y)  # right
        elif action == 2:
            self.location = (x, y-1)  # down
        elif action == 3:
            self.location = (x, y+1)  # up
        else:
            raise ValueError(f"Invalid action: {action}")
        


class ItemObject(GridObject):
    def __init__(self, location: tuple[int, int] | None = None):
        super().__init__(location)
        self.icon = 'I'