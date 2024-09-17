from __future__ import annotations
from abc import ABC
from random import randint, choice
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from fit5226_project.actions import Action
from fit5226_project.state import State, Assignment2State


DEFAULT_TIME_PENALTY = -1
GOAL_STATE_REWARD = 300
DEFAULT_ITEM_REWARD = 200


class Environment:
    def __init__(
        self,
        n: int = 5,
        item: ItemObject | None = None,
        goal_location: tuple[int, int] = (4, 0),
        time_penalty: int | float = DEFAULT_TIME_PENALTY,
        item_state_reward: int | float = DEFAULT_ITEM_REWARD,
        goal_state_reward: int | float = GOAL_STATE_REWARD,
        with_animation: bool = True,
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
        self.with_animation = with_animation

    def initialize_for_new_episode(self, agent_location: tuple[int, int] | None = None) -> None:
        if agent_location is None:
            self.agent.set_location_randomly(self.n, self.n, [self.item.get_location()]) 
        else:
            self.agent.location = agent_location
        self.agent.has_item = False if randint(0, 1) == 0 else True
        self.state = State(
            agent_location=self.agent.get_location(),
            item_location=self.item.get_location(),
            has_item=self.agent.has_item,
        )
        
        # ensure that no multiple matplotlib windows open
        if hasattr(self, "fig"):
            plt.close(self.fig)  # type: ignore
        self.fig, self.ax = plt.subplots(figsize=(8, 8)) if self.with_animation else (None, None)
        self.animate()  # Initial drawing of the grid

        # Reset the last action and reward
        self.last_action = None
        self.last_reward = None


    def get_state(self) -> State:
        return self.state
    
    def set_with_animation(self, with_animation: bool) -> None:
        self.with_animation = with_animation

    def get_available_actions(self, state: State | None = None) -> list[Action]:
        """
        Assumes that the current state is not the goal state
        """
        # logic to determine available actions
        actions = []
        current_state = state if state is not None else self.state
        x, y = current_state.agent_location

        if current_state.agent_location == current_state.item_location and not current_state.has_item:
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

    def get_reward(self, prev_state: State, current_state: State, action: Action) -> float:
        """
        Calculate the reward based on the agent's actions and state transitions.
        """

        # Large penalty for reaching the goal without the item
        if current_state.agent_location == self.goal_location and not current_state.has_item:
            return -self.item_state_reward //2 # Large penalty for going to goal without item

        # Large reward for reaching the goal with the item
        if self.is_goal_state(current_state):
            return self.goal_state_reward *2 # High reward for successfully reaching the goal with item
        
        # #  # Large penalty for reaching the goal without the item
        # if prev_state.agent_location == current_state.item_location and current_state.has_item and prev_state.has_item :
        #     return -50

        # Reward for collecting the item
        if action == Action.COLLECT and prev_state.agent_location == current_state.item_location and not prev_state.has_item:
            return self.item_state_reward  # Reward for collecting the item

        # Penalize if attempting to collect when not at item location or already has item
        if action == Action.COLLECT and (prev_state.has_item or prev_state.agent_location != current_state.item_location):
            return -50  # Penalty for attempting to collect when not at item or already collected

        # Calculate distance-based reward or penalty
        reward = self.time_penalty  # Default time penalty


        return reward




    def update_state(self, action: Action) -> None:
        """
        Be careful: this method updates the state of the environment
        """
        self.agent.move(action,self.n)
        self.state = State(
            agent_location=self.agent.get_location(),
            item_location=self.item.get_location(),
            has_item=self.agent.has_item,
        )

    def is_goal_state(self, state: State) -> bool:
        return self.state.has_item and self.goal_location == state.agent_location

    def animate(self, state: Assignment2State | None = None, prev_state: Assignment2State | None = None, is_greedy: bool | None = None, all_qvals: np.ndarray | None = None, chosen_action: Action | None = None) -> None:
        """
        Animates the action
        (basically just prints out the new state, but because it seems like the agent is "moving" because it's updated in the same figure)
        """
        if not self.with_animation:
            return
        self.ax.clear()
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(0, self.n)
        self.ax.set_xticks(np.arange(0, self.n + 1, 1))
        self.ax.set_yticks(np.arange(0, self.n + 1, 1))
        self.ax.grid(True)

        # Plotting the agent, item, and goal
        self.ax.text(
            self.agent.location[0] + 0.5,
            self.agent.location[1] + 0.5,
            "A",
            ha="center",
            va="center",
            fontsize=16,
            color="blue" if not self.agent.has_item else "purple",
        )
        self.ax.text(
            self.item.location[0] + 0.5,
            self.item.location[1] + 0.5,
            "I",
            ha="center",
            va="center",
            fontsize=16,
            color="green",
        )
        self.ax.text(
            self.goal_location[0] + 0.5,
            self.goal_location[1] + 0.5,
            "G",
            ha="center",
            va="center",
            fontsize=16,
            color="red",
        )
        
        # FIXME: this doesn't work for State (only works for Assignment2State)
        state_str = str(self.state) if state is None else str(state)
        state_text = "".join(state_str.split(',')[:5]) + '\n' + "".join(state_str.split(',')[5:])
        
        # show state info
        self.ax.text(
            2,
            2,
            state_text,
            ha="center",
            va="center",
            fontsize=10,
            color="orange",
        )
        
        # prints: if the action selected was greedy or random
        if is_greedy is not None:
            self.ax.text(
            self.n,
            self.n,
            "Action is greedy" if is_greedy else "Action is random",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )
        
        # prints the q values for all possible actions in the previous state
        # note: this is only printed if the action was greedy (because if random, the q values did not matter for action selection)
        # note2: only "possible" actions are printed i.e. (if agent is not at the item position, it does not print the collect q value)
        if all_qvals is not None and prev_state is not None and is_greedy:
            left_q, right_q, down_q, up_q, collect_q = all_qvals
            possible_actions = self.get_available_actions(prev_state)
            # show left q value
            prev_agent_location_on_plot_x = prev_state.agent_location[0] + 0.5
            prev_agent_location_on_plot_y = prev_state.agent_location[1] + 0.5
            box_center_to_val_location = 0.3
            if Action.LEFT in possible_actions:
                self.ax.text(
                    prev_agent_location_on_plot_x - box_center_to_val_location,
                    prev_agent_location_on_plot_y,
                    f'{left_q:.2f}',
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="red" if chosen_action == Action.LEFT else "black",
                )
            if Action.RIGHT in possible_actions:    
                self.ax.text(
                    prev_agent_location_on_plot_x + box_center_to_val_location,
                    prev_agent_location_on_plot_y,
                    f'{right_q:.2f}',
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="red" if chosen_action == Action.RIGHT else "black",
                )
            if Action.DOWN in possible_actions:
                self.ax.text(
                    prev_agent_location_on_plot_x,
                    prev_agent_location_on_plot_y - box_center_to_val_location,
                    f'{down_q:.2f}',
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="red" if chosen_action == Action.DOWN else "black",
                )
            if Action.UP in possible_actions:
                self.ax.text(
                    prev_agent_location_on_plot_x,
                    prev_agent_location_on_plot_y + box_center_to_val_location,
                    f'{up_q:.2f}',
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="red" if chosen_action == Action.UP else "black",
                )
            if Action.COLLECT in possible_actions:
                self.ax.text(
                    prev_agent_location_on_plot_x,
                    prev_agent_location_on_plot_y,
                    f'{collect_q:.2f}',
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="red" if chosen_action == Action.COLLECT else "black",
                )
            
                

        # TODO: add a message saying "item collected" if the agent has collected the item
        # or else there is a single frame where the agent is at the same location twice,
        # so it looks like the agent is not moving
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="Agent (A)")
            if not self.agent.has_item
            else plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="purple", markersize=8, label="Agent (A) with item"
            ),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Item (I)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Goal (G)"),
        ]
        self.ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(right=0.75, left=0.1)
        self.fig.canvas.draw_idle()
        plt.pause(0.7)  # Pause to allow visualization of the movement

    def step(self, action: Action) -> tuple[float, State]:
        prev_state = self.get_state()
        self.update_state(action)
        next_state = self.get_state()
        self.animate()
        reward = self.get_reward(prev_state, next_state,action)
        return reward, next_state


class Assignment2Environment:
    """
    A wrapper class for multiple environments for Assignment 2
    This environment consits of multiple "sub-environments" where each sub-environment has a different goal and item location
    """
    def __init__(
        self, 
        n: int = 5,
        time_penalty: int | float = DEFAULT_TIME_PENALTY,
        item_state_reward: int | float = DEFAULT_ITEM_REWARD,
        goal_state_reward: int | float = GOAL_STATE_REWARD,
        direction_reward_multiplier: int | float = 10,
        with_animation: bool = True,
    ) -> None:
        self.n = n
        # initialize a list of environments for all possible goal and item positions
        self.environments = []
        
        for goal_x in range(self.n):
            for goal_y in range(self.n):
                for item_x in range(self.n):
                    for item_y in range(self.n):
                        if (goal_x, goal_y) == (item_x, item_y):
                            continue
                        environment = Environment(
                            n=self.n,
                            goal_location=(goal_x, goal_y),
                            item=ItemObject(location=(item_x, item_y)),
                            with_animation=with_animation,
                            time_penalty=time_penalty,
                            item_state_reward=item_state_reward,
                            goal_state_reward=goal_state_reward,
                        )
                        self.environments.append(environment)
        
        self.environments = [self.environments[10]]
                
        
        self.direction_reward_multiplier = direction_reward_multiplier
        
        self.current_sub_environment: Environment
        self.state: Assignment2State
    
    def get_random_sub_environment(self) -> Environment:
        return choice(self.environments)
    
    def initialize_for_new_episode(self, agent_location: tuple[int, int] | None = None, index: int | None = None) -> None:
        self.current_sub_environment = self.get_random_sub_environment() if index is None else self.environments[index]
        self.current_sub_environment.initialize_for_new_episode(agent_location)
        
        self.state = Assignment2State(
            agent_location=self.current_sub_environment.agent.get_location(),
            item_location=self.current_sub_environment.item.get_location(),
            has_item=self.current_sub_environment.agent.has_item,
            goal_location=self.current_sub_environment.goal_location,
            goal_direction=self.get_goal_direction(),
            item_direction=self.get_item_direction(),
        )
        # NOTE: animation should be handled by individual sub-environments
    
    def get_available_actions(self, state:Assignment2State) -> list[Action]:
        return self.current_sub_environment.get_available_actions(state)

    def set_with_animation(self, with_animation: bool) -> None:
        for environment in self.environments:
            environment.set_with_animation(with_animation)
    
    def get_direction_reward(self, action: Action) -> float:
        """
        Use cosine similarity to calculate the reward based on the direction of the action.
        """
        has_collected_item = self.state.has_item

        # Define action direction vectors
        if action == Action.LEFT:
            action_direction = (-1, 0)
        elif action == Action.RIGHT:
            action_direction = (1, 0)
        elif action == Action.DOWN:
            action_direction = (0, -1)
        elif action == Action.UP:
            action_direction = (0, 1)
        else:
            action_direction = (0, 0)  # Invalid action, handle accordingly

        # Calculate the direction reward based on the goal or item direction
        if has_collected_item:
            target_direction = self.state.goal_direction
        else:
            target_direction = self.state.item_direction

        # Check if either vector is zero to avoid division by zero
        if np.linalg.norm(action_direction) == 0 or np.linalg.norm(target_direction) == 0:
            return 0.0  # No direction reward if either vector is zero

        # Calculate the cosine similarity (1 - cosine distance)
        try:
            reward = 1 - cosine(action_direction, target_direction)
        except ValueError:
            # Handle any errors from invalid vectors
            reward = 0.0

        return reward * self.direction_reward_multiplier

    
    # def get_reward(self, prev_state: Assignment2State, current_state: Assignment2State, action: Action) -> float:
    #     state_raward = self.current_sub_environment.get_reward(prev_state, current_state,action)
    #     action_reward = self.get_direction_reward(action)
    #     return state_raward + action_reward
    def get_reward(self, prev_state: Assignment2State, current_state: Assignment2State, action: Action) -> float:
        state_raward = self.current_sub_environment.get_reward(prev_state, current_state, action)
        return state_raward
    
    
    def get_state(self) -> Assignment2State:
        return self.state
    
    def get_goal_direction(self) -> tuple[float, float]:
        return (
            self.current_sub_environment.goal_location[0] - self.current_sub_environment.agent.get_location()[0],
            self.current_sub_environment.goal_location[1] - self.current_sub_environment.agent.get_location()[1],
        )
    
    def get_item_direction(self) -> tuple[float, float]:
        return (
            self.current_sub_environment.item.get_location()[0] - self.current_sub_environment.agent.get_location()[0],
            self.current_sub_environment.item.get_location()[1] - self.current_sub_environment.agent.get_location()[1],
        )
    
    def is_goal_state(self, state: State) -> bool:
        return self.current_sub_environment.state.has_item and self.current_sub_environment.goal_location == state.agent_location
    
    def update_state(self, action: Action) -> None:
        """
        Be careful: this method updates the state of the environment
        """
        self.current_sub_environment.update_state(action)
        self.state = Assignment2State(
            agent_location=self.current_sub_environment.agent.get_location(),
            item_location=self.current_sub_environment.item.get_location(),
            has_item=self.current_sub_environment.agent.has_item,
            goal_location=self.current_sub_environment.goal_location,
            goal_direction=self.get_goal_direction(),
            item_direction=self.get_item_direction(),
        )
    
    def step(self, action: Action, is_greedy: bool, all_qvals: np.ndarray) -> tuple[float, Assignment2State]:
        prev_state = self.get_state()
        self.update_state(action)
        next_state = self.get_state()
        self.current_sub_environment.animate(self.get_state(), prev_state, is_greedy, all_qvals, action)
        reward = self.get_reward(prev_state, next_state, action)
        return reward, next_state


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

    def move(self, action: Action, grid_size: int) -> None:
        """
        Move the agent based on the given action while ensuring it doesn't leave the bounds of the grid.
        """
        if self.location is None:
            raise ValueError("Agent location is not set")

        x, y = self.location

        # Check each action and ensure it stays within the bounds
        if action == Action.LEFT:
            if x > 0:  # Ensure not moving out of bounds on the left
                self.location = (x - 1, y)
        elif action == Action.RIGHT:
            if x < grid_size - 1:  # Ensure not moving out of bounds on the right
                self.location = (x + 1, y)
        elif action == Action.DOWN:
            if y > 0:  # Ensure not moving out of bounds downwards
                self.location = (x, y - 1)
        elif action == Action.UP:
            if y < grid_size - 1:  # Ensure not moving out of bounds upwards
                self.location = (x, y + 1)
        elif action == Action.COLLECT:
            self.has_item = True  # Action to collect the item (no bounds check needed)



class ItemObject(GridObject):
    def __init__(self, location: tuple[int, int] | None = None):
        super().__init__(location)
        self.icon = "I"
