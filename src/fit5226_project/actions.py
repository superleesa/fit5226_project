from enum import Enum


class Action(Enum):
    # NOTE: QValue matrix used these action values as their indices
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3

    # actions when agent just got the item and is moving to item_reached state
    COLLECT = 4  # goes to item reached state
