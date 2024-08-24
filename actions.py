from enum import Enum


class Action(Enum):
    # NOTE: QValue matrix used these action values as their indices
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3
    
    # actions when agent just got the item and is moving to item_reached state
    GOT_ITEM_LEFT = 4
    GOT_ITEM_RIGHT = 5
    GOT_ITEM_DOWN = 6
    GOT_ITEM_UP = 7
