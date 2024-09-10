from dataclasses import dataclass


@dataclass
class State:
    # it doesn not hold AgentObject / ItemObject because I want State to be immutable
    # but in the future, we might want to add more attributes to State
    # in that case we need to make a copy of the AgentObject / ItemObject
    agent_location: tuple[int, int]
    item_location: tuple[int, int]
    
    has_item: bool = False


@dataclass
class Assignment2State(State):
    goal_location: tuple[int, int]
    
    # https://edstem.org/au/courses/17085/discussion/2192014
    # these two attributes should be unit vectors
    goal_direction: tuple[float, float]
    item_direction: tuple[int, int]