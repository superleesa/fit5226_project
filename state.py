from dataclasses import dataclass


@dataclass
class State:
    # it doesn not hold AgentObject / ItemObject because I want State to be immutable
    # but in the future, we might want to add more attributes to State
    # in that case we need to make a copy of the AgentObject / ItemObject
    agent_location: tuple[int, int]
    item_location: tuple[int, int]