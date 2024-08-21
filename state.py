from dataclasses import dataclass


@dataclass
class State:
    agent_location: tuple[int, int]
    item_location: tuple[int, int]