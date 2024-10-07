import datetime
from typing import Any

def generate_grid_location_list(max_x: int, max_y) -> list[tuple[int, int]]:
    """
    Generate the grid location list for all possible cases
    """
    return [(i, j) for i in range(max_x) for j in range(max_y)]


def generate_unique_id() -> str:
    # Get the current time and format it for uniqueness
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

def ensure_key_exists(config: dict[str, dict[str, Any] | None]) -> None:
    for required_key in ["agent", "env", "trainer"]:
        if required_key not in config:
            config[required_key] = {}
        elif required_key in config and config[required_key] is None:
            config[required_key] = {}