def calculate_manhattan_distance(start_location: tuple[int, int], goal_location: tuple[int, int]) -> int:
    """
    Calculates the Manhattan distance between two points.
    """
    start_x, start_y = start_location
    goal_x, goal_y = goal_location
    return abs(start_x - goal_x) + abs(start_y - goal_y)


def calculate_metrics_score(
    predicted_distance: int,
    start_location: tuple[int, int],
    item_location: tuple[int, int],
    goal_location: tuple[int, int],
) -> float:
    """
    Calculates the proportion of the distance to the shortest distance.
    NOTE: if predicted_distance is 0, the function will return 0.
    """
    # Calculate shortest distance from start to item to goal
    shortest_distance = (
        calculate_manhattan_distance(start_location, item_location)
        + 1
        + calculate_manhattan_distance(item_location, goal_location)
    )
    return (shortest_distance / predicted_distance) if predicted_distance != 0 else 0
