"""
Utility functions for bounding box operations.
"""


def get_center_of_bbox(bbox: list) -> tuple:
    """Get the center of the bounding box."""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def measure_distance(point1: tuple, point2: tuple) -> float:
    """Measure the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def get_foot_position(bbox: list) -> tuple:
    """Get the position of the foot from a player's bounding box."""
    x1, _, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(
    position,
    original_court_keypoints,
    keypoint_indices: list,
):
    """ "Get the index of the closest keypoint to the foot position."""
    closest_distance = float("inf")
    keypoint_index = keypoint_indices[0]

    for point in keypoint_indices:
        keypoint_tuple = (
            original_court_keypoints[point * 2],
            original_court_keypoints[point * 2 + 1],
        )
        distance = abs(position[1] - keypoint_tuple[1])

        if distance < closest_distance:
            closest_distance = distance
            keypoint_index = point
    return keypoint_index


def get_height_of_bbox(bbox: list) -> float:
    """Get the height of the bounding box."""
    return bbox[3] - bbox[1]


def measure_xy_distance(point1: tuple, point2: tuple) -> tuple:
    """Measure the Euclidean distance between two points."""
    return abs(point1[0] - point2[0]), abs(point1[1] - point2[1])