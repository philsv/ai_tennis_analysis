"""
Utility functions for conversions.
"""

def convert_pixel_distance_to_meters(
    pixel_distance: float,
    reference_heigth_in_meters: float,
    reference_heigth_in_pixels: float,
) -> float:
    """Convert pixel distance to meters."""
    return (pixel_distance * reference_heigth_in_meters) / reference_heigth_in_pixels


def convert_meters_to_pixel_distance(
    meters: float,
    reference_heigth_in_meters: float,
    reference_heigth_in_pixels: float,
) -> float:
    """Convert meters to pixel distance."""
    return (meters * reference_heigth_in_pixels) / reference_heigth_in_meters
