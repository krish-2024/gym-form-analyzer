"""Two-dimensional joint angle utilities."""

from __future__ import annotations

import math
from typing import Protocol


class Point2D(Protocol):
    """A point with normalized or pixel-space x/y coordinates."""

    x: float
    y: float


def angle_between(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Return the angle ABC in degrees, in the inclusive range 0 to 180."""
    first = (a.x - b.x, a.y - b.y)
    second = (c.x - b.x, c.y - b.y)
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length == 0.0 or second_length == 0.0:
        raise ValueError("Cannot calculate an angle with coincident points.")
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_length * second_length
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def vertical_angle(p1: Point2D, p2: Point2D) -> float:
    """Return the acute angle, in degrees, between a segment and vertical."""
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    length = math.hypot(delta_x, delta_y)
    if length == 0.0:
        raise ValueError("Cannot calculate an angle for coincident points.")
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(delta_y) / length))))
