"""Tests for joint angle calculations."""

from types import SimpleNamespace

import pytest

from src.angles import angle_between, vertical_angle


def test_angle_between_returns_right_angle() -> None:
    first = SimpleNamespace(x=1.0, y=0.0)
    vertex = SimpleNamespace(x=0.0, y=0.0)
    last = SimpleNamespace(x=0.0, y=1.0)

    assert angle_between(first, vertex, last) == pytest.approx(90.0)


def test_vertical_angle_is_zero_for_vertical_line_in_either_direction() -> None:
    top = SimpleNamespace(x=1.0, y=0.0)
    bottom = SimpleNamespace(x=1.0, y=2.0)

    assert vertical_angle(top, bottom) == pytest.approx(0.0)
    assert vertical_angle(bottom, top) == pytest.approx(0.0)


def test_angle_between_rejects_coincident_points() -> None:
    point = SimpleNamespace(x=1.0, y=1.0)

    with pytest.raises(ValueError, match="coincident"):
        angle_between(point, point, point)
