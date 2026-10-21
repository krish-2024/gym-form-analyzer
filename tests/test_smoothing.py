"""Tests for moving-average smoothing."""

import pytest

from src.smoothing import MovingAverage


def test_moving_average_uses_only_latest_window() -> None:
    smoother = MovingAverage(window_size=3)
    assert smoother.update(10) == pytest.approx(10)
    assert smoother.update(20) == pytest.approx(15)
    assert smoother.update(30) == pytest.approx(20)
    assert smoother.update(40) == pytest.approx(30)


def test_moving_average_reset_clears_samples() -> None:
    smoother = MovingAverage(window_size=2)
    smoother.update(10)
    smoother.reset()

    assert smoother.update(6) == pytest.approx(6)


def test_moving_average_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        MovingAverage(window_size=0)
