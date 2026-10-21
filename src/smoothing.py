"""Small stateful smoothing utilities."""

from __future__ import annotations

from collections import deque


class MovingAverage:
    """Compute a trailing arithmetic mean over a fixed number of samples."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self._values: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """Add a sample and return the current moving average."""
        self._values.append(float(value))
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        """Discard all samples."""
        self._values.clear()
