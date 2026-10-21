"""Common exercise result and landmark types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Sequence


class LandmarkPoint(Protocol):
    """The MediaPipe landmark fields used by exercise analyzers."""

    x: float
    y: float
    z: float
    visibility: float


@dataclass
class FrameResult:
    """Exercise state and feedback for one processed frame."""

    phase: str
    rep_count: int
    angles: dict[str, float] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)
    rep_completed: bool = False
    rep_score: float | None = None


class Exercise(ABC):
    """Base interface for camera-independent exercise analyzers."""

    @abstractmethod
    def update(
        self, landmarks: Sequence[LandmarkPoint] | None, timestamp: float
    ) -> FrameResult:
        """Consume one pose observation and return the updated exercise state."""
