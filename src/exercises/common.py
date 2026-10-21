"""Shared camera-independent helpers for exercise state machines."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.angles import angle_between
from src.exercises.base import LandmarkPoint


class SideSelector:
    """Select the anatomical side with the highest rolling visibility."""

    def __init__(
        self,
        landmark_indices: Mapping[str, Sequence[int]],
        window_size: int,
        visibility_threshold: float,
    ) -> None:
        if not landmark_indices:
            raise ValueError("At least one side must be configured.")
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self._indices = dict(landmark_indices)
        self._visibility_threshold = visibility_threshold
        self._history = {
            side: deque(maxlen=window_size) for side in self._indices
        }
        self._selected_side = next(iter(self._indices))

    def update(self, landmarks: Sequence[LandmarkPoint]) -> str:
        """Add a pose sample and return the best rolling-visibility side."""
        for side, indices in self._indices.items():
            visible = [
                landmarks[index].visibility
                for index in indices
                if index < len(landmarks)
            ]
            self._history[side].append(
                sum(visible) / len(visible) if visible else 0.0
            )
        scores = {
            side: sum(values) / len(values) if values else 0.0
            for side, values in self._history.items()
        }
        highest_score = max(scores.values())
        if highest_score >= self._visibility_threshold:
            self._selected_side = max(scores, key=scores.__getitem__)
        return self._selected_side


@dataclass(frozen=True)
class RepetitionUpdate:
    """The phase and completion event from a repetition state update."""

    phase: str
    rep_completed: bool


class RepetitionStateMachine:
    """Count complete low-angle/high-angle cycles with minimum dwell times."""

    def __init__(
        self,
        initial_phase: str,
        lower_phase: str,
        lower_angle: float,
        upper_angle: float,
        min_transition_seconds: float,
    ) -> None:
        self._phase = initial_phase
        self._initial_phase = initial_phase
        self._lower_phase = lower_phase
        self._lower_angle = lower_angle
        self._upper_angle = upper_angle
        self._min_transition_seconds = min_transition_seconds
        self._candidate_phase: str | None = None
        self._candidate_since: float | None = None
        self.rep_count = 0

    @property
    def phase(self) -> str:
        """Return the committed movement phase."""
        return self._phase

    def reset_candidate(self) -> None:
        """Cancel an incomplete transition after a missing pose observation."""
        self._candidate_phase = None
        self._candidate_since = None

    def update(self, angle: float, timestamp: float) -> RepetitionUpdate:
        """Advance state only after a threshold persists for the configured time."""
        if self._phase == self._initial_phase and angle < self._lower_angle:
            target_phase = self._lower_phase
        elif self._phase == self._lower_phase and angle > self._upper_angle:
            target_phase = self._initial_phase
        else:
            self.reset_candidate()
            return RepetitionUpdate(self._phase, False)

        if target_phase != self._candidate_phase:
            self._candidate_phase = target_phase
            self._candidate_since = timestamp
            return RepetitionUpdate(self._phase, False)

        if (
            self._candidate_since is None
            or timestamp - self._candidate_since < self._min_transition_seconds
        ):
            return RepetitionUpdate(self._phase, False)

        self._phase = target_phase
        self.reset_candidate()
        rep_completed = self._phase == self._initial_phase
        if rep_completed:
            self.rep_count += 1
        return RepetitionUpdate(self._phase, rep_completed)


class PartialRangeTracker:
    """Flag a reversal that fails to reach the configured lower-angle target."""

    def __init__(self, initial_phase: str, lower_angle: float) -> None:
        self._initial_phase = initial_phase
        self._lower_angle = lower_angle
        self._previous_angle: float | None = None
        self._minimum_angle: float | None = None

    def update(self, angle: float, phase: str) -> bool:
        """Return true when movement reverses before crossing its range target."""
        if phase != self._initial_phase:
            self.reset()
            return False
        partial_range = (
            self._previous_angle is not None
            and angle > self._previous_angle
            and self._minimum_angle is not None
            and self._minimum_angle >= self._lower_angle
        )
        if partial_range:
            self._minimum_angle = None
        elif self._previous_angle is None or angle < self._previous_angle:
            self._minimum_angle = (
                angle
                if self._minimum_angle is None
                else min(self._minimum_angle, angle)
            )
        self._previous_angle = angle
        return partial_range

    def reset(self) -> None:
        """Clear a partially observed attempt."""
        self._previous_angle = None
        self._minimum_angle = None


def body_line_metrics(
    shoulder: LandmarkPoint,
    hip: LandmarkPoint,
    ankle: LandmarkPoint,
) -> tuple[float, float]:
    """Return shoulder-hip-ankle angle and signed hip offset/body-length ratio."""
    alignment_angle = angle_between(shoulder, hip, ankle)
    body_length = math.hypot(ankle.x - shoulder.x, ankle.y - shoulder.y)
    if body_length == 0.0:
        raise ValueError("Cannot assess alignment with a zero-length body line.")
    horizontal_span = ankle.x - shoulder.x
    if horizontal_span == 0.0:
        expected_hip_y = (shoulder.y + ankle.y) / 2
    else:
        fraction = (hip.x - shoulder.x) / horizontal_span
        expected_hip_y = shoulder.y + fraction * (ankle.y - shoulder.y)
    signed_offset_ratio = (hip.y - expected_hip_y) / body_length
    return alignment_angle, signed_offset_ratio
