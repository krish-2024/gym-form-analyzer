"""Rule-based biceps curl repetition and form analysis."""

from __future__ import annotations

from typing import Sequence

from src.angles import angle_between, vertical_angle
from src.exercises.base import Exercise, FrameResult, LandmarkPoint
from src.exercises.common import (
    PartialRangeTracker,
    RepetitionStateMachine,
    SideSelector,
)
from src.smoothing import MovingAverage

LEFT = {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23}
RIGHT = {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24}


class Curl(Exercise):
    """Count complete curls and provide side-view swing and elbow-position cues."""

    def __init__(self, config: dict[str, object]) -> None:
        self._visibility_threshold = float(config["landmark_visibility_threshold"])
        self._max_torso_swing = float(config["max_torso_swing_degrees"])
        self._max_elbow_drift_ratio = float(config["max_elbow_drift_ratio"])
        self._partial_angle = float(config["partial_range_angle"])
        smoothing_window = int(config["smoothing_window_size"])
        self._elbow_smoother = MovingAverage(smoothing_window)
        self._torso_smoother = MovingAverage(smoothing_window)
        self._side_selector = SideSelector(
            {
                "left": tuple(LEFT.values()),
                "right": tuple(RIGHT.values()),
            },
            int(config["side_visibility_window"]),
            self._visibility_threshold,
        )
        self._repetitions = RepetitionStateMachine(
            initial_phase="extended",
            lower_phase="curled",
            lower_angle=float(config["elbow_curled_angle"]),
            upper_angle=float(config["elbow_extended_angle"]),
            min_transition_seconds=float(config["min_transition_seconds"]),
        )
        self._rep_min_torso: float | None = None
        self._rep_max_torso: float | None = None
        self._minimum_elbow_angle: float | None = None
        self._partial_range = PartialRangeTracker(
            "extended", float(config["elbow_curled_angle"])
        )

    def update(
        self, landmarks: Sequence[LandmarkPoint] | None, timestamp: float
    ) -> FrameResult:
        """Update curl phase, count, and coaching feedback from one pose."""
        if landmarks is None:
            return self._missing_result()
        side = self._side_selector.update(landmarks)
        joints = LEFT if side == "left" else RIGHT
        if not self._has_required_landmarks(landmarks, joints):
            return self._missing_result()

        points = {name: landmarks[index] for name, index in joints.items()}
        try:
            elbow_angle = angle_between(
                points["shoulder"], points["elbow"], points["wrist"]
            )
            elbow_angle = self._elbow_smoother.update(elbow_angle)
            torso_angle = self._torso_smoother.update(
                vertical_angle(points["hip"], points["shoulder"])
            )
        except ValueError:
            return self._missing_result()
        torso_length = self._distance(points["hip"], points["shoulder"])
        if torso_length == 0.0:
            return self._missing_result()
        elbow_drift_ratio = (
            abs(points["elbow"].x - points["shoulder"].x) / torso_length
        )

        was_curled = self._repetitions.phase == "curled"
        partial_curl = self._partial_range.update(
            elbow_angle, self._repetitions.phase
        )
        if was_curled:
            self._rep_min_torso = self._minimum(self._rep_min_torso, torso_angle)
            self._rep_max_torso = self._maximum(self._rep_max_torso, torso_angle)
            self._minimum_elbow_angle = self._minimum(
                self._minimum_elbow_angle, elbow_angle
            )
        transition = self._repetitions.update(elbow_angle, timestamp)
        if transition.phase == "curled" and not was_curled:
            self._rep_min_torso = torso_angle
            self._rep_max_torso = torso_angle
            self._minimum_elbow_angle = elbow_angle

        torso_range = 0.0
        if self._rep_min_torso is not None and self._rep_max_torso is not None:
            torso_range = self._rep_max_torso - self._rep_min_torso
        feedback: list[str] = []
        if partial_curl:
            feedback.append("Curl through the full range")
        if was_curled and torso_range > self._max_torso_swing:
            feedback.append("Keep your torso still")
        if elbow_drift_ratio > self._max_elbow_drift_ratio:
            feedback.append("Keep your elbows close to your sides")
        if transition.rep_completed:
            if (
                self._minimum_elbow_angle is None
                or self._minimum_elbow_angle > self._partial_angle
            ):
                feedback.append("Curl through the full range")
            self._rep_min_torso = None
            self._rep_max_torso = None
            self._minimum_elbow_angle = None

        return FrameResult(
            phase=transition.phase,
            rep_count=self._repetitions.rep_count,
            angles={
                "elbow": elbow_angle,
                "torso_swing": torso_range,
                "elbow_drift_ratio": elbow_drift_ratio,
                "torso_lean": torso_angle,
            },
            feedback=feedback,
            rep_completed=transition.rep_completed,
        )

    def _has_required_landmarks(
        self, landmarks: Sequence[LandmarkPoint], joints: dict[str, int]
    ) -> bool:
        return all(
            index < len(landmarks)
            and landmarks[index].visibility >= self._visibility_threshold
            for index in joints.values()
        )

    @staticmethod
    def _distance(first: LandmarkPoint, second: LandmarkPoint) -> float:
        return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5

    @staticmethod
    def _minimum(current: float | None, value: float) -> float:
        return value if current is None else min(current, value)

    @staticmethod
    def _maximum(current: float | None, value: float) -> float:
        return value if current is None else max(current, value)

    def _missing_result(self) -> FrameResult:
        self._repetitions.reset_candidate()
        self._elbow_smoother.reset()
        self._torso_smoother.reset()
        self._partial_range.reset()
        return FrameResult(
            phase=self._repetitions.phase,
            rep_count=self._repetitions.rep_count,
            feedback=["Move into frame"],
        )
