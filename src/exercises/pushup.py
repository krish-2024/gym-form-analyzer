"""Rule-based push-up phase and side-view alignment analysis."""

from __future__ import annotations

from typing import Sequence

from src.angles import angle_between
from src.exercises.base import Exercise, FrameResult, LandmarkPoint
from src.exercises.common import (
    PartialRangeTracker,
    RepetitionStateMachine,
    SideSelector,
    body_line_metrics,
)
from src.smoothing import MovingAverage

LEFT = {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "ankle": 27}
RIGHT = {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "ankle": 28}


class Pushup(Exercise):
    """Count complete push-ups and report body-line and elbow-depth cues."""

    def __init__(self, config: dict[str, object]) -> None:
        self._visibility_threshold = float(config["landmark_visibility_threshold"])
        self._ideal_alignment = float(config["ideal_body_alignment_degrees"])
        self._posture_tolerance = float(config["posture_tolerance_degrees"])
        self._depth_angle = float(config["incomplete_bend_angle"])
        self._elbow_smoother = MovingAverage(int(config["smoothing_window_size"]))
        side_visibility = {
            "left": tuple(LEFT.values()),
            "right": tuple(RIGHT.values()),
        }
        self._side_selector = SideSelector(
            side_visibility,
            int(config["side_visibility_window"]),
            self._visibility_threshold,
        )
        self._repetitions = RepetitionStateMachine(
            initial_phase="up",
            lower_phase="down",
            lower_angle=float(config["elbow_down_angle"]),
            upper_angle=float(config["elbow_up_angle"]),
            min_transition_seconds=float(config["min_transition_seconds"]),
        )
        self._minimum_elbow_angle: float | None = None
        self._partial_range = PartialRangeTracker("up", self._depth_angle)

    def update(
        self, landmarks: Sequence[LandmarkPoint] | None, timestamp: float
    ) -> FrameResult:
        """Update push-up count and form feedback from one pose."""
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
            alignment_angle, hip_offset = body_line_metrics(
                points["shoulder"], points["hip"], points["ankle"]
            )
        except ValueError:
            return self._missing_result()

        was_down = self._repetitions.phase == "down"
        partial_bend = self._partial_range.update(
            elbow_angle, self._repetitions.phase
        )
        if was_down:
            self._minimum_elbow_angle = (
                elbow_angle
                if self._minimum_elbow_angle is None
                else min(self._minimum_elbow_angle, elbow_angle)
            )
        transition = self._repetitions.update(elbow_angle, timestamp)
        if transition.phase == "down" and not was_down:
            self._minimum_elbow_angle = elbow_angle
        feedback: list[str] = []
        if abs(self._ideal_alignment - alignment_angle) > self._posture_tolerance:
            feedback.append(
                "Lower your hips"
                if hip_offset < 0
                else "Keep your hips from sagging"
            )
        if partial_bend:
            feedback.append("Bend lower")
        if transition.rep_completed:
            if (
                self._minimum_elbow_angle is None
                or self._minimum_elbow_angle > self._depth_angle
            ):
                feedback.append("Bend lower")
            self._minimum_elbow_angle = None

        return FrameResult(
            phase=transition.phase,
            rep_count=self._repetitions.rep_count,
            angles={
                "elbow": elbow_angle,
                "body_alignment": alignment_angle,
                "hip_offset_ratio": hip_offset,
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

    def _missing_result(self) -> FrameResult:
        self._repetitions.reset_candidate()
        self._elbow_smoother.reset()
        self._partial_range.reset()
        return FrameResult(
            phase=self._repetitions.phase,
            rep_count=self._repetitions.rep_count,
            feedback=["Move into frame"],
        )
