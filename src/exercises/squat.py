"""Rule-based side-view squat phase and form analysis."""

from __future__ import annotations

from collections import deque
from typing import Sequence

from src.angles import angle_between, vertical_angle
from src.exercises.base import Exercise, FrameResult, LandmarkPoint
from src.smoothing import MovingAverage

LEFT = {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27, "heel": 29}
RIGHT = {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28, "heel": 30}
REQUIRED_JOINTS = ("shoulder", "hip", "knee", "ankle", "heel")


class Squat(Exercise):
    """Detect full squat cycles and provide simple form cues."""

    def __init__(self, config: dict[str, object]) -> None:
        self._visibility_threshold = float(config["landmark_visibility_threshold"])
        self._down_angle = float(config["knee_down_angle"])
        self._up_angle = float(config["knee_up_angle"])
        self._min_transition = float(config["min_transition_seconds"])
        self._depth_angle = float(config["depth_knee_angle"])
        self._max_trunk_lean = float(config["max_trunk_lean_degrees"])
        self._max_heel_lift_ratio = float(config["max_heel_lift_ratio"])
        self._smoother = MovingAverage(int(config["smoothing_window_size"]))
        side_window = int(config["side_visibility_window"])
        self._visibility_history: dict[str, deque[float]] = {
            "left": deque(maxlen=side_window),
            "right": deque(maxlen=side_window),
        }
        self._phase = "up"
        self._candidate_phase: str | None = None
        self._candidate_since: float | None = None
        self._rep_count = 0
        self._min_rep_knee_angle: float | None = None
        self._selected_side: str | None = None

    def update(
        self, landmarks: Sequence[LandmarkPoint] | None, timestamp: float
    ) -> FrameResult:
        """Update the squat state from one MediaPipe landmark observation."""
        if landmarks is None:
            return self._missing_result()

        self._record_side_visibility(landmarks)
        side = self._choose_side()
        side_joints = LEFT if side == "left" else RIGHT
        if not self._has_required_landmarks(landmarks, side_joints):
            return self._missing_result()

        points = {name: landmarks[index] for name, index in side_joints.items()}
        knee_angle = self._smoother.update(
            angle_between(points["hip"], points["knee"], points["ankle"])
        )
        hip_angle = angle_between(
            points["shoulder"], points["hip"], points["knee"]
        )
        trunk_lean = vertical_angle(points["hip"], points["shoulder"])
        leg_length = self._distance(points["hip"], points["ankle"])
        if leg_length == 0.0:
            return self._missing_result()
        heel_lift_ratio = max(
            0.0,
            (points["ankle"].y - points["heel"].y) / leg_length,
        )
        angles = {
            "knee": knee_angle,
            "hip": hip_angle,
            "trunk_lean": trunk_lean,
            "heel_lift_ratio": heel_lift_ratio,
        }

        rep_completed, rep_was_shallow = self._advance_phase(knee_angle, timestamp)
        if self._phase == "down":
            self._min_rep_knee_angle = (
                knee_angle
                if self._min_rep_knee_angle is None
                else min(self._min_rep_knee_angle, knee_angle)
            )

        feedback = self._form_feedback(trunk_lean, heel_lift_ratio)
        if rep_was_shallow:
            feedback.insert(0, "Go lower")

        return FrameResult(
            phase=self._phase,
            rep_count=self._rep_count,
            angles=angles,
            feedback=feedback,
            rep_completed=rep_completed,
        )

    def _record_side_visibility(self, landmarks: Sequence[LandmarkPoint]) -> None:
        for side, joints in (("left", LEFT), ("right", RIGHT)):
            visibility = [
                landmarks[joints[name]].visibility
                for name in REQUIRED_JOINTS
                if joints[name] < len(landmarks)
            ]
            self._visibility_history[side].append(
                sum(visibility) / len(visibility) if visibility else 0.0
            )

    def _choose_side(self) -> str:
        if self._selected_side is None:
            self._selected_side = "left"
        left_score = self._mean(self._visibility_history["left"])
        right_score = self._mean(self._visibility_history["right"])
        if max(left_score, right_score) >= self._visibility_threshold:
            self._selected_side = "left" if left_score >= right_score else "right"
        return self._selected_side

    def _has_required_landmarks(
        self, landmarks: Sequence[LandmarkPoint], joints: dict[str, int]
    ) -> bool:
        return all(
            index < len(landmarks)
            and landmarks[index].visibility >= self._visibility_threshold
            for index in joints.values()
        )

    def _advance_phase(self, knee_angle: float, timestamp: float) -> tuple[bool, bool]:
        target_phase: str | None = None
        if self._phase == "up" and knee_angle < self._down_angle:
            target_phase = "down"
        elif self._phase == "down" and knee_angle > self._up_angle:
            target_phase = "up"

        if target_phase is None:
            self._candidate_phase = None
            self._candidate_since = None
            return False, False

        if target_phase != self._candidate_phase:
            self._candidate_phase = target_phase
            self._candidate_since = timestamp
            return False, False

        if (
            self._candidate_since is None
            or timestamp - self._candidate_since < self._min_transition
        ):
            return False, False

        self._phase = target_phase
        self._candidate_phase = None
        self._candidate_since = None
        if target_phase == "down":
            self._min_rep_knee_angle = knee_angle
            return False, False

        self._rep_count += 1
        completed_depth = self._min_rep_knee_angle
        self._min_rep_knee_angle = None
        return True, completed_depth is None or completed_depth > self._depth_angle

    def _form_feedback(
        self, trunk_lean: float, heel_lift_ratio: float
    ) -> list[str]:
        feedback: list[str] = []
        if trunk_lean > self._max_trunk_lean:
            feedback.append("Keep your chest up")
        if heel_lift_ratio > self._max_heel_lift_ratio:
            feedback.append("Keep heels down")
        # TODO: Knee cave requires a front-view camera angle.
        return feedback

    @staticmethod
    def _distance(first: LandmarkPoint, second: LandmarkPoint) -> float:
        return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5

    @staticmethod
    def _mean(values: deque[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _missing_result(self) -> FrameResult:
        self._candidate_phase = None
        self._candidate_since = None
        self._smoother.reset()
        return FrameResult(
            phase=self._phase,
            rep_count=self._rep_count,
            feedback=["Move into frame"],
        )
