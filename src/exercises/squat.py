"""Rule-based side-view squat phase and form analysis."""

from __future__ import annotations

from typing import Sequence

from src.angles import angle_between, vertical_angle
from src.exercises.base import Exercise, FrameResult, LandmarkPoint
from src.exercises.common import RepetitionStateMachine, SideSelector
from src.scoring import (
    RepScorer,
    ScoreSummary,
    SetScoreTracker,
    config_section,
    symmetry_score,
    tempo_score,
    threshold_score,
)
from src.smoothing import MovingAverage

LEFT = {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27, "heel": 29}
RIGHT = {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28, "heel": 30}


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
        self._side_selector = SideSelector(
            {
                "left": tuple(LEFT.values()),
                "right": tuple(RIGHT.values()),
            },
            int(config["side_visibility_window"]),
            self._visibility_threshold,
        )
        self._repetitions = RepetitionStateMachine(
            initial_phase="up",
            lower_phase="down",
            lower_angle=self._down_angle,
            upper_angle=self._up_angle,
            min_transition_seconds=self._min_transition,
        )
        scoring_config = config_section(config, "scoring")
        self._scorer = RepScorer(scoring_config)
        self._score_tracker = SetScoreTracker()
        tempo_config = config_section(scoring_config, "tempo")
        range_failures = config_section(scoring_config, "range_failure_angles")
        posture_failures = config_section(scoring_config, "posture_failure")
        self._ideal_rep_duration = float(tempo_config["ideal_rep_duration_seconds"])
        self._tempo_tolerance = float(tempo_config["deviation_for_zero_score_seconds"])
        self._range_failure_angle = float(range_failures["squat"])
        self._poor_trunk_lean = float(posture_failures["squat_trunk_lean_degrees"])
        self._poor_heel_lift = float(posture_failures["squat_heel_lift_ratio"])
        self._max_symmetry_difference = float(
            scoring_config["max_symmetry_difference_degrees"]
        )
        self._min_rep_knee_angle: float | None = None
        self._max_rep_trunk_lean = 0.0
        self._max_rep_heel_lift = 0.0
        self._rep_started: float | None = None
        self._rep_mistakes: set[str] = set()

    @property
    def score_summary(self) -> ScoreSummary:
        """Return the scores and mistake counts accumulated for this set."""
        return self._score_tracker.summary

    def update(
        self, landmarks: Sequence[LandmarkPoint] | None, timestamp: float
    ) -> FrameResult:
        """Update the squat state from one MediaPipe landmark observation."""
        if landmarks is None:
            return self._missing_result()

        side = self._side_selector.update(landmarks)
        side_joints = LEFT if side == "left" else RIGHT
        if not self._has_required_landmarks(landmarks, side_joints):
            return self._missing_result()

        points = {name: landmarks[index] for name, index in side_joints.items()}
        try:
            knee_angle = self._smoother.update(
                angle_between(points["hip"], points["knee"], points["ankle"])
            )
            hip_angle = angle_between(
                points["shoulder"], points["hip"], points["knee"]
            )
            trunk_lean = vertical_angle(points["hip"], points["shoulder"])
        except ValueError:
            return self._missing_result()
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

        was_down = self._repetitions.phase == "down"
        if was_down:
            self._min_rep_knee_angle = (
                knee_angle
                if self._min_rep_knee_angle is None
                else min(self._min_rep_knee_angle, knee_angle)
            )
            self._max_rep_trunk_lean = max(self._max_rep_trunk_lean, trunk_lean)
            self._max_rep_heel_lift = max(self._max_rep_heel_lift, heel_lift_ratio)
        transition = self._repetitions.update(knee_angle, timestamp)
        if transition.phase == "down" and not was_down:
            self._min_rep_knee_angle = knee_angle
            self._rep_started = timestamp
            self._max_rep_trunk_lean = trunk_lean
            self._max_rep_heel_lift = heel_lift_ratio
            self._rep_mistakes.clear()
        rep_was_shallow = False
        if transition.rep_completed:
            rep_was_shallow = (
                self._min_rep_knee_angle is None
                or self._min_rep_knee_angle > self._depth_angle
            )

        feedback = self._form_feedback(trunk_lean, heel_lift_ratio)
        if rep_was_shallow:
            feedback.insert(0, "Go lower")
        rep_score: float | None = None
        if transition.phase == "down" or transition.rep_completed:
            self._rep_mistakes.update(feedback)
        if transition.rep_completed:
            rep_score = self._score_completed_rep(timestamp, landmarks)
            self._score_tracker.add_rep(rep_score, self._rep_mistakes)
            self._min_rep_knee_angle = None
            self._rep_started = None
            self._rep_mistakes.clear()

        return FrameResult(
            phase=transition.phase,
            rep_count=self._repetitions.rep_count,
            angles=angles,
            feedback=feedback,
            rep_completed=transition.rep_completed,
            rep_score=rep_score,
            set_score=self.score_summary.average_score,
        )

    def _score_completed_rep(
        self, timestamp: float, landmarks: Sequence[LandmarkPoint]
    ) -> float:
        min_knee_angle = (
            self._min_rep_knee_angle
            if self._min_rep_knee_angle is not None
            else self._range_failure_angle
        )
        range_score = threshold_score(
            min_knee_angle, self._depth_angle, self._range_failure_angle
        )
        trunk_score = threshold_score(
            self._max_rep_trunk_lean,
            self._max_trunk_lean,
            self._poor_trunk_lean,
        )
        heel_score = threshold_score(
            self._max_rep_heel_lift,
            self._max_heel_lift_ratio,
            self._poor_heel_lift,
        )
        duration = (
            max(0.0, timestamp - self._rep_started)
            if self._rep_started is not None
            else 0.0
        )
        tempo = tempo_score(duration, self._ideal_rep_duration, self._tempo_tolerance)
        symmetry = self._bilateral_score(landmarks)
        return self._scorer.score(
            range_score,
            (trunk_score + heel_score) / 2,
            tempo,
            symmetry,
        )

    def _bilateral_score(
        self, landmarks: Sequence[LandmarkPoint]
    ) -> float | None:
        angles: list[float] = []
        for joints in (LEFT, RIGHT):
            indices = (joints["hip"], joints["knee"], joints["ankle"])
            if any(
                index >= len(landmarks)
                or landmarks[index].visibility < self._visibility_threshold
                for index in indices
            ):
                return None
            try:
                angles.append(
                    angle_between(*(landmarks[index] for index in indices))
                )
            except ValueError:
                return None
        return symmetry_score(
            angles[0], angles[1], self._max_symmetry_difference
        )

    def _has_required_landmarks(
        self, landmarks: Sequence[LandmarkPoint], joints: dict[str, int]
    ) -> bool:
        return all(
            index < len(landmarks)
            and landmarks[index].visibility >= self._visibility_threshold
            for index in joints.values()
        )

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

    def _missing_result(self) -> FrameResult:
        self._repetitions.reset_candidate()
        self._smoother.reset()
        return FrameResult(
            phase=self._repetitions.phase,
            rep_count=self._repetitions.rep_count,
            feedback=["Move into frame"],
        )
