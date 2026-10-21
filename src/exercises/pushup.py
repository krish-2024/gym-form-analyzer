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
        scoring_config = config_section(config, "scoring")
        self._scorer = RepScorer(scoring_config)
        self._score_tracker = SetScoreTracker()
        tempo_config = config_section(scoring_config, "tempo")
        range_failures = config_section(scoring_config, "range_failure_angles")
        posture_failures = config_section(scoring_config, "posture_failure")
        self._ideal_rep_duration = float(tempo_config["ideal_rep_duration_seconds"])
        self._tempo_tolerance = float(tempo_config["deviation_for_zero_score_seconds"])
        self._range_failure_angle = float(range_failures["pushup"])
        self._poor_alignment = float(
            posture_failures["pushup_alignment_deviation_degrees"]
        )
        self._poor_hip_offset = float(posture_failures["pushup_hip_offset_ratio"])
        self._max_symmetry_difference = float(
            scoring_config["max_symmetry_difference_degrees"]
        )
        self._rep_started: float | None = None
        self._max_alignment_deviation = 0.0
        self._max_hip_offset = 0.0
        self._rep_mistakes: set[str] = set()

    @property
    def score_summary(self) -> ScoreSummary:
        """Return the scores and mistake counts accumulated for this set."""
        return self._score_tracker.summary

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
            self._max_alignment_deviation = max(
                self._max_alignment_deviation,
                abs(self._ideal_alignment - alignment_angle),
            )
            self._max_hip_offset = max(self._max_hip_offset, abs(hip_offset))
        transition = self._repetitions.update(elbow_angle, timestamp)
        if transition.phase == "down" and not was_down:
            self._minimum_elbow_angle = elbow_angle
            self._rep_started = timestamp
            self._max_alignment_deviation = abs(
                self._ideal_alignment - alignment_angle
            )
            self._max_hip_offset = abs(hip_offset)
            self._rep_mistakes.clear()
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
        rep_score: float | None = None
        if transition.phase == "down" or transition.rep_completed:
            self._rep_mistakes.update(feedback)
        if transition.rep_completed:
            rep_score = self._score_completed_rep(timestamp, landmarks)
            self._score_tracker.add_rep(rep_score, self._rep_mistakes)
            self._minimum_elbow_angle = None
            self._rep_started = None
            self._rep_mistakes.clear()

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
            rep_score=rep_score,
            set_score=self.score_summary.average_score,
        )

    def _score_completed_rep(
        self, timestamp: float, landmarks: Sequence[LandmarkPoint]
    ) -> float:
        minimum_angle = (
            self._minimum_elbow_angle
            if self._minimum_elbow_angle is not None
            else self._range_failure_angle
        )
        range_score = threshold_score(
            minimum_angle, self._depth_angle, self._range_failure_angle
        )
        alignment_score = threshold_score(
            self._max_alignment_deviation,
            self._posture_tolerance,
            self._poor_alignment,
        )
        hip_score = threshold_score(
            self._max_hip_offset,
            float(config_section(self._config, "unused")["unused"]),
            self._poor_hip_offset,
        )
        duration = (
            max(0.0, timestamp - self._rep_started)
            if self._rep_started is not None
            else 0.0
        )
        tempo = tempo_score(duration, self._ideal_rep_duration, self._tempo_tolerance)
        return self._scorer.score(
            range_score,
            (alignment_score + hip_score) / 2,
            tempo,
            self._bilateral_score(landmarks),
        )

    def _bilateral_score(
        self, landmarks: Sequence[LandmarkPoint]
    ) -> float | None:
        angles: list[float] = []
        for joints in (LEFT, RIGHT):
            indices = (joints["shoulder"], joints["elbow"], joints["wrist"])
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

    def _missing_result(self) -> FrameResult:
        self._repetitions.reset_candidate()
        self._elbow_smoother.reset()
        self._partial_range.reset()
        return FrameResult(
            phase=self._repetitions.phase,
            rep_count=self._repetitions.rep_count,
            feedback=["Move into frame"],
        )
