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
        scoring_config = config_section(config, "scoring")
        self._scorer = RepScorer(scoring_config)
        self._score_tracker = SetScoreTracker()
        tempo_config = config_section(scoring_config, "tempo")
        range_failures = config_section(scoring_config, "range_failure_angles")
        posture_failures = config_section(scoring_config, "posture_failure")
        self._ideal_rep_duration = float(tempo_config["ideal_rep_duration_seconds"])
        self._tempo_tolerance = float(tempo_config["deviation_for_zero_score_seconds"])
        self._range_failure_angle = float(range_failures["curl"])
        self._poor_torso_swing = float(posture_failures["curl_torso_swing_degrees"])
        self._poor_elbow_drift = float(posture_failures["curl_elbow_drift_ratio"])
        self._max_symmetry_difference = float(
            scoring_config["max_symmetry_difference_degrees"]
        )
        self._rep_started: float | None = None
        self._max_rep_elbow_drift = 0.0
        self._rep_mistakes: set[str] = set()

    @property
    def score_summary(self) -> ScoreSummary:
        """Return the scores and mistake counts accumulated for this set."""
        return self._score_tracker.summary

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
            self._max_rep_elbow_drift = max(
                self._max_rep_elbow_drift, elbow_drift_ratio
            )
        transition = self._repetitions.update(elbow_angle, timestamp)
        if transition.phase == "curled" and not was_curled:
            self._rep_min_torso = torso_angle
            self._rep_max_torso = torso_angle
            self._minimum_elbow_angle = elbow_angle
            self._rep_started = timestamp
            self._max_rep_elbow_drift = elbow_drift_ratio
            self._rep_mistakes.clear()

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
        rep_torso_range = torso_range
        rep_score: float | None = None
        if transition.phase == "curled" or transition.rep_completed:
            self._rep_mistakes.update(feedback)
        if transition.rep_completed:
            rep_score = self._score_completed_rep(timestamp, landmarks, rep_torso_range)
            self._score_tracker.add_rep(rep_score, self._rep_mistakes)
            self._rep_min_torso = None
            self._rep_max_torso = None
            self._minimum_elbow_angle = None
            self._rep_started = None
            self._rep_mistakes.clear()

        return FrameResult(
            phase=transition.phase,
            rep_count=self._repetitions.rep_count,
            angles={
                "elbow": elbow_angle,
                "torso_swing": rep_torso_range,
                "elbow_drift_ratio": elbow_drift_ratio,
                "torso_lean": torso_angle,
            },
            feedback=feedback,
            rep_completed=transition.rep_completed,
            rep_score=rep_score,
            set_score=self.score_summary.average_score,
        )

    def _score_completed_rep(
        self,
        timestamp: float,
        landmarks: Sequence[LandmarkPoint],
        torso_range: float,
    ) -> float:
        minimum_angle = (
            self._minimum_elbow_angle
            if self._minimum_elbow_angle is not None
            else self._range_failure_angle
        )
        range_score = threshold_score(
            minimum_angle,
            float(self._partial_angle),
            self._range_failure_angle,
        )
        torso_score = threshold_score(
            torso_range, self._max_torso_swing, self._poor_torso_swing
        )
        elbow_score = threshold_score(
            self._max_rep_elbow_drift,
            self._max_elbow_drift_ratio,
            self._poor_elbow_drift,
        )
        duration = (
            max(0.0, timestamp - self._rep_started)
            if self._rep_started is not None
            else 0.0
        )
        tempo = tempo_score(duration, self._ideal_rep_duration, self._tempo_tolerance)
        return self._scorer.score(
            range_score,
            (torso_score + elbow_score) / 2,
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
        self._rep_min_torso = None
        self._rep_max_torso = None
        self._minimum_elbow_angle = None
        self._max_rep_elbow_drift = 0.0
        self._rep_started = None
        self._rep_mistakes.clear()
        return FrameResult(
            phase=self._repetitions.phase,
            rep_count=self._repetitions.rep_count,
            feedback=["Move into frame"],
        )
