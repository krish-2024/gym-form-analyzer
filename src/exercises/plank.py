"""Side-view plank hold and alignment analysis."""

from __future__ import annotations

from typing import Sequence

from src.angles import angle_between
from src.exercises.base import Exercise, FrameResult, LandmarkPoint
from src.exercises.common import SideSelector, body_line_metrics
from src.scoring import ScoreSummary, SetScoreTracker
from src.smoothing import MovingAverage

LEFT = {"ear": 7, "shoulder": 11, "hip": 23, "ankle": 27}
RIGHT = {"ear": 8, "shoulder": 12, "hip": 24, "ankle": 28}


class Plank(Exercise):
    """Track aligned hold duration, alignment score, and posture cues."""

    def __init__(self, config: dict[str, object]) -> None:
        self._visibility_threshold = float(config["landmark_visibility_threshold"])
        self._ideal_body_alignment = float(config["ideal_body_alignment_degrees"])
        self._ideal_head_alignment = float(config["ideal_head_alignment_degrees"])
        self._alignment_tolerance = float(config["alignment_tolerance_degrees"])
        self._hip_offset_tolerance = float(config["hip_offset_tolerance_ratio"])
        self._head_droop_tolerance = float(config["head_droop_degrees"])
        self._side_selector = SideSelector(
            {
                "left": tuple(LEFT.values()),
                "right": tuple(RIGHT.values()),
            },
            int(config["side_visibility_window"]),
            self._visibility_threshold,
        )
        smoothing_window = int(config["smoothing_window_size"])
        self._alignment_smoother = MovingAverage(smoothing_window)
        self._head_smoother = MovingAverage(smoothing_window)
        self._hold_started: float | None = None
        self._accumulated_hold = 0.0
        self._score_tracker = SetScoreTracker()
        self._active_mistakes: set[str] = set()

    @property
    def score_summary(self) -> ScoreSummary:
        """Return mean alignment and common cue statistics for the hold session."""
        return self._score_tracker.summary

    def update(
        self, landmarks: Sequence[LandmarkPoint] | None, timestamp: float
    ) -> FrameResult:
        """Update continuous aligned hold time and posture feedback."""
        if landmarks is None:
            self._end_hold(timestamp)
            return self._missing_result()
        side = self._side_selector.update(landmarks)
        joints = LEFT if side == "left" else RIGHT
        if not self._has_required_landmarks(landmarks, joints):
            self._end_hold(timestamp)
            return self._missing_result()

        points = {name: landmarks[index] for name, index in joints.items()}
        try:
            alignment_angle, hip_offset = body_line_metrics(
                points["shoulder"], points["hip"], points["ankle"]
            )
            alignment_angle = self._alignment_smoother.update(alignment_angle)
            head_angle = angle_between(
                points["ear"], points["shoulder"], points["hip"]
            )
            head_angle = self._head_smoother.update(head_angle)
        except ValueError:
            self._end_hold(timestamp)
            return self._missing_result()

        angle_deviation = abs(self._ideal_body_alignment - alignment_angle)
        ratio_deviation = abs(hip_offset) / self._hip_offset_tolerance
        angle_score = angle_deviation / self._alignment_tolerance
        head_drooping = (
            points["ear"].y > points["shoulder"].y
            and self._ideal_head_alignment - head_angle > self._head_droop_tolerance
        )
        head_score = (
            (self._ideal_head_alignment - head_angle) / self._head_droop_tolerance
            if head_drooping
            else 0.0
        )
        alignment_score = max(
            0.0,
            100.0 * (1.0 - max(angle_score, ratio_deviation, head_score)),
        )
        feedback: list[str] = []
        if hip_offset < -self._hip_offset_tolerance:
            feedback.append("Lower your hips")
        elif hip_offset > self._hip_offset_tolerance:
            feedback.append("Keep your hips from sagging")
        if head_drooping:
            feedback.append("Keep your head in line")

        aligned = (
            angle_deviation <= self._alignment_tolerance
            and abs(hip_offset) <= self._hip_offset_tolerance
            and not head_drooping
        )
        if aligned:
            if self._hold_started is None:
                self._hold_started = timestamp
            phase = "holding"
        else:
            self._end_hold(timestamp)
            phase = "correcting"
        self._score_tracker.add_hold_score(alignment_score)
        current_mistakes = set(feedback)
        self._score_tracker.add_mistakes(current_mistakes - self._active_mistakes)
        self._active_mistakes = current_mistakes
        hold_time = self._current_hold_time(timestamp)
        return FrameResult(
            phase=phase,
            rep_count=0,
            angles={
                "body_alignment": alignment_angle,
                "hip_offset_ratio": hip_offset,
                "head_alignment": head_angle,
            },
            feedback=feedback,
            hold_time=hold_time,
            alignment_score=alignment_score,
            set_score=self.score_summary.average_score,
        )

    def _has_required_landmarks(
        self, landmarks: Sequence[LandmarkPoint], joints: dict[str, int]
    ) -> bool:
        return all(
            index < len(landmarks)
            and landmarks[index].visibility >= self._visibility_threshold
            for index in joints.values()
        )

    def _end_hold(self, timestamp: float) -> None:
        if self._hold_started is not None:
            self._accumulated_hold += max(0.0, timestamp - self._hold_started)
            self._hold_started = None

    def _current_hold_time(self, timestamp: float) -> float:
        if self._hold_started is None:
            return self._accumulated_hold
        return self._accumulated_hold + max(0.0, timestamp - self._hold_started)

    def _missing_result(self) -> FrameResult:
        self._alignment_smoother.reset()
        self._head_smoother.reset()
        self._active_mistakes.clear()
        return FrameResult(
            phase="correcting",
            rep_count=0,
            feedback=["Move into frame"],
            hold_time=self._accumulated_hold,
            set_score=self.score_summary.average_score,
        )
