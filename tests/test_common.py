"""Tests for shared exercise state and geometry helpers."""

from types import SimpleNamespace

import pytest

from src.exercises.common import (
    PartialRangeTracker,
    RepetitionStateMachine,
    SideSelector,
    body_line_metrics,
)


def make_landmarks(
    left_visibility: float, right_visibility: float
) -> list[SimpleNamespace]:
    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0)
        for _ in range(4)
    ]
    landmarks[0].visibility = left_visibility
    landmarks[2].visibility = right_visibility
    return landmarks


def test_side_selector_uses_rolling_visibility_average() -> None:
    selector = SideSelector(
        {"left": (0,), "right": (2,)},
        window_size=2,
        visibility_threshold=0.5,
    )

    assert selector.update(make_landmarks(0.7, 0.9)) == "right"
    assert selector.update(make_landmarks(0.99, 0.51)) == "left"


def test_repetition_machine_requires_dwell_time_for_both_transitions() -> None:
    machine = RepetitionStateMachine("up", "down", 100, 160, 0.3)
    assert machine.update(90, 0.0).phase == "up"
    assert machine.update(90, 0.31).phase == "down"
    assert not machine.update(170, 0.4).rep_completed
    result = machine.update(170, 0.71)

    assert result.phase == "up"
    assert result.rep_completed
    assert machine.rep_count == 1


def test_repetition_machine_reset_cancels_incomplete_transition() -> None:
    machine = RepetitionStateMachine("up", "down", 100, 160, 0.3)
    machine.update(90, 0.0)
    machine.reset_candidate()

    assert machine.update(90, 0.4).phase == "up"
    assert machine.rep_count == 0


def test_partial_range_tracker_flags_reversal_before_target() -> None:
    tracker = PartialRangeTracker("up", lower_angle=90.0)
    assert not tracker.update(170.0, "up")
    assert not tracker.update(110.0, "up")
    assert tracker.update(115.0, "up")
    assert not tracker.update(80.0, "up")
    assert not tracker.update(100.0, "down")


def test_body_line_metrics_report_alignment_and_signed_offset() -> None:
    shoulder = SimpleNamespace(x=0.0, y=0.0)
    hip = SimpleNamespace(x=1.0, y=0.2)
    ankle = SimpleNamespace(x=2.0, y=0.0)

    angle, offset_ratio = body_line_metrics(shoulder, hip, ankle)

    assert angle < 180.0
    assert offset_ratio == pytest.approx(0.1)
