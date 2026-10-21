"""Synthetic tests for squat state and form logic."""

from types import SimpleNamespace

import pytest

from src.exercises.squat import LEFT, RIGHT, Squat

BASE_CONFIG: dict[str, float | int] = {
    "landmark_visibility_threshold": 0.5,
    "smoothing_window_size": 1,
    "side_visibility_window": 3,
    "knee_down_angle": 100.0,
    "knee_up_angle": 160.0,
    "min_transition_seconds": 0.3,
    "depth_knee_angle": 100.0,
    "max_trunk_lean_degrees": 45.0,
    "max_heel_lift_ratio": 0.04,
}


def make_landmarks(
    knee_angle: float = 180.0,
    *,
    left_visibility: float = 1.0,
    right_visibility: float = 0.4,
    trunk_lean: bool = False,
    lifted_heel: bool = False,
) -> list[SimpleNamespace]:
    """Build normalized pose landmarks with a configurable knee angle."""
    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0)
        for _ in range(33)
    ]
    x_offset = 1.0 if knee_angle < 180.0 else 0.0
    coordinates = {
        "shoulder": (2.0 if trunk_lean else 0.0, -1.0),
        "hip": (0.0, 0.0),
        "knee": (0.0, 1.0),
        "ankle": (x_offset, 1.0 if knee_angle < 180.0 else 2.0),
        "heel": (
            x_offset,
            (0.7 if lifted_heel else 1.05)
            if knee_angle < 180.0
            else (1.7 if lifted_heel else 2.05),
        ),
    }
    for joints, visibility in (
        (LEFT, left_visibility),
        (RIGHT, right_visibility),
    ):
        for name, index in joints.items():
            x, y = coordinates[name]
            landmarks[index] = SimpleNamespace(
                x=x, y=y, z=0.0, visibility=visibility
            )
    return landmarks


def make_squat(**overrides: float | int) -> Squat:
    return Squat({**BASE_CONFIG, **overrides})


def test_counts_one_rep_only_after_full_down_up_cycle() -> None:
    squat = make_squat()
    squat.update(make_landmarks(), 0.0)
    squat.update(make_landmarks(90.0), 0.1)
    down_result = squat.update(make_landmarks(90.0), 0.41)
    squat.update(make_landmarks(180.0), 0.5)
    completed = squat.update(make_landmarks(180.0), 0.81)

    assert down_result.phase == "down"
    assert completed.phase == "up"
    assert completed.rep_count == 1
    assert completed.rep_completed


def test_transition_shorter_than_minimum_is_ignored() -> None:
    squat = make_squat()
    squat.update(make_landmarks(), 0.0)
    squat.update(make_landmarks(90.0), 0.1)
    squat.update(make_landmarks(180.0), 0.2)

    assert squat.update(make_landmarks(180.0), 0.5).phase == "up"
    assert squat.update(make_landmarks(90.0), 0.6).rep_count == 0


def test_missing_or_low_visibility_landmarks_skip_frame() -> None:
    squat = make_squat()
    assert squat.update(None, 0.0).feedback == ["Move into frame"]

    landmarks = make_landmarks()
    landmarks[LEFT["ankle"]].visibility = 0.1
    result = squat.update(landmarks, 0.1)

    assert result.feedback == ["Move into frame"]
    assert result.angles == {}


def test_side_selection_uses_window_average_not_last_frame() -> None:
    squat = make_squat()
    first = make_landmarks(left_visibility=0.7, right_visibility=0.9)
    first[RIGHT["ankle"]] = SimpleNamespace(
        x=1.0, y=1.0, z=0.0, visibility=0.9
    )
    second = make_landmarks(left_visibility=0.99, right_visibility=0.51)
    squat.update(first, 0.0)

    result = squat.update(second, 0.1)

    assert result.angles["knee"] == pytest.approx(180.0)


def test_form_feedback_uses_trunk_lean_and_leg_length_ratio() -> None:
    squat = make_squat()
    result = squat.update(
        make_landmarks(trunk_lean=True, lifted_heel=True),
        0.0,
    )

    assert "Keep your chest up" in result.feedback
    assert "Keep heels down" in result.feedback
