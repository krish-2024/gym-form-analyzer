"""Synthetic tests for push-up counting and form feedback."""

import math
from types import SimpleNamespace

from main import load_config
from src.exercises.pushup import LEFT, Pushup


def make_landmarks(
    elbow_angle: float = 180.0, *, hip_y: float = 0.0
) -> list[SimpleNamespace]:
    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.4)
        for _ in range(33)
    ]
    positions = {
        "shoulder": (0.0, 0.0),
        "elbow": (1.0, 0.0),
        "wrist": (
            1.0 + math.cos(math.radians(180.0 - elbow_angle)),
            math.sin(math.radians(180.0 - elbow_angle)),
        ),
        "hip": (1.0, hip_y),
        "ankle": (2.0, 0.0),
    }
    for name, index in LEFT.items():
        x, y = positions[name]
        landmarks[index] = SimpleNamespace(x=x, y=y, z=0.0, visibility=0.9)
    return landmarks


def make_pushup() -> Pushup:
    config = load_config()
    return Pushup(
        {
            **config["pushup"],
            "smoothing_window_size": 1,
            "scoring": config["scoring"],
        }
    )


def test_counts_complete_pushup_and_reports_sagging_hips() -> None:
    pushup = make_pushup()
    pushup.update(make_landmarks(), 0.0)
    pushup.update(make_landmarks(89.0, hip_y=0.5), 0.1)
    down = pushup.update(make_landmarks(89.0, hip_y=0.5), 0.41)
    pushup.update(make_landmarks(), 0.5)
    completed = pushup.update(make_landmarks(), 0.81)

    assert down.phase == "down"
    assert "Keep your hips from sagging" in down.feedback
    assert completed.rep_count == 1
    assert completed.rep_completed


def test_missing_required_joint_skips_pushup_frame() -> None:
    landmarks = make_landmarks()
    landmarks[LEFT["wrist"]].visibility = 0.1

    result = make_pushup().update(landmarks, 0.0)

    assert result.feedback == ["Move into frame"]
    assert result.angles == {}


def test_high_hips_receive_a_lowering_cue() -> None:
    result = make_pushup().update(make_landmarks(hip_y=-0.5), 0.0)

    assert "Lower your hips" in result.feedback


def test_partial_bend_reversal_gets_a_cue_without_counting() -> None:
    pushup = make_pushup()
    pushup.update(make_landmarks(), 0.0)
    pushup.update(make_landmarks(100.0), 0.1)
    result = pushup.update(make_landmarks(120.0), 0.2)

    assert "Bend lower" in result.feedback
    assert result.rep_count == 0
