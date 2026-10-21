"""Synthetic tests for curl counting and corrective feedback."""

import math
from types import SimpleNamespace

from main import load_config
from src.exercises.curl import LEFT, Curl


def make_landmarks(
    elbow_angle: float = 180.0,
    *,
    torso_lean: float = 0.0,
    elbow_offset: float = 0.1,
) -> list[SimpleNamespace]:
    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.4)
        for _ in range(33)
    ]
    shoulder = (math.tan(math.radians(torso_lean)), 0.0)
    hip = (0.0, 1.0)
    elbow = (shoulder[0] + elbow_offset, 0.4)
    first = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    first_angle = math.atan2(first[1], first[0]) + math.radians(elbow_angle)
    wrist = (
        elbow[0] + math.cos(first_angle),
        elbow[1] + math.sin(first_angle),
    )
    positions = {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "hip": hip,
    }
    for name, index in LEFT.items():
        x, y = positions[name]
        landmarks[index] = SimpleNamespace(x=x, y=y, z=0.0, visibility=0.9)
    return landmarks


def make_curl() -> Curl:
    config = load_config()
    return Curl(
        {
            **config["curl"],
            "smoothing_window_size": 1,
            "scoring": config["scoring"],
        }
    )


def test_counts_full_curl_and_feedback_detects_torso_swing() -> None:
    curl = make_curl()
    curl.update(make_landmarks(), 0.0)
    curl.update(make_landmarks(40.0), 0.1)
    curl.update(make_landmarks(40.0), 0.41)
    swing = curl.update(make_landmarks(40.0, torso_lean=15.0), 0.5)
    curl.update(make_landmarks(), 0.6)
    completed = curl.update(make_landmarks(), 0.91)

    assert "Keep your torso still" in swing.feedback
    assert completed.rep_count == 1
    assert completed.rep_completed


def test_elbow_drift_feedback_uses_torso_length_ratio() -> None:
    result = make_curl().update(
        make_landmarks(40.0, elbow_offset=0.5),
        0.0,
    )

    assert result.angles["elbow_drift_ratio"] > 0.2
    assert "Keep your elbows close to your sides" in result.feedback


def test_partial_curl_reversal_gets_a_cue_without_counting() -> None:
    curl = make_curl()
    curl.update(make_landmarks(), 0.0)
    curl.update(make_landmarks(60.0), 0.1)
    result = curl.update(make_landmarks(70.0), 0.2)

    assert "Curl through the full range" in result.feedback
    assert result.rep_count == 0
