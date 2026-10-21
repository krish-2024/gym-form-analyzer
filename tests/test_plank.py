"""Synthetic tests for plank alignment and hold tracking."""

from types import SimpleNamespace

from main import load_config
from src.exercises.plank import LEFT, Plank


def make_landmarks(
    *, hip_y: float = 0.0, drooping_head: bool = False
) -> list[SimpleNamespace]:
    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.4)
        for _ in range(33)
    ]
    positions = {
        "ear": (0.2, 0.5) if drooping_head else (-0.3, 0.0),
        "shoulder": (0.0, 0.0),
        "hip": (1.0, hip_y),
        "ankle": (2.0, 0.0),
    }
    for name, index in LEFT.items():
        x, y = positions[name]
        landmarks[index] = SimpleNamespace(x=x, y=y, z=0.0, visibility=0.9)
    return landmarks


def make_plank() -> Plank:
    return Plank({**load_config()["plank"], "smoothing_window_size": 1})


def test_aligned_plank_reports_hold_time_and_perfect_alignment() -> None:
    plank = make_plank()
    first = plank.update(make_landmarks(), 0.0)
    later = plank.update(make_landmarks(), 3.0)

    assert first.phase == "holding"
    assert first.rep_count == 0
    assert first.hold_time == 0.0
    assert later.hold_time == 3.0
    assert later.alignment_score == 100.0


def test_plank_flags_hips_and_head_out_of_alignment() -> None:
    result = make_plank().update(
        make_landmarks(hip_y=0.2, drooping_head=True),
        0.0,
    )

    assert "Keep your hips from sagging" in result.feedback
    assert "Keep your head in line" in result.feedback
    assert result.alignment_score == 0.0


def test_plank_accumulates_only_valid_hold_segments() -> None:
    plank = make_plank()
    plank.update(make_landmarks(), 0.0)
    plank.update(make_landmarks(), 2.0)
    missing = plank.update(None, 2.0)
    plank.update(make_landmarks(), 3.0)
    later = plank.update(make_landmarks(), 5.0)

    assert missing.hold_time == 2.0
    assert later.hold_time == 4.0


def test_drooping_head_prevents_a_valid_hold() -> None:
    result = make_plank().update(make_landmarks(drooping_head=True), 0.0)

    assert result.phase == "correcting"
    assert "Keep your head in line" in result.feedback
