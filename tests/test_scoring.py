"""Tests for weighted rep scores and set summaries."""

import pytest

from main import load_config
from src.scoring import (
    RepScorer,
    SetScoreTracker,
    symmetry_score,
    tempo_score,
    threshold_score,
)


def test_rep_score_uses_configured_weights_with_symmetry() -> None:
    scorer = RepScorer(load_config()["scoring"])

    score = scorer.score(100.0, 80.0, 60.0, 40.0)

    assert score == pytest.approx(78.0)


def test_missing_symmetry_weight_is_redistributed_proportionally() -> None:
    scorer = RepScorer(load_config()["scoring"])

    score = scorer.score(100.0, 80.0, 60.0, None)

    assert score == pytest.approx((30.0 + 32.0 + 12.0) / 90.0 * 100.0)


def test_component_score_helpers_use_configured_thresholds() -> None:
    assert threshold_score(10.0, 10.0, 30.0) == 100.0
    assert threshold_score(20.0, 10.0, 30.0) == 50.0
    assert threshold_score(30.0, 10.0, 30.0) == 0.0
    assert tempo_score(3.0, 3.0, 1.5) == 100.0
    assert tempo_score(4.5, 3.0, 1.5) == 0.0
    assert symmetry_score(90.0, 105.0, 30.0) == 50.0


def test_set_tracker_averages_rep_scores_and_counts_unique_mistakes() -> None:
    tracker = SetScoreTracker()
    tracker.add_rep(80.0, ["Go lower", "Go lower"])
    tracker.add_rep(60.0, ["Go lower", "Keep your chest up"])

    assert tracker.summary.repetitions == 2
    assert tracker.summary.average_score == 70.0
    assert tracker.summary.common_mistakes == (
        ("Go lower", 2),
        ("Keep your chest up", 1),
    )


def test_set_tracker_uses_hold_alignment_scores_without_reps() -> None:
    tracker = SetScoreTracker()
    tracker.add_hold_score(80.0)
    tracker.add_hold_score(100.0)

    assert tracker.summary.repetitions == 0
    assert tracker.summary.average_score == 90.0


def test_rep_scorer_rejects_negative_weights() -> None:
    config = load_config()["scoring"]
    config["weights"]["symmetry"] = -1.0

    with pytest.raises(ValueError, match="cannot be negative"):
        RepScorer(config)
