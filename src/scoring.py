"""Rep scoring and set-level score summaries."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


def config_section(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a required mapping-valued scoring config section."""
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


@dataclass(frozen=True)
class ScoreSummary:
    """Aggregate score and common form mistakes for one exercise set."""

    repetitions: int
    average_score: float | None
    common_mistakes: tuple[tuple[str, int], ...]


class RepScorer:
    """Combine normalized form components using configured score weights."""

    def __init__(self, config: Mapping[str, object]) -> None:
        weights_config = config["weights"]
        if not isinstance(weights_config, Mapping):
            raise ValueError("Scoring weights must be a mapping.")
        self._weights = {
            name: float(weights_config[name])
            for name in ("range_of_motion", "posture_alignment", "tempo", "symmetry")
        }
        if any(weight < 0 for weight in self._weights.values()):
            raise ValueError("Scoring weights cannot be negative.")
        if sum(self._weights.values()) == 0:
            raise ValueError("At least one scoring weight must be positive.")

    def score(
        self,
        range_of_motion: float,
        posture_alignment: float,
        tempo: float,
        symmetry: float | None,
    ) -> float:
        """Return a weighted 0-100 score, redistributing unavailable symmetry."""
        components = {
            "range_of_motion": range_of_motion,
            "posture_alignment": posture_alignment,
            "tempo": tempo,
        }
        if symmetry is not None:
            components["symmetry"] = symmetry
        for component in components.values():
            if not math.isfinite(component):
                raise ValueError("Score components must be finite.")
        total_weight = sum(self._weights[name] for name in components)
        if total_weight == 0:
            raise ValueError("Available score components have zero total weight.")
        weighted_score = sum(
            self._weights[name] * min(100.0, max(0.0, value))
            for name, value in components.items()
        )
        return weighted_score / total_weight


def threshold_score(value: float, target: float, failure: float) -> float:
    """Map a lower-is-better measurement to a configured 0-100 score."""
    if failure <= target:
        raise ValueError("failure threshold must be greater than target.")
    if value <= target:
        return 100.0
    if value >= failure:
        return 0.0
    return 100.0 * (failure - value) / (failure - target)


def tempo_score(
    duration: float, ideal_duration: float, zero_score_deviation: float
) -> float:
    """Score rep duration linearly by deviation from configured ideal tempo."""
    if zero_score_deviation <= 0:
        raise ValueError("zero_score_deviation must be greater than zero.")
    deviation = abs(duration - ideal_duration)
    return max(0.0, 100.0 * (1.0 - deviation / zero_score_deviation))


def symmetry_score(
    first_angle: float, second_angle: float, max_difference: float
) -> float:
    """Score bilateral angle agreement, or reject an invalid tolerance."""
    if max_difference <= 0:
        raise ValueError("max_difference must be greater than zero.")
    difference = abs(first_angle - second_angle)
    return max(0.0, 100.0 * (1.0 - difference / max_difference))


class SetScoreTracker:
    """Track rep or hold scores and count distinct corrective cues."""

    def __init__(self) -> None:
        self._rep_scores: list[float] = []
        self._hold_scores: list[float] = []
        self._mistakes: Counter[str] = Counter()

    def add_rep(self, score: float, mistakes: Iterable[str]) -> None:
        """Record one completed repetition and its unique cues."""
        self._rep_scores.append(score)
        self.add_mistakes(mistakes)

    def add_hold_score(self, score: float) -> None:
        """Record one valid plank frame's alignment score."""
        self._hold_scores.append(score)

    def add_mistakes(self, mistakes: Iterable[str]) -> None:
        """Increment the count for each supplied cue."""
        self._mistakes.update(set(mistakes))

    @property
    def summary(self) -> ScoreSummary:
        """Return rep mean, or mean valid hold alignment when no reps exist."""
        scores = self._rep_scores or self._hold_scores
        average = sum(scores) / len(scores) if scores else None
        common = tuple(
            sorted(self._mistakes.items(), key=lambda item: (-item[1], item[0]))
        )
        return ScoreSummary(
            repetitions=len(self._rep_scores),
            average_score=average,
            common_mistakes=common,
        )
