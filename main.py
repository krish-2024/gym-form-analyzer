"""Command-line entry point for real-time exercise form analysis."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import yaml
from yaml import YAMLError

from src.exercises.base import Exercise, FrameResult
from src.exercises.curl import Curl
from src.exercises.plank import Plank
from src.exercises.pushup import Pushup
from src.exercises.squat import Squat
from src.pose import PoseDetector

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    """Load the application YAML configuration."""
    try:
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except (OSError, YAMLError) as error:
        raise ValueError(f"Could not load configuration file: {path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def _get_section(config: dict[str, object], name: str) -> dict[str, object]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return section


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or path to a video file (default: 0).",
    )
    parser.add_argument(
        "--exercise",
        choices=("squat", "pushup", "curl", "plank"),
        default="squat",
        help="Exercise to analyze.",
    )
    return parser.parse_args(arguments)


def open_capture(source: str) -> cv2.VideoCapture:
    """Open the requested camera or video, raising a clear error on failure."""
    capture_source: int | str = int(source) if source.isdecimal() else source
    if isinstance(capture_source, str) and not Path(capture_source).is_file():
        raise FileNotFoundError(f"Video file does not exist: {capture_source}")

    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open camera or video source: {source}")
    return capture


def draw_result(frame: np.ndarray, result: FrameResult) -> None:
    """Draw the current exercise state and feedback on a video frame."""
    primary_angle = result.angles.get("knee", result.angles.get("elbow"))
    angle_text = (
        f"Joint angle: {primary_angle:.0f} deg"
        if primary_angle is not None
        else "Joint angle: --"
    )
    lines = [angle_text]
    if result.hold_time is None:
        lines.append(f"Phase: {result.phase}  Reps: {result.rep_count}")
    else:
        lines.append(f"Phase: {result.phase}  Hold: {result.hold_time:.1f}s")
    if result.alignment_score is not None:
        lines.append(f"Alignment: {result.alignment_score:.0f}/100")
    lines.extend(result.feedback)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (15, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if not result.feedback else (0, 165, 255),
            2,
        )


def create_exercise(name: str, config: dict[str, object]) -> Exercise:
    """Create an exercise analyzer from its named YAML configuration section."""
    exercise_types: dict[str, Callable[[dict[str, object]], Exercise]] = {
        "squat": Squat,
        "pushup": Pushup,
        "curl": Curl,
        "plank": Plank,
    }
    try:
        exercise_type = exercise_types[name]
    except KeyError as error:
        raise ValueError(f"Unsupported exercise: {name}") from error
    exercise_config = _get_section(config, name).copy()
    exercise_config["smoothing_window_size"] = _get_section(
        config, "smoothing"
    )["window_size"]
    return exercise_type(exercise_config)


def run(source: str, config: dict[str, object], exercise_name: str = "squat") -> int:
    """Run the video analysis loop and release resources on every exit path."""
    capture: cv2.VideoCapture | None = None
    detector: PoseDetector | None = None
    try:
        capture = open_capture(source)
        detector = PoseDetector(_get_section(config, "pose"))
        exercise = create_exercise(exercise_name, config)
        while True:
            success, frame = capture.read()
            if not success:
                LOGGER.error("Could not read a frame from source: %s", source)
                return 1

            landmarks = detector.process(frame)
            result = exercise.update(landmarks, time.monotonic())
            detector.draw(frame)
            draw_result(frame, result)
            cv2.imshow(f"Gym Form Analyzer - {exercise_name.title()}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    except (
        cv2.error,
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        LOGGER.error("%s", error)
        return 1
    finally:
        try:
            if detector is not None:
                detector.close()
        finally:
            try:
                if capture is not None:
                    capture.release()
            finally:
                cv2.destroyAllWindows()


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the configured squat analyzer."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(arguments)
    try:
        config = load_config()
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    try:
        return run(args.source, config, args.exercise)
    except ValueError as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
