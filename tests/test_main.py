"""Tests for command-line configuration and source validation."""

import pytest

from main import load_config, open_capture, parse_args


def test_load_config_reads_squat_thresholds() -> None:
    config = load_config()

    assert config["squat"]["knee_down_angle"] == 100.0


def test_load_config_reports_malformed_yaml(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("pose: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not load configuration"):
        load_config(config_path)


def test_open_capture_reports_missing_video(tmp_path) -> None:
    video_path = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        open_capture(str(video_path))


def test_cli_defaults_to_webcam_and_squat() -> None:
    arguments = parse_args([])

    assert arguments.source == "0"
    assert arguments.exercise == "squat"
