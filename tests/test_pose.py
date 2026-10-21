"""Tests for pose detector behavior without a webcam."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.pose import PoseDetector


def test_process_returns_landmarks_and_draw_uses_last_pose(monkeypatch) -> None:
    landmarks = [SimpleNamespace(x=0.5, y=0.25, z=0.0, visibility=1.0)]
    pose_landmarks = SimpleNamespace(landmark=landmarks)
    detector_graph = Mock()
    detector_graph.process.return_value = SimpleNamespace(
        pose_landmarks=pose_landmarks
    )
    detector_graph.close = Mock()
    pose_module = SimpleNamespace(
        Pose=Mock(return_value=detector_graph),
        POSE_CONNECTIONS=object(),
    )
    drawing = SimpleNamespace(draw_landmarks=Mock())
    styles = SimpleNamespace(get_default_pose_landmarks_style=Mock())
    fake_mp = SimpleNamespace(
        solutions=SimpleNamespace(
            pose=pose_module,
            drawing_utils=drawing,
            drawing_styles=styles,
        )
    )
    monkeypatch.setattr("src.pose.mp", fake_mp)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detector = PoseDetector(
        {
            "model_complexity": 1,
            "min_detection_confidence": 0.6,
            "min_tracking_confidence": 0.6,
        }
    )

    result = detector.process(frame)
    returned_frame = detector.draw(frame)

    assert result is not None
    assert result[0].x == 4.0
    assert result[0].y == 2.0
    assert result[0].visibility == 1.0
    assert returned_frame is frame
    drawing.draw_landmarks.assert_called_once()
    detector.close()
    detector_graph.close.assert_called_once()


def test_draw_does_nothing_when_no_pose_was_detected(monkeypatch) -> None:
    detector_graph = Mock()
    detector_graph.process.return_value = SimpleNamespace(pose_landmarks=None)
    detector_graph.close = Mock()
    drawing = SimpleNamespace(draw_landmarks=Mock())
    monkeypatch.setattr(
        "src.pose.mp",
        SimpleNamespace(
            solutions=SimpleNamespace(
                pose=SimpleNamespace(Pose=Mock(return_value=detector_graph)),
                drawing_utils=drawing,
                drawing_styles=SimpleNamespace(),
            )
        ),
    )
    detector = PoseDetector(
        {
            "model_complexity": 1,
            "min_detection_confidence": 0.6,
            "min_tracking_confidence": 0.6,
        }
    )

    assert detector.process(np.zeros((8, 8, 3), dtype=np.uint8)) is None
    detector.draw(np.zeros((8, 8, 3), dtype=np.uint8))
    drawing.draw_landmarks.assert_not_called()
