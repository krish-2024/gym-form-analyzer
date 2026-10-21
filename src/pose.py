"""Classic MediaPipe Pose detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from src.exercises.base import LandmarkPoint


@dataclass(frozen=True)
class PoseLandmark:
    """A MediaPipe landmark projected into image-pixel x/y coordinates."""

    x: float
    y: float
    z: float
    visibility: float


class PoseDetector:
    """Detect and draw classic MediaPipe Pose landmarks on video frames."""

    def __init__(self, config: dict[str, object]) -> None:
        self._pose_module = mp.solutions.pose
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._detector = self._pose_module.Pose(
            static_image_mode=False,
            model_complexity=int(config["model_complexity"]),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(config["min_detection_confidence"]),
            min_tracking_confidence=float(config["min_tracking_confidence"]),
        )
        self._landmarks: tuple[PoseLandmark, ...] | None = None
        self._pose_landmarks = None

    def process(self, frame: np.ndarray) -> Sequence[LandmarkPoint] | None:
        """Estimate pose landmarks for a BGR frame in image-pixel coordinates."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._detector.process(rgb_frame)
        height, width = frame.shape[:2]
        self._landmarks = None
        if results.pose_landmarks:
            self._landmarks = tuple(
                PoseLandmark(
                    x=landmark.x * width,
                    y=landmark.y * height,
                    z=landmark.z,
                    visibility=landmark.visibility,
                )
                for landmark in results.pose_landmarks.landmark
            )
        self._pose_landmarks = results.pose_landmarks
        return self._landmarks

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the last detected pose on a BGR frame and return that frame."""
        if self._pose_landmarks is not None:
            self._drawing.draw_landmarks(
                frame,
                self._pose_landmarks,
                self._pose_module.POSE_CONNECTIONS,
                landmark_drawing_spec=self._styles.get_default_pose_landmarks_style(),
            )
        return frame

    def close(self) -> None:
        """Release MediaPipe graph resources."""
        self._detector.close()
