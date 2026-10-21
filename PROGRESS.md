# Project Progress

## Phase 1 - Foundation and squat

- [x] Added config-driven classic MediaPipe pose detection and drawing.
- [x] Added 2D joint angles, a five-frame moving average, and side-view squat
  phase/form logic with configurable visibility and transition requirements.
- [x] Added camera/video CLI handling and cleanup on normal and error exits.
- [x] Added synthetic tests for angle math, smoothing, pose wrapper, and squat
  state/feedback behavior.
- [x] Ruff passes and all 17 pytest tests pass.
- [x] Added camera-free test coverage for the Phase 1 logic and CLI.

Known environment mismatch: the existing `.venv` uses Python 3.12.10, not the
specified Python 3.11, and no Python 3.11 interpreter is installed.

Requirement note: the specified squat-down transition (`knee < 100°`) and the
depth warning (`minimum knee angle > 100°`) make the warning unreachable for a
completed cycle; both configured values are retained as specified.

## Phase 2 - Push-up, curl, and plank

- [x] Extracted reusable visibility-side selection and repetition state logic.
- [x] Added push-up, curl, and plank analyzers and config-driven form cues.
- [x] Added plank hold-time and alignment-score result fields and UI output.
- [x] Ruff passes and all 33 pytest tests pass.

Partial-range cues detect a reversal before the configured lower-angle
transition, so the specified strict rep thresholds remain unchanged.
