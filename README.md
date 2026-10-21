# Intelligent Gym Exercise & Form Analysis System

The application uses classic MediaPipe Pose to analyze webcam or video input.
Phases 1 and 2 provide camera-independent angle, squat, push-up, curl, and
plank logic. The live view shows movement phase, repetitions or plank hold
time, alignment when available, and side-view coaching cues. It asks the user
to move into frame when required landmarks are unavailable. It is a coaching
aid, not a replacement for a qualified trainer. Landmark x/y values are
converted to image coordinates before angle and ratio calculations to account
for the camera frame's aspect ratio.

## Setup

Use Python 3.11 and install the pinned dependencies from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The command installs the project requirements into `.venv`, not globally.
Confirm that `.venv` uses Python 3.11 before installing; the workspace
environment observed during Phase 1 uses Python 3.12.10.

## Run

```powershell
.\.venv\Scripts\python.exe main.py --source 0 --exercise squat
```

For a video file, pass its path instead of camera index `0`:

```powershell
.\.venv\Scripts\python.exe main.py --source .\data\raw\squat.mp4 --exercise squat
```

Allow camera access if prompted. Press **Q** in the preview to quit. Camera
access and real-time performance have not been verified in this environment.

Select the supported Phase 2 analyzer by replacing `squat` with `pushup`,
`curl`, or `plank`.
# gym-form-analyzer
