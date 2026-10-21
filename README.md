# Intelligent Gym Exercise & Form Analysis System

Phase 1 provides camera-independent angle and squat logic plus a webcam/video
preview using classic MediaPipe Pose. It detects squat phases and repetitions,
shows knee angle and simple side-view coaching cues, and asks the user to move
into frame when required landmarks are unavailable. It is a coaching aid, not
a replacement for a qualified trainer. Landmark x/y values are converted to
image coordinates before angle and ratio calculations to account for the
camera frame's aspect ratio.

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
