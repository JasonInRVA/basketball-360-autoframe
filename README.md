# basketball-360-autoframe

Generate AI-predicted camera trajectories for Insta360 Studio projects to auto-reframe 360° basketball video using behavioral cloning.

Takes equirectangular video from an Insta360 X4 camera, predicts a camera trajectory using a neural network trained on your own manual camera work, and injects the keyframes into an existing Insta360 Studio project. Studio renders the final video with full quality, audio, and stabilization.

## How it works

**You are the training data.** When you manually pan through a 360° video in Insta360 Studio, your pan/tilt/zoom decisions are saved as keyframes in the `.insproj` project file. This project trains a neural network to replicate those decisions on new footage.

1. **Parse** `.insproj` project files to extract per-frame camera trajectories (radians, frame numbers)
2. **Train** a CNN+GRU model to predict camera parameters from equirectangular frames
3. **Infer** on new 360° video to produce a predicted camera trajectory
4. **Inject** AI-generated keyframes back into an Insta360 Studio project for rendering

The AI does not render video. Insta360 Studio handles that — with full quality, audio passthrough, lens correction, and stabilization already solved.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of the approach, model design, and rationale.
See [INSTA360_FORMAT.md](INSTA360_FORMAT.md) for the reverse-engineered Insta360 Studio project format specification.

## Prerequisites

- Python 3.10+
- Insta360 Studio (for creating training projects and rendering final output)
- A GPU is recommended for training (CUDA or Apple MPS). CPU works but is slow.

## Install

```bash
git clone https://github.com/JasonInRVA/basketball-360-autoframe.git
cd basketball-360-autoframe
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare training data

Create Insta360 Studio projects for your recorded games. For each game, import
the 360° video into Studio, manually reframe it (pan, tilt, zoom to follow the
action), and save the project. Each project directory contains a `.insproj` file
with your keyframe trajectory.

Place all project directories under a training data folder:

```
data/training/
    Game 2024-01-15/
        <uuid>.insproj
        project_medias.json
        ...
    Game 2024-02-03/
        <uuid>.insproj
        ...
```

Verify your projects parse correctly:
```bash
autoframe inspect-project "data/training/Game 2024-01-15/"
```

### 2. Train the model

```bash
autoframe train data/training/
```

### 3. Generate keyframes for new footage

First, create an Insta360 Studio project with the new video on the timeline (no
keyframes needed — this is just a template). Then run:

```bash
autoframe reframe new_game.mp4 runs/<run>/best.pt --project "path/to/project.insproj"
```

This injects AI-predicted keyframes into the project file (a backup is created automatically).

### 4. Render in Insta360 Studio

Open Studio, load the modified project, preview the AI-generated camera work, and export. Upload to YouTube.

### Manual edit walkthrough (for sanity checks)
1. Make a copy of a Studio project directory (contains `*.insproj`, `project_medias.json`, etc.).
2. Edit the `*.insproj` JSON: append keyframes under `tracks[1].clips[0].key_frame_track.node_list`, alternating keyframe (`node_type: 0`) and transition (`node_type: 1`) nodes. Update `camera_transform` to match the first keyframe.
3. Set `lastModifiedTime` to the current timestamp (format `YYYY.MM.DD HH:MM:SS`).
4. Re-open the project in Studio; it will read the updated keyframes and render accordingly.
   - If Studio doesn’t list the project after copying, it may need a registry refresh: `~/Library/Application Support/Insta360/Insta360 Studio/nle/pro.insproj` holds the list Studio shows. Add an entry pointing `projectPath` to your copied directory, then relaunch Studio.

## CLI Reference

```bash
# Train the model
autoframe train DATA_DIR [--epochs 100] [--batch-size 16] [--lr 1e-4]
                         [--backbone resnet18] [--seq-len 60] [--sample-fps 5.0]

# Inject AI-predicted keyframes into an Insta360 Studio project
autoframe reframe INPUT_VIDEO MODEL_CHECKPOINT --project PROJECT.insproj
                  [-o output.insproj] [--smooth 15] [--keyframe-interval 6]

# Show video info
autoframe info VIDEO_FILE

# Inspect an Insta360 Studio project (show keyframes, clip info)
autoframe inspect-project PROJECT_PATH
```

## Project structure

```
src/autoframe/
    cli.py              - Typer CLI (train, reframe, info, inspect-project)
    config.py           - Dataclass configuration
    insta360_parser.py  - Read/write Insta360 Studio .insproj project files (JSON)
    dataset.py          - PyTorch Dataset (frame sequences + camera labels)
    model.py            - CameraPredictor (ResNet-18 + GRU -> sin_yaw/cos_yaw/pitch/fov)
    train.py            - Training loop with weighted Huber loss
    camera.py           - Model-based inference controller
    pipeline.py         - Single-pass inference: predict -> smooth -> inject keyframes
    projection.py       - Equirectangular math (training preprocessing only)
    video_io.py         - Video frame reading (OpenCV)

INSTA360_FORMAT.md      - Reverse-engineered Insta360 Studio project format docs
```

## How much training data do I need?

5-10 manually-reframed games is a good starting point. The model subsamples to 5fps during training and uses a pretrained ImageNet backbone, so it converges with relatively modest data.

## Next steps

- [ ] Verify modified `.insproj` files load cleanly in Insta360 Studio
- [ ] Run Studio compatibility matrix across target footage + Studio versions and log results in `artifacts/validation/compatibility-matrix.csv`
- [ ] Complete Media SDK fallback spike (`insv` -> stitched `mp4`) for one reference clip
- [ ] Train initial model and evaluate yaw accuracy
- [ ] Experiment with temporal augmentation (vary playback speed)
- [ ] Decide go/no-go using thresholds in `VALIDATION_SPRINT_PLAN.md`; if go, continue model iteration against validated schema; if no-go, pivot to SDK-backed render pipeline
- [ ] Investigate Insta360 Desktop-MediaSDK-Cpp for programmatic rendering

Validation helper commands:

```bash
python scripts/validation_matrix.py generate --camera-models X4 --studio-versions 5.4.4 5.3.2 --clips game01_q1 game01_q2
python scripts/validation_matrix.py summarize
```
