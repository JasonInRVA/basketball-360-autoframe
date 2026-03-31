# basketball-360-autoframe

Auto-reframe Insta360 X4 360-degree basketball video into standard rectangular video for YouTube.

Takes equirectangular video from a 360 camera and produces a smoothly-panned 1080p video that follows the game action — learned from your own manual camera work using behavioral cloning.

## How it works

**You are the training data.** When you manually pan through a 360 video in the Insta360 app and export it, the app saves your pan/tilt/zoom decisions in an `.insprj` sidecar file. This project trains a neural network to replicate those decisions automatically on new footage.

1. **Parse** `.insprj` sidecar files to extract per-frame camera trajectories
2. **Train** a CNN+GRU model to predict camera parameters from equirectangular frames
3. **Infer** on new 360 video to produce a predicted camera trajectory
4. **Render** by extracting smoothed rectilinear crops from the equirectangular source

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of the approach, model design, and rationale.

## Prerequisites

- Python 3.10+
- FFmpeg: `brew install ffmpeg`
- A GPU is recommended for training (CUDA or Apple MPS). CPU works but is slow.

## Install

```bash
git clone https://github.com/JasonInRVA/basketball-360-autoframe.git
cd basketball-360-autoframe
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare training data

Place your Insta360 source videos and their `.insprj` sidecar files (from Insta360 Studio) in the same directory:

```
data/training/
    game_01.mp4
    game_01.insprj
    game_02.mp4
    game_02.insprj
```

Verify your sidecars parse correctly:
```bash
autoframe parse-sidecar data/training/game_01.insprj
```

### 2. Train the model

```bash
autoframe train data/training/
```

This will:
- Pair each video with its sidecar trajectory
- Train a ResNet-18 + GRU model via behavioral cloning
- Save checkpoints to `runs/<timestamp>/`
- Report per-epoch loss and yaw error in degrees

### 3. Reframe new footage

```bash
autoframe reframe new_game.mp4 runs/<run>/best.pt
```

### 4. Upload to YouTube

The output is a standard 1920x1080 H.264 MP4 — ready to upload.

## CLI Reference

```bash
# Train the model
autoframe train DATA_DIR [--epochs 100] [--batch-size 16] [--lr 1e-4]
                         [--backbone resnet18] [--seq-len 60] [--sample-fps 5.0]

# Reframe a video using a trained model
autoframe reframe INPUT_VIDEO MODEL_CHECKPOINT [-o output.mp4]
                  [--width 1920] [--height 1080] [--smooth 15] [--preview]

# Show video info
autoframe info VIDEO_FILE

# Inspect a sidecar file
autoframe parse-sidecar FILE.insprj
```

## Project structure

```
src/autoframe/
    cli.py              — Typer CLI (train, reframe, info, parse-sidecar)
    config.py           — Dataclass configuration
    insta360_parser.py  — Parse .insprj XML sidecar files
    dataset.py          — PyTorch Dataset (frame sequences + camera labels)
    model.py            — CameraPredictor (ResNet-18 + GRU → yaw/pitch/fov)
    train.py            — Training loop with weighted Huber loss
    camera.py           — Model-based inference camera controller
    pipeline.py         — Two-pass inference pipeline (predict → render)
    projection.py       — Equirectangular ↔ rectilinear projection math
    video_io.py         — Video read (OpenCV) / write (FFmpeg)
```

## How much training data do I need?

5-10 manually-reframed games is a good starting point. At 30fps, that's 500K-1M frames. The model subsamples to 5fps during training (camera motion is smooth) and uses a pretrained ImageNet backbone, so it converges with relatively modest data.

All games from the same venue is fine for V1 — the model may not generalize to other courts without additional training data.

## Next steps

- [ ] Test with real Insta360 X4 footage and validate .insprj parsing
- [ ] Train initial model and evaluate yaw accuracy
- [ ] Add audio passthrough from source video
- [ ] Experiment with temporal augmentation (varying playback speed)
- [ ] Support for .insdata sidecar format (from mobile app)
