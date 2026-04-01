# basketball-360-autoframe

Generate Insta360 Studio sidecar files to auto-reframe 360° basketball video using behavioral cloning.

Takes equirectangular video from an Insta360 X4 camera, predicts a camera trajectory using a neural network trained on your own manual camera work, and outputs an `.insprj` sidecar file. Insta360 Studio renders the final video with full quality, audio, and stabilization.

## How it works

**You are the training data.** When you manually pan through a 360° video in the Insta360 app, your pan/tilt/zoom decisions are saved in an `.insprj` sidecar file. This project trains a neural network to replicate those decisions on new footage.

1. **Parse** `.insprj` sidecar files to extract per-frame camera trajectories
2. **Train** a CNN+GRU model to predict camera parameters from equirectangular frames
3. **Infer** on new 360° video to produce a predicted camera trajectory
4. **Write** an `.insprj` sidecar file that Insta360 Studio can render

The AI does not render video. Insta360 Studio handles that — with full quality, audio passthrough, lens correction, and stabilization already solved.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of the approach, model design, and rationale.

## Prerequisites

- Python 3.10+
- Insta360 Studio (for rendering the final video from generated sidecars)
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

### 3. Generate sidecar for new footage

```bash
autoframe reframe new_game.mp4 runs/<run>/best.pt
```

This outputs `new_game.insprj` alongside the source video.

### 4. Render in Insta360 Studio

Open Insta360 Studio, import the source video, load the generated `.insprj` sidecar, and export. Upload to YouTube.

## CLI Reference

```bash
# Train the model
autoframe train DATA_DIR [--epochs 100] [--batch-size 16] [--lr 1e-4]
                         [--backbone resnet18] [--seq-len 60] [--sample-fps 5.0]

# Generate an .insprj sidecar using a trained model
autoframe reframe INPUT_VIDEO MODEL_CHECKPOINT [-o output.insprj]
                  [--smooth 15] [--keyframe-interval 200]

# Show video info
autoframe info VIDEO_FILE

# Inspect a sidecar file
autoframe parse-sidecar FILE.insprj
```

## Project structure

```
src/autoframe/
    cli.py              - Typer CLI (train, reframe, info, parse-sidecar)
    config.py           - Dataclass configuration
    insta360_parser.py  - Read/write .insprj XML sidecar files
    dataset.py          - PyTorch Dataset (frame sequences + camera labels)
    model.py            - CameraPredictor (ResNet-18 + GRU -> yaw/pitch/fov)
    train.py            - Training loop with weighted Huber loss
    camera.py           - Model-based inference controller
    pipeline.py         - Single-pass inference: predict -> smooth -> write .insprj
    projection.py       - Equirectangular math (training preprocessing only)
    video_io.py         - Video frame reading (OpenCV)
```

## How much training data do I need?

5-10 manually-reframed games is a good starting point. The model subsamples to 5fps during training and uses a pretrained ImageNet backbone, so it converges with relatively modest data.

## Next steps

Immediate execution plan (see also `VALIDATION_SPRINT_PLAN.md`):

- [ ] Run Studio sidecar compatibility matrix across target footage + Studio versions
- [ ] Log each import/export round-trip result in `artifacts/validation/compatibility-matrix.csv`
- [ ] Complete Media SDK fallback spike (`insv` -> stitched `mp4`) for one reference clip
- [ ] Decide go/no-go using documented thresholds in `VALIDATION_SPRINT_PLAN.md`
- [ ] If go: continue model iteration against validated sidecar schema
- [ ] If no-go: pivot to SDK-backed render pipeline

Validation helper commands:

```bash
python scripts/validation_matrix.py generate --camera-models X4 --studio-versions 5.4.4 5.3.2 --clips game01_q1 game01_q2
python scripts/validation_matrix.py summarize
```
