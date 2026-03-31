# basketball-360-autoframe

Auto-reframe Insta360 X4 360-degree basketball video into standard rectangular video for YouTube.

Takes equirectangular video from a 360 camera, detects the basketball action using YOLO, and outputs a smoothly-panned 1080p video that follows the game.

## How it works

1. **Tiled detection** — Extracts overlapping perspective tiles from the equirectangular frame, runs YOLO on each tile, then merges detections with NMS
2. **Virtual camera** — Computes a target yaw/pitch/FOV based on player and ball positions (weighted toward the ball)
3. **Trajectory smoothing** — Two-stage smoothing: per-frame exponential smoothing + post-hoc uniform filter to eliminate jitter
4. **Rectilinear extraction** — Reprojects the equirectangular frame to a standard perspective crop at the computed camera position

## Prerequisites

- Python 3.10+
- FFmpeg (for video output): `brew install ffmpeg`

## Install

```bash
git clone https://github.com/JasonInRVA/basketball-360-autoframe.git
cd basketball-360-autoframe
pip install -e ".[dev]"
```

## Usage

```bash
# Basic usage — reframe a 360 video
autoframe reframe path/to/360_video.mp4

# Specify output path and FOV
autoframe reframe input.mp4 -o output.mp4 --fov 80

# Adjust smoothing (higher = smoother camera, slower to react)
autoframe reframe input.mp4 --smoothing 0.9

# Use a custom YOLO model
autoframe reframe input.mp4 --model path/to/custom_model.pt

# Show live preview while processing
autoframe reframe input.mp4 --preview

# Check video info (resolution, fps, equirectangular detection)
autoframe info path/to/video.mp4
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--output, -o` | `<input>_reframed.mp4` | Output video path |
| `--model, -m` | `yolov8n.pt` | YOLO model (auto-downloads if not found) |
| `--fov` | `90` | Virtual camera field of view (degrees) |
| `--width, -w` | `1920` | Output width |
| `--height, -h` | `1080` | Output height |
| `--smoothing, -s` | `0.85` | Camera smoothing (0=instant, 0.99=frozen) |
| `--confidence, -c` | `0.4` | Detection confidence threshold |
| `--detect-every` | `3` | Run detection every N frames |
| `--preview` | off | Show live preview window |

## Project structure

```
src/autoframe/
    cli.py          — Typer CLI entry point
    pipeline.py     — Two-pass processing pipeline
    projection.py   — Equirectangular/rectilinear math
    detector.py     — Tiled YOLO detection on 360 frames
    camera.py       — Virtual camera controller + smoothing
    video_io.py     — Video read/write via OpenCV + FFmpeg
    config.py       — Dataclass configuration
```

## Next steps

- [ ] Test with real Insta360 X4 footage and tune defaults
- [ ] Add audio passthrough from source video
- [ ] Train or fine-tune YOLO on basketball-specific data for better ball detection
- [ ] Add court-line detection for smarter framing
- [ ] Support batch processing of multiple videos
