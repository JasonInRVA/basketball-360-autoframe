# Architecture: Behavioral Cloning for 360° Basketball Video Reframing

## The Problem

You have a stationary Insta360 X4 camera recording basketball games in full 360°
equirectangular format. You want to produce a standard 16:9 video that follows the
game action — the kind of smoothly-panned footage you'd get from a human camera
operator — suitable for uploading to YouTube.

## Why Not Object Detection (YOLO)?

The first version of this project used YOLO to detect players and the ball, then
applied heuristics to decide where to point a virtual camera. This approach fails
for several reasons:

1. **YOLO is trained on rectilinear images.** Equirectangular frames have severe
   distortion, especially away from the equator. Even tiling the sphere into
   perspective patches introduces artifacts at seams and splits objects across tiles.

2. **Ball detection is unreliable.** A basketball is small, fast, and often occluded.
   COCO's "sports ball" class doesn't have enough basketball-specific training data
   for reliable detection in game footage.

3. **Detection doesn't solve framing.** Knowing *where objects are* is different from
   knowing *where to point the camera*. A good camera operator anticipates plays,
   leads the action, adjusts zoom based on context, and makes smooth transitions.
   These are cinematographic decisions that heuristics can't capture.

4. **It's solving the wrong problem.** We don't need to understand the scene — we
   need to replicate a human's camera control decisions.

## The Approach: Behavioral Cloning

We have something much more valuable than object labels: **expert demonstrations.**

When you manually pan through a 360° video in the Insta360 app and export it, the
app creates an `.insprj` sidecar file containing the exact pan/tilt/roll/FOV
trajectory you applied, timestamped in milliseconds. This is a complete record of
every camera decision the human operator made.

Paired with the source equirectangular footage, this gives us supervised training
data: for each frame, we know both what the world looked like and what the human
decided to do.

**Behavioral cloning** is the technique of training a model to replicate expert
behavior from demonstrations. It's the same approach used in autonomous driving
(learn steering angles from human driving), robotic manipulation (learn actions
from teleoperation), and game-playing AI (learn moves from expert replays).

### Training Data Format

```
data/training/
    game_2024_01_15.mp4          <- Raw equirectangular video
    game_2024_01_15.insprj       <- Sidecar with keyframed pan/tilt/fov trajectory
    game_2024_02_03.mp4
    game_2024_02_03.insprj
    ...
```

The `.insprj` file is XML from Insta360 Studio:
```xml
<keyframe time="15845" pan="0" tilt="0" roll="0" fov="1.309" distance="0.6"/>
<keyframe time="30000" pan="45.5" tilt="-10" roll="0" fov="1.1" distance="0.5"/>
```

These sparse keyframes are interpolated to produce dense per-frame labels.

## Key Design Decision: Output .insprj, Not Video

A critical simplification: **the AI does not render video.** It outputs an `.insprj`
sidecar file, and Insta360 Studio handles the actual rendering.

This means we don't need to implement:
- Equirectangular-to-rectilinear projection for output
- Video encoding (H.264, audio muxing, color space handling)
- Audio passthrough
- Roll/distance parameter interpretation
- Lens correction, stabilization, or any Insta360-specific processing

The AI's job is narrow and well-defined: **predict a camera trajectory and write it
as keyframes in a format Insta360 Studio understands.** Studio does the rest with
full quality, audio, and all the edge cases already solved.

### Why This Matters

Every line of rendering code we don't write is a line that can't have bugs. Insta360
has spent years perfecting their renderer. We leverage that instead of reimplementing
it poorly.

## Model Architecture

```
Equirectangular frame (640x320, RGB)
  |
  v
+----------------------------------+
|  ResNet-18 (pretrained ImageNet) |   CNN backbone -- extracts spatial features.
|  Last FC removed -> 512-dim     |   Early layers frozen (edges/textures transfer
|  Early 75% frozen               |   from ImageNet). Only last block fine-tuned.
+----------------+-----------------+
                 | 512-dim feature vector per frame
                 v
+----------------------------------+
|  2-layer GRU (hidden_size=256)   |   Temporal model -- learns smooth camera motion,
|  Batch-first, dropout=0.1       |   anticipation, and context from recent frames.
+----------------+-----------------+
                 | 256-dim hidden state per frame
                 v
+----------------------------------+
|  Linear(256 -> 64) -> ReLU      |   Prediction head -- outputs 4 values per frame.
|  Linear(64 -> 4)                |
+----------------+-----------------+
                 |
                 v
      (sin_yaw, cos_yaw, pitch, fov)
```

### Why sin/cos encoding for yaw?

Yaw (pan) is circular: 179 degrees and -179 degrees are 2 degrees apart, but
naively they look 358 degrees apart. If we predicted raw yaw degrees, the model
would see a huge loss spike at the +/-180 degree boundary and produce erratic
behavior there.

By encoding yaw as `(sin(yaw), cos(yaw))`, the representation is continuous
everywhere on the circle. At inference time, we recover the angle with `atan2`.

### Why GRU over LSTM?

Slightly fewer parameters, trains faster, and performs comparably for this task.
The temporal context helps the model:
- Produce smooth camera motion (not jittery frame-to-frame)
- Anticipate action (e.g., camera starts panning before the ball arrives)
- Maintain consistent framing during momentary occlusions or ambiguous frames

## Training Details

| Parameter | Value | Rationale |
|---|---|---|
| Loss | Smooth-L1 (Huber) | Robust to outliers, better than MSE for regression |
| Yaw weight | 2x | Basketball action is primarily horizontal (side-to-side) |
| Optimizer | AdamW | Standard for fine-tuning pretrained models |
| LR schedule | Cosine annealing | Smooth decay, avoids sharp LR drops |
| Grad clipping | max_norm=1.0 | Prevents gradient explosions in GRU |
| Sequence length | 60 frames | ~12 seconds at 5fps -- enough temporal context |
| Sample FPS | 5.0 | Camera motion is smooth; 30fps is redundant for labels |
| Batch size | 16 | 16 sequences of 60 frames each |
| Augmentation | Horizontal flip | Mirrors the court, negates sin(yaw), doubles effective data |

### Data augmentation: horizontal flip

Flipping the equirectangular frame horizontally is equivalent to mirroring the
court. When we flip, we negate `sin(yaw)` but keep `cos(yaw)` -- this correctly
mirrors the angular target. This effectively doubles the training data and prevents
the model from developing a left/right bias.

## Inference Pipeline

```
+-----------------------+
| Predict trajectory    |  Read equirectangular frames -> resize to 640x320
|                       |  -> run through trained model -> collect raw
|                       |  (yaw, pitch, fov) predictions per frame
+-----------+-----------+
            v
+-----------------------+
| Smooth trajectory     |  Post-hoc uniform filter on sin/cos(yaw), pitch, fov.
|                       |  Removes any residual jitter the model produces.
+-----------+-----------+
            v
+-----------------------+
| Write .insprj         |  Convert dense per-frame predictions to sparse
|                       |  keyframes (every 200ms). Write XML sidecar file
|                       |  compatible with Insta360 Studio.
+-----------+-----------+
            v
+-----------------------+
| Insta360 Studio       |  User opens Studio, loads source video + sidecar,
| (external)            |  and exports the final reframed video with full
|                       |  quality, audio, stabilization, lens correction.
+-----------------------+
```

This is a single-pass pipeline. No video rendering, no FFmpeg, no audio handling.
The AI predicts, we write XML, Studio renders.

## Preparing Training Data

### Step 1: Record games
Mount the Insta360 X4 on a tripod at the court. Record full games.

### Step 2: Create the camera trajectory
In the Insta360 app (mobile) or Insta360 Studio (desktop), manually reframe the
video -- pan, tilt, and zoom to follow the game as you'd want the final video to
look. Export the reframed video to YouTube as you normally would.

### Step 3: Extract the sidecar
Open the same project in **Insta360 Studio** (desktop). The `.insprj` XML file
is saved alongside the project. This contains all your keyframe data.

Verify with: `autoframe parse-sidecar path/to/project.insprj`

### Step 4: Organize
Place each source video and its `.insprj` sidecar in `data/training/` with
matching filenames.

### Step 5: Train
```bash
autoframe train data/training/ --epochs 100
```

### Step 6: Generate sidecar for new footage
```bash
autoframe reframe new_game.mp4 runs/<run>/best.pt
```

### Step 7: Render in Insta360 Studio
Open Studio, import the source video, load the generated `.insprj`, and export.

## What This Architecture Cannot Do (Known Limitations)

1. **Generalization across venues.** If trained on games from one gym, the model
   learns that gym's visual appearance. It may not transfer well to a different
   court/lighting. Solution: train on diverse venues, or fine-tune on new ones.

2. **Unusual events.** The model replicates what it's seen. If the training data
   never includes a specific scenario (e.g., all players running to one corner),
   the model's behavior there is unpredictable.

3. **No semantic understanding.** The model doesn't "know" it's watching basketball.
   It learns a visual-motor mapping: "when the scene looks like this, point the
   camera there." This is a strength (simple, fast) and a limitation (no reasoning).

4. **Drift on very long videos.** The GRU hidden state may accumulate errors over
   very long sequences (>30 min). The post-hoc smoothing mitigates this, and
   the model can be reset at natural breaks (halftime, timeouts).

5. **Dependent on Insta360 Studio.** The output is an `.insprj` file, not a
   standalone video. You need Insta360 Studio to produce the final output. This
   is a deliberate tradeoff: we leverage their polished renderer instead of
   building our own.

6. **Sidecar format is not officially documented.** Our `.insprj` writer is based
   on reverse-engineered format documentation. If Insta360 changes the format in
   a future Studio update, the writer may need updating.

## Key References

- **Deep 360 Pilot** (Hu et al., CVPR 2017) -- Pioneered RL-based virtual camera
  control for 360 sports video. Our approach uses supervised behavioral cloning
  instead of RL, which is simpler and leverages the available expert demonstrations.
  Reference implementation: `yenchenlin/Deep360Pilot-CVPR17`

- **Pano2Vid** (Su et al., 2016) -- Automatic cinematography from 360 video using
  candidate viewing directions.

- **Insta360 .insprj format** -- Documented at `insta.pk360.de/studio202x_insprj/`.
  XML format with keyframe pan/tilt/roll/fov and easing curves.
