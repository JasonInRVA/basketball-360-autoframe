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

When you manually reframe a 360° video in Insta360 Studio, your pan/tilt/zoom
decisions are saved in a `.insproj` project file — a JSON-based directory structure
containing the exact camera trajectory as keyframes with all angles in **radians**
and all timestamps as **frame numbers**. This is a complete record of every camera
decision the human operator made.

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
    Game 2024-01-15/                             <- Insta360 Studio project directory
        <uuid>.insproj                           <- Main project file (JSON)
        project_medias.json
        footage_info.json
        rough_cut.json
        project_settings/
            <uuid>.json
    Game 2024-02-03/
        <uuid>.insproj
        ...
```

The `.insproj` file is JSON. Keyframes live inside the clip's `key_frame_track.node_list`:

```json
{
    "auto_fov": 0,
    "distance": 0.4,
    "fov": 1.19,
    "is_headtrack": 0,
    "name": "point0",
    "node_type": 0,
    "pan": 0.2096,
    "roll": 0,
    "src_time": 0,
    "state": 7,
    "tilt": -0.311,
    "time": 0
}
```

All angles are in **radians**. `time` and `src_time` are **frame numbers**.
These sparse keyframes are interpolated to produce dense per-frame labels for training.

See [INSTA360_FORMAT.md](INSTA360_FORMAT.md) for the full format specification.

## Key Design Decision: Output .insproj, Not Video

A critical simplification: **the AI does not render video.** It modifies an existing
Insta360 Studio project by replacing the keyframe track, and Studio handles rendering.

This means we don't need to implement:
- Equirectangular-to-rectilinear projection for output
- Video encoding (H.264, audio muxing, color space handling)
- Audio passthrough
- Roll/distance parameter interpretation
- Lens correction, stabilization, or any Insta360-specific processing

The AI's job is narrow and well-defined: **predict a camera trajectory and inject it
as keyframes into an existing Insta360 Studio project.** Studio does the rest with
full quality, audio, and all the edge cases already solved.

### Template-Based Approach

Rather than generating a project from scratch (which requires getting every UUID,
path reference, and setting correct), we modify an existing project:

1. User creates a project in Studio with the source video on the timeline
2. Our tool reads the `.insproj`, locates the clip's `key_frame_track.node_list`
3. We replace `node_list` with AI-generated keyframes (alternating keyframe and
   transition nodes)
4. We update `camera_transform` to match the first keyframe
5. We clear any deep track data (Studio's built-in tracking)
6. User reopens Studio and exports the final video

### Operational workflow (practical steps)

1. **Create a template project in Insta360 Studio** with the target clip on the timeline (no keyframes required). Studio writes a directory containing `<uuid>.insproj`, `project_medias.json`, etc.
2. **Run inference**: `autoframe reframe <video> <checkpoint> --project <path/to/<uuid>.insproj> [--smooth 15] [--keyframe-interval 6]`.  
   - Pipeline reads the `.insproj`, predicts pan/tilt/fov per frame (radians), subsamples to keyframes, writes them into `key_frame_track.node_list`, updates `camera_transform`, and saves (with backup).
3. **Re-open in Studio and export**. No rendering is done by our code.
4. **If Studio doesn’t list the project** after copying/moving: Studio tracks projects in `~/Library/Application Support/Insta360/Insta360 Studio/nle/pro.insproj`. Add an entry pointing `projectPath` to the project directory (id/name/fps from the `.insproj`), then relaunch Studio to re-index.

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
          all values in radians
```

### Why sin/cos encoding for yaw?

Yaw (pan) is circular: 179° and -179° are 2° apart, but naively they look 358°
apart. If we predicted raw yaw, the model would see a huge loss spike at the ±180°
boundary and produce erratic behavior there.

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
|                       |  (pan, tilt, fov) predictions per frame (radians)
+-----------+-----------+
            v
+-----------------------+
| Smooth trajectory     |  Post-hoc uniform filter on sin/cos(pan), tilt, fov.
|                       |  Circular smoothing avoids discontinuity at ±pi.
+-----------+-----------+
            v
+-----------------------+
| Inject keyframes      |  Convert dense per-frame predictions to sparse
|                       |  keyframes (every 6 frames = 5/sec at 30fps).
|                       |  Inject into existing .insproj project file (JSON).
+-----------+-----------+
            v
+-----------------------+
| Insta360 Studio       |  User opens Studio, loads the modified project,
| (external)            |  and exports the final reframed video with full
|                       |  quality, audio, stabilization, lens correction.
+-----------------------+
```

This is a single-pass pipeline. No video rendering, no FFmpeg, no audio handling.
The AI predicts, we inject keyframes into the project JSON, Studio renders.

## Preparing Training Data

### Step 1: Record games
Mount the Insta360 X4 on a tripod at the court. Record full games.

### Step 2: Create the camera trajectory
In Insta360 Studio (desktop), import the 360° video and manually reframe it —
pan, tilt, and zoom to follow the game as you'd want the final video to look.
Save the project.

### Step 3: Organize training data
Each Insta360 Studio project is a directory containing a `.insproj` file with
your keyframe data. Place these project directories in `data/training/`.

Verify your projects parse correctly:
```bash
autoframe inspect-project "data/training/Game 2024-01-15/"
```

### Step 4: Train
```bash
autoframe train data/training/ --epochs 100
```

### Step 5: Generate keyframes for new footage
First, create a project in Insta360 Studio with the new video on the timeline
(no keyframes needed). Then:
```bash
autoframe reframe new_game.mp4 runs/<run>/best.pt --project "path/to/project.insproj"
```

### Step 6: Render in Insta360 Studio
Open Studio, load the modified project, and export. Upload to YouTube.

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

5. **Dependent on Insta360 Studio.** The output is a modified `.insproj` project,
   not a standalone video. You need Insta360 Studio to produce the final output.
   This is a deliberate tradeoff: we leverage their polished renderer instead of
   building our own.

6. **Project format is not officially documented.** Our `.insproj` reader/writer
   is based on reverse-engineered format observation from real Studio projects.
   If Insta360 changes the format in a future Studio update, the code may need
   updating. See [INSTA360_FORMAT.md](INSTA360_FORMAT.md) for our format docs.

## Key References

- **Deep 360 Pilot** (Hu et al., CVPR 2017) -- Pioneered RL-based virtual camera
  control for 360 sports video. Our approach uses supervised behavioral cloning
  instead of RL, which is simpler and leverages the available expert demonstrations.
  Reference implementation: `yenchenlin/Deep360Pilot-CVPR17`

- **Pano2Vid** (Su et al., 2016) -- Automatic cinematography from 360 video using
  candidate viewing directions.

- **Insta360 Studio project format** -- Documented in [INSTA360_FORMAT.md](INSTA360_FORMAT.md),
  reverse-engineered from real project files. JSON-based directory structure with
  keyframes using radians and frame numbers.
