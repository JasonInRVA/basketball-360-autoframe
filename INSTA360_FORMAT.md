# Insta360 Studio Project Format Reference

This document describes the project file format used by **Insta360 Studio**
(macOS/Windows desktop app) as reverse-engineered from a real project created
with Studio version circa early 2026. This is NOT the `.insprj` XML format
described in older third-party documentation — Studio now uses a JSON-based
project directory structure.

> **Important:** This format is undocumented by Insta360 and may change in
> future Studio updates. All field names and value ranges below are based on
> empirical observation of a single test project. When in doubt, create a
> reference project in Studio and compare.

---

## Project Directory Structure

An Insta360 Studio project is a **directory** (not a single file) containing:

```
<Project Name>/
    <project-uuid>.insproj              Main project file (JSON)
    <project-uuid>_backup.insproj       Auto-backup of main project file
    project_medias.json                 Source media references
    footage_info.json                   Per-clip camera state snapshot
    rough_cut.json                      Trim/clip selection data
    cloud_rough_cut.json                Cloud sync state (usually empty)
    project_settings/
        <media-settings-uuid>.json      Default settings per imported media
```

All files are JSON. The project UUID appears in the main filename and inside
the project file as the `id` field.

---

## Main Project File: `<uuid>.insproj`

This is the primary file. It contains the project metadata, track structure,
clip settings, and — critically — the **keyframe track** that defines the
camera trajectory.

### Top-Level Fields

```json
{
    "aspectRatio": "16:9",
    "autoFps": true,
    "creationTime": "2026.04.02 12:11:42",
    "definitionEdited": false,
    "definitionTypeByUser": 0,
    "enableEffects": true,
    "fps": 30,
    "frames": 450,
    "id": "<project-uuid>",
    "isPanoramic": false,
    "lastModifiedTime": "2026.04.02 12:15:42",
    "mainMagnetizable": true,
    "markTrimEnd": 450,
    "markTrimIn": 0,
    "markTrimVisible": false,
    "name": "Test Project",
    "projectPath": "/path/to/project/directory",
    "tracks": [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `aspectRatio` | string | Output aspect ratio ("16:9") |
| `fps` | int | Output frame rate |
| `frames` | int | Total output frames (after trimming) |
| `id` | string | UUID matching the filename |
| `isPanoramic` | bool | `false` for reframed output, `true` for 360 export |
| `name` | string | Project display name |
| `projectPath` | string | Absolute path to the project directory |
| `tracks` | array | Timeline tracks (see below) |
| `markTrimIn` / `markTrimEnd` | int | Mark in/out points (frame numbers) |

### Track Structure

The `tracks` array contains 3 tracks in a fixed order:

| Index | `type` | Purpose |
|---|---|---|
| 0 | 3 | **Title/overlay track** (usually empty) |
| 1 | 0 | **Main video track** — contains clips with keyframes |
| 2 | 2 | **Audio/music track** (usually empty) |

The main video track (`tracks[1]`) has a `clips` array. Each clip represents
one source video on the timeline.

---

## Clip Object (inside `tracks[1].clips[]`)

Each clip is a large JSON object containing all settings for one piece of
source footage. The fields most relevant to our project:

### Source Reference

```json
{
    "name": "VID_012.insv",
    "url": "/Volumes/Untitled/DCIM/Camera01/VID_20260228_100654_00_012.insvVID_012.insv",
    "id": "<clip-uuid>",
    "sourceFrames": 18263,
    "source_total_frames": 18263
}
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name of the source file |
| `url` | string | Full path to the source `.insv` file (note: concatenated with display name) |
| `id` | string | UUID for this clip |
| `sourceFrames` | int | Total frames in the source video |
| `source_total_frames` | int | Same as sourceFrames |

### Trim Points

```json
{
    "leftTrim": 0,
    "rightTrim": 17813,
    "original_left_trim": 0,
    "original_right_trim": 17813,
    "source_left_trim": 0,
    "source_right_trim": 17813
}
```

These are **frame numbers** in the source video. `leftTrim` is the in-point,
`rightTrim` is the out-point. The clip plays from `leftTrim` to `rightTrim`.

### Camera Transform (Default/Current View)

The static camera position when no keyframes are active:

```json
{
    "camera_transform": {
        "distance": 0.4007705748081207,
        "eulers": {
            "pitch": { "type": 0, "value": -0.32462576031684875 },
            "roll":  { "type": 0, "value": 0 },
            "yaw":   { "type": 0, "value": 0.0646330714225769 }
        },
        "fov": { "type": 0, "value": 1.1827789545059204 }
    }
}
```

| Field | Type | Unit | Description |
|---|---|---|---|
| `eulers.yaw.value` | float | **radians** | Horizontal pan. 0 = forward. |
| `eulers.pitch.value` | float | **radians** | Vertical tilt. Negative = looking down. |
| `eulers.roll.value` | float | **radians** | Roll. 0 = level. |
| `fov.value` | float | **radians** | Field of view. ~1.18 rad ≈ 67.7°. |
| `distance` | float | unitless | Zoom/perspective parameter (0-1 range observed). Lower = more zoomed in. |
| `*.type` | int | — | Observed as 0 (constant). Purpose unclear, possibly animation type. The `original_camera_transform` has `fov.type: 1` with `fov.value: 75` (degrees?), suggesting `type` may switch between radians and degrees. |

> **Critical observation:** `fov.type: 0` appears to mean the value is in
> radians. `fov.type: 1` with a value of `75` in `original_camera_transform`
> suggests that type might indicate the unit system. This needs further
> testing to confirm.

### Keyframe Track (THE IMPORTANT PART)

This is where the camera trajectory lives. Located at
`clip.key_frame_track.node_list`:

```json
{
    "key_frame_track": {
        "node_list": [
            { ... keyframe node ... },
            { ... transition node ... },
            { ... keyframe node ... },
            ...
        ]
    }
}
```

The `node_list` alternates between **keyframe nodes** (`node_type: 0`) and
**transition nodes** (`node_type: 1`). The pattern is:

```
keyframe0 → transition0-1 → keyframe1 → transition1-2 → keyframe2 → ...
```

#### Keyframe Node (`node_type: 0`)

```json
{
    "auto_fov": 0,
    "distance": 0.4007705748081207,
    "fov": 1.1926275254160816,
    "is_headtrack": 0,
    "name": "point0",
    "node_type": 0,
    "pan": 0.20957749527329927,
    "roll": 0,
    "src_time": 0,
    "state": 7,
    "tilt": -0.31096054257368516,
    "time": 0
}
```

| Field | Type | Unit | Description |
|---|---|---|---|
| `pan` | float | **radians** | Yaw / horizontal angle |
| `tilt` | float | **radians** | Pitch / vertical angle. Negative = looking down. |
| `roll` | float | **radians** | Roll angle |
| `fov` | float | **radians** | Field of view at this keyframe |
| `distance` | float | unitless | Zoom/perspective (0-1 range) |
| `time` | int | **frame number** | When this keyframe occurs on the timeline |
| `src_time` | int | **frame number** | Corresponding frame in the source video |
| `name` | string | — | Identifier like "point0", "point1", or deep track area name |
| `node_type` | int | — | Always `0` for keyframes |
| `state` | int | — | Observed values: `3` (deep track) and `7` (manual keyframe). May indicate keyframe origin/type. |
| `is_headtrack` | int | — | `0` = not from head tracking, presumably `1` if from head tracking |
| `auto_fov` | int | — | `0` = manual FOV. Presumably `1` if FOV was auto-calculated. |

**Angle ranges observed:**
- `pan`: -0.52 to +0.21 radians (roughly -30° to +12°) — but this is just the test clip
- `tilt`: -0.31 to -0.37 radians (roughly -18° to -21°) — looking slightly down
- `fov`: 0.77 to 1.19 radians (roughly 44° to 68°)

#### Transition Node (`node_type: 1`)

```json
{
    "name": "point0-point1",
    "node_type": 1,
    "point1X": 0.5,
    "point1Y": 0.5,
    "point2X": 0.5,
    "point2Y": 0.5,
    "type": 1
}
```

| Field | Type | Description |
|---|---|---|
| `name` | string | `"<from_name>-<to_name>"` |
| `node_type` | int | Always `1` for transitions |
| `point1X`, `point1Y` | float | Bezier control point 1 (0-1 range) |
| `point2X`, `point2Y` | float | Bezier control point 2 (0-1 range) |
| `type` | int | Easing type. `1` = linear (both control points at 0.5, 0.5) |

**Bezier easing:** The control points define a cubic bezier curve for the
interpolation between keyframes. `(0.5, 0.5)` / `(0.5, 0.5)` produces
linear interpolation. Other values produce ease-in, ease-out, etc.

### Deep Track

Studio's built-in object tracking feature. When the user draws a tracking
box, Studio generates a deep track entry:

```json
{
    "deep_track": {
        "cache_path": "/path/to/cache/deeptrack_cache",
        "deep_track_list": [
            {
                "id": 1775146319315,
                "name": "deepTrackArea1775146202888",
                "startTime": 134,
                "endTime": 733,
                "originalStartTime": 134,
                "originalEndTime": 733,
                "src_start_time": 134,
                "src_end_time": 733,
                "fov": 0.7682389085149958,
                "distance": 0.4007705748081207,
                "pitch_offset": 0,
                "yaw_offset": 0,
                "roll_offset": 0,
                "type": 1,
                "flags": 0,
                "fake": false,
                "resourceId": "",
                "resourceName": ""
            }
        ],
        "original_clip_id": "dcf6d329-2969-40f9-a2bf-1e701a73902a"
    }
}
```

Deep track entries appear as keyframes in `key_frame_track.node_list` with
`state: 3` (vs `state: 7` for manual keyframes). The deep track defines
a time range where Studio's AI tracking takes over the camera movement.

### Other Clip Settings (not keyframe-related)

These exist in the clip object but are not related to camera trajectory:

- `dewarp_setting` — Lens dewarp/projection type
- `stab_setting` — Stabilization (enabled by default, directional lock)
- `stitching_setting` — Stitch calibration, camera accessory type
- `stitching_optimization` — AI stitching, optical flow, dynamic stitching
- `media_process` — Color adjustment, denoise, exposure, etc.
- `audio_setting` — Volume, denoise, fade in/out
- `play_rate_track` — Speed ramping / slow motion
- `aspect_ratio_info` — Supported aspect ratios for this clip
- `supplementFrameType` — Frame interpolation setting

---

## project_medias.json

Lists all imported media files:

```json
[
    {
        "default": "<settings-uuid>",
        "display_name": "VID_012.insv",
        "durations": 608.774833333,
        "icon_type": 24,
        "import_time": "1775146318899",
        "paths": [
            "/Volumes/Untitled/DCIM/Camera01/LRV_20260228_100654_01_012.lrv",
            "/Volumes/Untitled/DCIM/Camera01/VID_20260228_100654_00_012.insv"
        ],
        "settings": [ "<settings-uuid>" ],
        "uuid": "<source-path>VID_012.insv"
    }
]
```

| Field | Type | Description |
|---|---|---|
| `default` | string | UUID of the default settings file in `project_settings/` |
| `paths` | array | Full paths to source files (`.insv` + `.lrv` low-res proxy) |
| `durations` | float | Source video duration in seconds |
| `import_time` | string | Unix timestamp in milliseconds (as string) |
| `uuid` | string | Unique ID for this media (path + display name concatenated) |

---

## footage_info.json

Stores the current camera view state per media (what you see in the preview):

```json
{
    "<media-uuid>": {
        "camera_transform": {
            "distance": 0.4007705748081207,
            "eulers": {
                "pitch": { "type": 0, "value": -0.3246 },
                "roll":  { "type": 0, "value": 0 },
                "yaw":   { "type": 0, "value": 0.0646 }
            },
            "fov": { "type": 0, "value": 1.1827 }
        },
        "player_frame": 38
    }
}
```

This appears to be the last-viewed camera position (for restoring the preview
when reopening the project). The `player_frame` is the last playback position.

---

## rough_cut.json

Stores trim/selection state per media:

```json
{
    "<media-uuid>": {
        "Clip1": {
            "totalFrames": 18263,
            "trimEnd": 450,
            "trimStart": 0,
            "updateTime": 1775146421678
        },
        "default": {
            "totalFrames": 18263,
            "trimEnd": 18263,
            "trimStart": 0,
            "updateTime": 0
        }
    }
}
```

The `default` entry preserves the original (untrimmed) state. Named entries
like `Clip1` represent trimmed selections.

---

## project_settings/<uuid>.json

Default settings for each imported media file. Structure is very similar to
a clip object in the main `.insproj` — contains `camera_transform`,
`key_frame_track`, `stab_setting`, etc. This represents the Media Page
settings (before adding to the Project Page timeline).

The `key_frame_track` in this file is typically empty (`{}`) since keyframes
are usually added on the Project Page, not the Media Page.

---

## Strategy for Generating Projects

### Approach: Template-Based Modification

Rather than generating a project from scratch (which requires getting every
UUID, path reference, and setting correct), the recommended workflow is:

1. **User creates a project in Studio** with the source video imported and
   a short clip on the timeline (even with no keyframes)
2. **Our tool reads the `.insproj` file**, locates the clip's
   `key_frame_track.node_list`
3. **We replace `node_list`** with our AI-generated keyframes
4. **We update `camera_transform`** to match the first keyframe
5. **We save the modified `.insproj`** (Studio will read it on next open)
6. **User opens Studio**, sees the AI-generated keyframes, and exports

This avoids generating UUIDs, source paths, settings, and the rest of the
boilerplate — we only touch the keyframe data.

### Practical edit recipe (tested)

1. Copy or create a Studio project directory (contains `<uuid>.insproj` + JSON sidecars).
2. In the `.insproj`, edit `tracks[1].clips[0].key_frame_track.node_list`, appending keyframe (`node_type: 0`) and transition (`node_type: 1`) nodes. Keep the alternating pattern keyframe→transition→keyframe…
3. Update `camera_transform` to match the first keyframe and set `lastModifiedTime` to the current timestamp (`YYYY.MM.DD HH:MM:SS`).
4. Save with UTF-8 JSON formatting; Studio accepts human-readable indenting.
5. Relaunch Studio; the project will reflect the new keyframes. If Studio does not list the project after moving/copying, add it to `~/Library/Application Support/Insta360/Insta360 Studio/nle/pro.insproj` (Studio’s project registry) with `projectPath` pointing to the directory, then restart Studio.

### What We Need to Generate per Keyframe

For each keyframe in the AI's output trajectory:

```json
{
    "auto_fov": 0,
    "distance": <copy from original clip>,
    "fov": <predicted FOV in radians>,
    "is_headtrack": 0,
    "name": "point<N>",
    "node_type": 0,
    "pan": <predicted yaw in radians>,
    "roll": 0,
    "src_time": <frame number>,
    "state": 7,
    "tilt": <predicted pitch in radians>,
    "time": <frame number>
}
```

Between each pair of keyframes, insert a linear transition:

```json
{
    "name": "point<N>-point<N+1>",
    "node_type": 1,
    "point1X": 0.5,
    "point1Y": 0.5,
    "point2X": 0.5,
    "point2Y": 0.5,
    "type": 1
}
```

### Open Questions

1. **What are valid ranges for `distance`?** Observed: 0.4. The dewarp
   settings show `min_distance: 0, max_distance: 1`. Need more samples.

2. **What does `state: 7` vs `state: 3` mean exactly?** 7 appears to be
   manual keyframes, 3 appears to be deep-track-generated keyframes.

3. **Does `fov.type` switch units?** In `camera_transform`, `type: 0` with
   values ~1.18 (radians). In `original_camera_transform`, `type: 1` with
   value `75` (looks like degrees). This needs verification.

4. **Will Studio accept a modified `.insproj` without complaint?** Unknown.
   The backup file suggests Studio auto-saves, so it reads from disk. But
   it may validate checksums, timestamps, or other integrity markers.

5. **Does `time` always equal `src_time`?** In this test project they're
   identical, but with speed ramping they would diverge (`time` = timeline
   position, `src_time` = position in source footage).
