"""Read and write Insta360 Studio project files (.insproj).

Insta360 Studio uses a JSON-based project directory structure. The camera
trajectory (pan/tilt/fov keyframes) lives inside the main .insproj file
at: tracks[1].clips[0].key_frame_track.node_list

This module:
  - Reads existing .insproj files to extract keyframe trajectories (for training)
  - Modifies existing .insproj files to inject AI-generated keyframes (for output)
  - Interpolates sparse keyframes to dense per-frame trajectories

See INSTA360_FORMAT.md for a detailed description of the file format.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Keyframe:
    """A single keyframe from an Insta360 Studio project.

    All angles are in RADIANS (matching the .insproj format).
    Time values are FRAME NUMBERS (not milliseconds).
    """

    time: int  # Frame number on the timeline
    src_time: int  # Frame number in source video
    pan: float  # Yaw in radians
    tilt: float  # Pitch in radians (negative = looking down)
    roll: float  # Roll in radians
    fov: float  # Field of view in radians
    distance: float  # Zoom/perspective parameter (0-1)


# ── Reading ──────────────────────────────────────────────────────────────


def read_project(path: str | Path) -> dict:
    """Read an Insta360 Studio .insproj file.

    Args:
        path: Path to the .insproj file (or project directory).

    Returns:
        Parsed JSON as a dict.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file can't be parsed as JSON.
    """
    path = _resolve_insproj_path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {path}: {e}") from e


def extract_keyframes(project: dict, clip_index: int = 0) -> list[Keyframe]:
    """Extract keyframes from a parsed .insproj project.

    Args:
        project: Parsed .insproj dict (from read_project).
        clip_index: Which clip in the main video track (usually 0).

    Returns:
        List of Keyframe objects sorted by time.

    Raises:
        ValueError: If the project structure is unexpected.
    """
    clip = _get_clip(project, clip_index)

    key_frame_track = clip.get("key_frame_track", {})
    node_list = key_frame_track.get("node_list", [])

    if not node_list:
        raise ValueError(
            "No keyframes found. The clip's key_frame_track.node_list is empty. "
            "Add at least one keyframe in Insta360 Studio before using this tool."
        )

    keyframes: list[Keyframe] = []
    for node in node_list:
        if node.get("node_type") != 0:
            continue  # Skip transition nodes

        keyframes.append(
            Keyframe(
                time=int(node.get("time", 0)),
                src_time=int(node.get("src_time", 0)),
                pan=float(node.get("pan", 0)),
                tilt=float(node.get("tilt", 0)),
                roll=float(node.get("roll", 0)),
                fov=float(node.get("fov", 1.18)),
                distance=float(node.get("distance", 0.4)),
            )
        )

    keyframes.sort(key=lambda kf: kf.time)
    return keyframes


def get_clip_info(project: dict, clip_index: int = 0) -> dict:
    """Extract basic clip metadata.

    Returns:
        Dict with source_frames, left_trim, right_trim, name, url.
    """
    clip = _get_clip(project, clip_index)
    return {
        "source_frames": clip.get("source_total_frames", 0),
        "left_trim": clip.get("leftTrim", 0),
        "right_trim": clip.get("rightTrim", 0),
        "name": clip.get("name", ""),
        "url": clip.get("url", ""),
        "fps": project.get("fps", 30),
    }


# ── Writing ──────────────────────────────────────────────────────────────


def inject_keyframes(
    project: dict,
    keyframes: list[Keyframe],
    clip_index: int = 0,
) -> dict:
    """Replace the keyframe track in a project with new keyframes.

    This modifies the project dict in-place and returns it. The original
    project should be loaded from an existing Studio project (template).

    Args:
        project: Parsed .insproj dict.
        keyframes: New keyframes to inject.
        clip_index: Which clip to modify.

    Returns:
        The modified project dict.
    """
    clip = _get_clip(project, clip_index)

    # Build node_list: alternating keyframes and transitions
    node_list: list[dict] = []

    for i, kf in enumerate(keyframes):
        # Keyframe node
        node_list.append({
            "auto_fov": 0,
            "distance": kf.distance,
            "fov": kf.fov,
            "is_headtrack": 0,
            "name": f"point{i}",
            "node_type": 0,
            "pan": kf.pan,
            "roll": kf.roll,
            "src_time": kf.src_time,
            "state": 7,  # Manual keyframe
            "tilt": kf.tilt,
            "time": kf.time,
        })

        # Transition node (linear) between keyframes
        if i < len(keyframes) - 1:
            node_list.append({
                "name": f"point{i}-point{i + 1}",
                "node_type": 1,
                "point1X": 0.5,
                "point1Y": 0.5,
                "point2X": 0.5,
                "point2Y": 0.5,
                "type": 1,  # Linear easing
            })

    # Inject into clip
    if "key_frame_track" not in clip:
        clip["key_frame_track"] = {}
    clip["key_frame_track"]["node_list"] = node_list

    # Update camera_transform to match the first keyframe
    if keyframes:
        kf0 = keyframes[0]
        clip["camera_transform"] = {
            "distance": kf0.distance,
            "eulers": {
                "pitch": {"type": 0, "value": kf0.tilt},
                "roll": {"type": 0, "value": kf0.roll},
                "yaw": {"type": 0, "value": kf0.pan},
            },
            "fov": {"type": 0, "value": kf0.fov},
        }

    # Enable user keyframes
    clip["enable_user_keyframe"] = True
    clip["use_default_camera_transform"] = False

    # Clear any deep track data (our keyframes replace it)
    if "deep_track" in clip:
        clip["deep_track"]["deep_track_list"] = []

    return project


def save_project(
    project: dict,
    output_path: str | Path,
    backup: bool = True,
) -> Path:
    """Save a modified project to disk.

    Args:
        project: The project dict to save.
        output_path: Path to the .insproj file.
        backup: If True and the file already exists, create a _backup copy.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backup of existing file
    if backup and output_path.exists():
        backup_path = output_path.with_name(
            output_path.stem + "_backup" + output_path.suffix
        )
        shutil.copy2(output_path, backup_path)

    output_path.write_text(
        json.dumps(project, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


# ── Trajectory Conversion ────────────────────────────────────────────────


def trajectory_to_keyframes(
    pan_rad: list[float],
    tilt_rad: list[float],
    fov_rad: list[float],
    fps: float,
    keyframe_interval_frames: int = 6,
    default_roll: float = 0.0,
    default_distance: float = 0.4,
) -> list[Keyframe]:
    """Convert a dense per-frame trajectory to sparse keyframes.

    Insta360 Studio interpolates between keyframes, so we don't need one
    per frame. This subsamples at regular intervals.

    Args:
        pan_rad: Per-frame yaw in radians.
        tilt_rad: Per-frame pitch in radians.
        fov_rad: Per-frame FOV in radians.
        fps: Video frame rate (for reference, not used in frame-based timing).
        keyframe_interval_frames: Frames between keyframes. 6 frames at
            30fps = 5 keyframes/second.
        default_roll: Roll value (model doesn't predict roll).
        default_distance: Distance value.

    Returns:
        List of Keyframe objects.
    """
    total_frames = len(pan_rad)
    keyframes: list[Keyframe] = []

    for i in range(0, total_frames, keyframe_interval_frames):
        keyframes.append(
            Keyframe(
                time=i,
                src_time=i,
                pan=pan_rad[i],
                tilt=tilt_rad[i],
                roll=default_roll,
                fov=fov_rad[i],
                distance=default_distance,
            )
        )

    # Always include the last frame
    last = total_frames - 1
    if last % keyframe_interval_frames != 0:
        keyframes.append(
            Keyframe(
                time=last,
                src_time=last,
                pan=pan_rad[last],
                tilt=tilt_rad[last],
                roll=default_roll,
                fov=fov_rad[last],
                distance=default_distance,
            )
        )

    return keyframes


# ── Interpolation (for training) ─────────────────────────────────────────


def interpolate_trajectory(
    keyframes: list[Keyframe],
    total_frames: int,
) -> np.ndarray:
    """Interpolate sparse keyframes to dense per-frame camera parameters.

    Used during training to produce per-frame labels from project keyframes.

    Args:
        keyframes: Keyframes extracted from a project.
        total_frames: Total number of frames in the source video.

    Returns:
        Array of shape (total_frames, 3) with columns [pan_rad, tilt_rad, fov_rad].
        All values in radians.
    """
    if not keyframes:
        raise ValueError("Cannot interpolate empty keyframe list")

    kf_frames = [float(kf.time) for kf in keyframes]
    kf_pans = [kf.pan for kf in keyframes]
    kf_tilts = [kf.tilt for kf in keyframes]
    kf_fovs = [kf.fov for kf in keyframes]

    frame_indices = np.arange(total_frames, dtype=np.float64)

    # Circular interpolation for pan (yaw) to handle wraparound
    sin_interp = np.interp(frame_indices, kf_frames, np.sin(kf_pans))
    cos_interp = np.interp(frame_indices, kf_frames, np.cos(kf_pans))
    pan_rad = np.arctan2(sin_interp, cos_interp)

    # Linear interpolation for tilt and FOV
    tilt_rad = np.interp(frame_indices, kf_frames, kf_tilts)
    fov_rad = np.interp(frame_indices, kf_frames, kf_fovs)

    return np.stack([pan_rad, tilt_rad, fov_rad], axis=1)


# ── Discovery ────────────────────────────────────────────────────────────


def find_project(video_path: str | Path) -> Path | None:
    """Attempt to find an Insta360 Studio project for a given video.

    Searches for .insproj files in the same directory and parent directories.

    Args:
        video_path: Path to the source video.

    Returns:
        Path to project directory or .insproj file, or None if not found.
    """
    video_path = Path(video_path)

    # Check for .insproj files in the same directory
    for insproj in video_path.parent.glob("*.insproj"):
        return insproj

    # Check parent directory
    for insproj in video_path.parent.parent.glob("*.insproj"):
        return insproj

    return None


def find_insproj_in_directory(project_dir: str | Path) -> Path | None:
    """Find the .insproj file inside a project directory.

    Args:
        project_dir: Path to the project directory.

    Returns:
        Path to the .insproj file, or None.
    """
    project_dir = Path(project_dir)
    matches = [
        f for f in project_dir.glob("*.insproj")
        if "_backup" not in f.name
    ]
    return matches[0] if matches else None


# ── Internal helpers ─────────────────────────────────────────────────────


def _resolve_insproj_path(path: str | Path) -> Path:
    """Resolve a path that might be a directory or .insproj file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    if path.is_dir():
        found = find_insproj_in_directory(path)
        if found is None:
            raise FileNotFoundError(f"No .insproj file found in {path}")
        return found

    return path


def _get_clip(project: dict, clip_index: int) -> dict:
    """Get a clip from the main video track."""
    tracks = project.get("tracks", [])

    # Find the main video track (type 0)
    video_track = None
    for track in tracks:
        if track.get("type") == 0:
            video_track = track
            break

    if video_track is None:
        raise ValueError("No video track (type 0) found in project")

    clips = video_track.get("clips", [])
    if not clips:
        raise ValueError(
            "No clips in video track. Add a video clip in Insta360 Studio first."
        )

    if clip_index >= len(clips):
        raise ValueError(
            f"Clip index {clip_index} out of range (project has {len(clips)} clips)"
        )

    return clips[clip_index]
