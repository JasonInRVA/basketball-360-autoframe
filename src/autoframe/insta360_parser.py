"""Read and write Insta360 Studio .insprj sidecar files.

The .insprj format is XML containing keyframes with pan, tilt, roll, FOV,
and easing curves between them. This module:
  - Parses existing .insprj files (for training data extraction)
  - Writes new .insprj files (the primary output of the AI model)
  - Interpolates sparse keyframes to dense per-frame trajectories

The .insprj file is the bridge between this tool and Insta360 Studio:
the AI predicts a camera trajectory, we write it as .insprj, and Studio
renders the final video with full quality, audio, stabilization, etc.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Keyframe:
    """A single keyframe from an Insta360 project file."""

    time_ms: float  # Timestamp in milliseconds
    pan: float  # Horizontal angle (degrees)
    tilt: float  # Vertical angle (degrees)
    roll: float  # Roll angle (degrees)
    fov: float  # Field of view (radians, as stored in .insprj)
    distance: float  # Insta360's internal zoom/distance parameter


# ── Reading ──────────────────────────────────────────────────────────────


def parse_insprj(path: str | Path) -> list[Keyframe]:
    """Parse an Insta360 Studio .insprj XML file.

    Args:
        path: Path to the .insprj file.

    Returns:
        List of Keyframe objects sorted by time.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file can't be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sidecar file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    keyframes: list[Keyframe] = []

    for kf_elem in root.iter("keyframe"):
        try:
            keyframes.append(
                Keyframe(
                    time_ms=float(kf_elem.get("time", 0)),
                    pan=float(kf_elem.get("pan", 0)),
                    tilt=float(kf_elem.get("tilt", 0)),
                    roll=float(kf_elem.get("roll", 0)),
                    fov=float(kf_elem.get("fov", 1.309)),  # ~75° default
                    distance=float(kf_elem.get("distance", 0.6)),
                )
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to parse keyframe: {kf_elem.attrib}") from e

    if not keyframes:
        raise ValueError(f"No keyframes found in {path}")

    keyframes.sort(key=lambda kf: kf.time_ms)
    return keyframes


# ── Writing ──────────────────────────────────────────────────────────────


def write_insprj(
    keyframes: list[Keyframe],
    output_path: str | Path,
    source_video_path: str | Path | None = None,
) -> Path:
    """Write keyframes to an Insta360 Studio .insprj XML file.

    Generates a sidecar file that Insta360 Studio can open and use to
    render a reframed video with full quality, audio, and stabilization.

    Args:
        keyframes: List of keyframes to write.
        output_path: Where to save the .insprj file.
        source_video_path: Optional path to the source video (stored as
            metadata in the file for Insta360 Studio to find it).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("project", version="2.0.0")

    # Metadata
    meta = ET.SubElement(root, "meta")
    meta.set("app", "basketball-360-autoframe")
    meta.set("version", "0.2.0")

    # Source file reference
    if source_video_path is not None:
        file_group = ET.SubElement(root, "file_group")
        file_group.set("type", "video_normal")
        file_group.set("folder", str(Path(source_video_path).parent))

    # Scheme with timeline
    scheme = ET.SubElement(root, "scheme", id="autoframe_generated")
    timeline = ET.SubElement(scheme, "timeline")

    for i, kf in enumerate(keyframes):
        kf_elem = ET.SubElement(timeline, "keyframe")
        kf_elem.set("time", str(int(kf.time_ms)))
        kf_elem.set("pan", f"{kf.pan:.4f}")
        kf_elem.set("tilt", f"{kf.tilt:.4f}")
        kf_elem.set("roll", f"{kf.roll:.4f}")
        kf_elem.set("fov", f"{kf.fov:.6f}")
        kf_elem.set("distance", f"{kf.distance:.6f}")
        kf_elem.set("id", f"point{i}")

        # Add linear transition between keyframes
        if i < len(keyframes) - 1:
            transition = ET.SubElement(timeline, "transition")
            transition.set("type", "Linear")

    # Write with XML declaration
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

    return output_path


def trajectory_to_keyframes(
    yaw_deg: list[float],
    pitch_deg: list[float],
    fov_deg: list[float],
    fps: float,
    keyframe_interval_ms: float = 200.0,
    default_roll: float = 0.0,
    default_distance: float = 0.6,
) -> list[Keyframe]:
    """Convert a dense per-frame trajectory to sparse keyframes for .insprj.

    Insta360 Studio interpolates between keyframes, so we don't need one
    per frame. This function subsamples the trajectory at regular intervals,
    which is both more compact and allows Studio to apply its own easing.

    Args:
        yaw_deg: Per-frame yaw (pan) in degrees.
        pitch_deg: Per-frame pitch (tilt) in degrees.
        fov_deg: Per-frame FOV in degrees.
        fps: Video frame rate.
        keyframe_interval_ms: Time between keyframes in milliseconds.
            200ms = 5 keyframes/second, which captures camera motion
            without overwhelming Studio.
        default_roll: Roll value for all keyframes (model doesn't predict roll).
        default_distance: Distance value for all keyframes.

    Returns:
        List of Keyframe objects.
    """
    total_frames = len(yaw_deg)
    frame_interval = max(1, int(keyframe_interval_ms / 1000.0 * fps))

    keyframes: list[Keyframe] = []
    for i in range(0, total_frames, frame_interval):
        time_ms = (i / fps) * 1000.0
        keyframes.append(
            Keyframe(
                time_ms=time_ms,
                pan=yaw_deg[i],
                tilt=pitch_deg[i],
                roll=default_roll,
                fov=np.radians(fov_deg[i]),  # degrees → radians for .insprj
                distance=default_distance,
            )
        )

    # Always include the last frame
    if (total_frames - 1) % frame_interval != 0:
        last = total_frames - 1
        keyframes.append(
            Keyframe(
                time_ms=(last / fps) * 1000.0,
                pan=yaw_deg[last],
                tilt=pitch_deg[last],
                roll=default_roll,
                fov=np.radians(fov_deg[last]),
                distance=default_distance,
            )
        )

    return keyframes


# ── Interpolation (for training) ─────────────────────────────────────────


def interpolate_trajectory(
    keyframes: list[Keyframe],
    fps: float,
    total_frames: int,
) -> np.ndarray:
    """Interpolate sparse keyframes to dense per-frame camera parameters.

    Used during training to produce per-frame labels from .insprj keyframes.

    Args:
        keyframes: Parsed keyframes from .insprj.
        fps: Video frame rate.
        total_frames: Total number of frames in the video.

    Returns:
        Array of shape (total_frames, 3) with columns [yaw_deg, pitch_deg, fov_deg].
    """
    kf_frames = [kf.time_ms / 1000.0 * fps for kf in keyframes]
    kf_pans = [kf.pan for kf in keyframes]
    kf_tilts = [kf.tilt for kf in keyframes]
    kf_fovs = [np.degrees(kf.fov) for kf in keyframes]  # radians → degrees

    frame_indices = np.arange(total_frames, dtype=np.float64)

    # Circular interpolation for pan (yaw) to handle wraparound
    pan_rads = np.radians(kf_pans)
    sin_interp = np.interp(frame_indices, kf_frames, np.sin(pan_rads))
    cos_interp = np.interp(frame_indices, kf_frames, np.cos(pan_rads))
    yaw_deg = np.degrees(np.arctan2(sin_interp, cos_interp))

    pitch_deg = np.interp(frame_indices, kf_frames, kf_tilts)
    fov_deg = np.interp(frame_indices, kf_frames, kf_fovs)

    return np.stack([yaw_deg, pitch_deg, fov_deg], axis=1)


# ── Discovery ────────────────────────────────────────────────────────────


def find_sidecar(video_path: str | Path) -> Path | None:
    """Attempt to find the .insprj sidecar file for a given video.

    Searches for:
      1. Same name with .insprj extension
      2. Same name with .insproj extension (project-page variant)
      3. Any .insprj file in the same directory

    Args:
        video_path: Path to the source video.

    Returns:
        Path to sidecar file, or None if not found.
    """
    video_path = Path(video_path)
    parent = video_path.parent
    stem = video_path.stem

    for ext in [".insprj", ".insproj"]:
        candidate = parent / (stem + ext)
        if candidate.exists():
            return candidate

    for ext in ["*.insprj", "*.insproj"]:
        matches = list(parent.glob(ext))
        if len(matches) == 1:
            return matches[0]

    return None
