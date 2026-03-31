"""Parse Insta360 Studio .insprj sidecar files to extract camera trajectories.

The .insprj format is XML containing keyframes with pan, tilt, roll, FOV,
and easing curves between them. This module parses those keyframes and
interpolates them to produce per-frame camera parameters suitable for
training the behavioral cloning model.
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
    fov: float  # Field of view (radians in .insprj, converted to degrees)
    distance: float  # "Distance" parameter (Insta360's internal zoom)


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

    # Search for keyframe elements anywhere in the tree
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


def interpolate_trajectory(
    keyframes: list[Keyframe],
    fps: float,
    total_frames: int,
) -> np.ndarray:
    """Interpolate keyframes to produce per-frame camera parameters.

    Produces a dense trajectory by linearly interpolating between keyframes.
    Uses circular interpolation for pan (yaw) to handle wraparound.

    Args:
        keyframes: Parsed keyframes from .insprj.
        fps: Video frame rate.
        total_frames: Total number of frames in the video.

    Returns:
        Array of shape (total_frames, 3) with columns [yaw_deg, pitch_deg, fov_deg].
        Yaw and pitch in degrees. FOV converted from the .insprj radian format
        to degrees.
    """
    # Extract keyframe timestamps as frame indices
    kf_frames = [kf.time_ms / 1000.0 * fps for kf in keyframes]
    kf_pans = [kf.pan for kf in keyframes]
    kf_tilts = [kf.tilt for kf in keyframes]
    kf_fovs = [np.degrees(kf.fov) for kf in keyframes]  # radians → degrees

    frame_indices = np.arange(total_frames, dtype=np.float64)

    # Interpolate pan (yaw) using sin/cos to handle circular wraparound
    pan_rads = np.radians(kf_pans)
    sin_interp = np.interp(frame_indices, kf_frames, np.sin(pan_rads))
    cos_interp = np.interp(frame_indices, kf_frames, np.cos(pan_rads))
    yaw_deg = np.degrees(np.arctan2(sin_interp, cos_interp))

    # Interpolate tilt (pitch) — linear is fine, no wraparound
    pitch_deg = np.interp(frame_indices, kf_frames, kf_tilts)

    # Interpolate FOV — linear
    fov_deg = np.interp(frame_indices, kf_frames, kf_fovs)

    trajectory = np.stack([yaw_deg, pitch_deg, fov_deg], axis=1)
    return trajectory


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

    # Exact name match
    for ext in [".insprj", ".insproj"]:
        candidate = parent / (stem + ext)
        if candidate.exists():
            return candidate

    # Any sidecar in same directory
    for ext in ["*.insprj", "*.insproj"]:
        matches = list(parent.glob(ext))
        if len(matches) == 1:
            return matches[0]

    return None
