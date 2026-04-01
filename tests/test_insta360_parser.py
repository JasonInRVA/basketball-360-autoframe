"""Tests for the Insta360 .insprj sidecar parser and writer."""

import numpy as np
import pytest

from autoframe.insta360_parser import (
    Keyframe,
    interpolate_trajectory,
    parse_insprj,
    trajectory_to_keyframes,
    write_insprj,
)

SAMPLE_INSPRJ = """\
<?xml version="1.0" encoding="UTF-8"?>
<project version="2.0.0">
  <meta app="Insta360 Studio 2022 4.2.1" creation_time="1648983583" version="4.2.1"/>
  <scheme id="test">
    <timeline>
      <keyframe time="0" pan="0" tilt="0" roll="0" fov="1.5708" distance="0.6"/>
      <keyframe time="10000" pan="45.0" tilt="-10" roll="0" fov="1.309" distance="0.5"/>
      <keyframe time="20000" pan="-30.0" tilt="5" roll="0" fov="1.8" distance="0.7"/>
    </timeline>
  </scheme>
</project>
"""


@pytest.fixture
def sample_insprj_path(tmp_path):
    path = tmp_path / "test.insprj"
    path.write_text(SAMPLE_INSPRJ)
    return path


# ── Parser tests ─────────────────────────────────────────────────────────


def test_parse_insprj_keyframe_count(sample_insprj_path):
    keyframes = parse_insprj(sample_insprj_path)
    assert len(keyframes) == 3


def test_parse_insprj_values(sample_insprj_path):
    keyframes = parse_insprj(sample_insprj_path)
    assert keyframes[0].time_ms == 0.0
    assert keyframes[0].pan == 0.0
    assert keyframes[1].pan == 45.0
    assert keyframes[1].tilt == -10.0
    assert keyframes[2].fov == 1.8


def test_parse_insprj_sorted(sample_insprj_path):
    keyframes = parse_insprj(sample_insprj_path)
    times = [kf.time_ms for kf in keyframes]
    assert times == sorted(times)


def test_parse_insprj_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_insprj("/nonexistent/path.insprj")


# ── Interpolation tests ─────────────────────────────────────────────────


def test_interpolate_trajectory_shape(sample_insprj_path):
    keyframes = parse_insprj(sample_insprj_path)
    trajectory = interpolate_trajectory(keyframes, fps=30.0, total_frames=600)
    assert trajectory.shape == (600, 3)


def test_interpolate_trajectory_start_matches_first_keyframe(sample_insprj_path):
    keyframes = parse_insprj(sample_insprj_path)
    trajectory = interpolate_trajectory(keyframes, fps=30.0, total_frames=600)
    assert abs(trajectory[0, 0] - 0.0) < 1.0  # yaw ~0
    assert abs(trajectory[0, 1] - 0.0) < 1.0  # pitch ~0


# ── Writer tests ─────────────────────────────────────────────────────────


def test_write_insprj_creates_file(tmp_path):
    keyframes = [
        Keyframe(time_ms=0, pan=0, tilt=0, roll=0, fov=1.5, distance=0.6),
        Keyframe(time_ms=5000, pan=30, tilt=-5, roll=0, fov=1.3, distance=0.6),
    ]
    output = tmp_path / "output.insprj"
    result = write_insprj(keyframes, output)
    assert result.exists()
    assert result.stat().st_size > 0


def test_write_then_read_roundtrip(tmp_path):
    """Keyframes should survive a write→read roundtrip."""
    original = [
        Keyframe(time_ms=0, pan=10.5, tilt=-3.2, roll=1.0, fov=1.5708, distance=0.6),
        Keyframe(time_ms=5000, pan=45.0, tilt=0.0, roll=0.0, fov=1.309, distance=0.5),
        Keyframe(time_ms=10000, pan=-20.0, tilt=5.0, roll=0.0, fov=1.8, distance=0.7),
    ]
    output = tmp_path / "roundtrip.insprj"
    write_insprj(original, output)
    parsed = parse_insprj(output)

    assert len(parsed) == len(original)
    for orig, read in zip(original, parsed):
        assert abs(orig.time_ms - read.time_ms) < 1
        assert abs(orig.pan - read.pan) < 0.01
        assert abs(orig.tilt - read.tilt) < 0.01
        assert abs(orig.fov - read.fov) < 0.001


# ── Trajectory-to-keyframes tests ────────────────────────────────────────


def test_trajectory_to_keyframes_count():
    """Should produce keyframes at the requested interval."""
    fps = 30.0
    num_frames = 300  # 10 seconds
    yaw = [float(i) for i in range(num_frames)]
    pitch = [0.0] * num_frames
    fov = [90.0] * num_frames

    keyframes = trajectory_to_keyframes(yaw, pitch, fov, fps, keyframe_interval_ms=200)
    # 10 seconds / 200ms = 50 keyframes, plus possibly the last frame
    assert 49 <= len(keyframes) <= 52


def test_trajectory_to_keyframes_includes_last_frame():
    """Last frame should always be included."""
    fps = 30.0
    num_frames = 100
    yaw = [float(i) for i in range(num_frames)]
    pitch = [0.0] * num_frames
    fov = [90.0] * num_frames

    keyframes = trajectory_to_keyframes(yaw, pitch, fov, fps, keyframe_interval_ms=500)
    last_time_ms = (num_frames - 1) / fps * 1000
    assert abs(keyframes[-1].time_ms - last_time_ms) < 1.0


def test_trajectory_to_keyframes_fov_in_radians():
    """Output keyframes should have FOV in radians (as .insprj expects)."""
    fps = 30.0
    fov_deg = [90.0] * 30
    keyframes = trajectory_to_keyframes(
        [0.0] * 30, [0.0] * 30, fov_deg, fps, keyframe_interval_ms=1000
    )
    # 90 degrees ≈ 1.5708 radians
    assert abs(keyframes[0].fov - np.radians(90.0)) < 0.001
