"""Tests for the Insta360 Studio .insproj parser and writer."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from autoframe.insta360_parser import (
    Keyframe,
    extract_keyframes,
    get_clip_info,
    inject_keyframes,
    interpolate_trajectory,
    read_project,
    save_project,
    trajectory_to_keyframes,
)

# Path to real test project
TEST_PROJECT_DIR = Path(__file__).parent / "Insta360 Test Project"
TEST_INSPROJ = TEST_PROJECT_DIR / "9a741808-161c-4278-89e9-19c742b527f7.insproj"


# ── Reading tests (against real project) ─────────────────────────────────


@pytest.mark.skipif(not TEST_INSPROJ.exists(), reason="Test project not available")
class TestReadRealProject:
    def test_read_project(self):
        project = read_project(TEST_INSPROJ)
        assert isinstance(project, dict)
        assert "tracks" in project
        assert project["fps"] == 30

    def test_read_project_from_directory(self):
        project = read_project(TEST_PROJECT_DIR)
        assert isinstance(project, dict)
        assert "tracks" in project

    def test_extract_keyframes(self):
        project = read_project(TEST_INSPROJ)
        keyframes = extract_keyframes(project)
        assert len(keyframes) == 4  # point0, point1, and 2 deep track keyframes

    def test_keyframe_values_are_radians(self):
        project = read_project(TEST_INSPROJ)
        keyframes = extract_keyframes(project)
        for kf in keyframes:
            # All angles should be in radian range (roughly -pi to pi)
            assert -4 < kf.pan < 4, f"pan {kf.pan} doesn't look like radians"
            assert -4 < kf.tilt < 4, f"tilt {kf.tilt} doesn't look like radians"
            assert 0 < kf.fov < 4, f"fov {kf.fov} doesn't look like radians"

    def test_keyframe_times_are_frame_numbers(self):
        project = read_project(TEST_INSPROJ)
        keyframes = extract_keyframes(project)
        # Frame numbers should be non-negative integers
        for kf in keyframes:
            assert kf.time >= 0
            assert kf.time == int(kf.time)

    def test_keyframes_sorted_by_time(self):
        project = read_project(TEST_INSPROJ)
        keyframes = extract_keyframes(project)
        times = [kf.time for kf in keyframes]
        assert times == sorted(times)

    def test_first_keyframe_matches_expected(self):
        project = read_project(TEST_INSPROJ)
        keyframes = extract_keyframes(project)
        kf0 = keyframes[0]
        assert kf0.time == 0
        assert abs(kf0.pan - 0.2096) < 0.01
        assert abs(kf0.tilt - (-0.3110)) < 0.01

    def test_get_clip_info(self):
        project = read_project(TEST_INSPROJ)
        info = get_clip_info(project)
        assert info["source_frames"] == 18263
        assert info["name"] == "VID_012.insv"
        assert info["fps"] == 30


# ── Writing tests ────────────────────────────────────────────────────────


def test_inject_keyframes_into_project(tmp_path):
    """Inject keyframes into a minimal project structure."""
    project = {
        "fps": 30,
        "tracks": [
            {"type": 3, "clips": []},
            {"type": 0, "clips": [
                {
                    "key_frame_track": {"node_list": []},
                    "camera_transform": {},
                    "enable_user_keyframe": False,
                    "use_default_camera_transform": True,
                }
            ]},
            {"type": 2, "clips": []},
        ],
    }

    keyframes = [
        Keyframe(time=0, src_time=0, pan=0.1, tilt=-0.3, roll=0, fov=1.2, distance=0.4),
        Keyframe(time=30, src_time=30, pan=-0.5, tilt=-0.2, roll=0, fov=0.9, distance=0.4),
    ]

    inject_keyframes(project, keyframes)
    clip = project["tracks"][1]["clips"][0]

    # Check node_list has keyframes and transitions
    node_list = clip["key_frame_track"]["node_list"]
    assert len(node_list) == 3  # kf, transition, kf

    # Check keyframe values
    assert node_list[0]["node_type"] == 0
    assert node_list[0]["pan"] == 0.1
    assert node_list[0]["state"] == 7

    # Check transition
    assert node_list[1]["node_type"] == 1
    assert node_list[1]["type"] == 1  # linear

    # Check camera_transform updated
    assert abs(clip["camera_transform"]["eulers"]["yaw"]["value"] - 0.1) < 0.001

    # Check flags
    assert clip["enable_user_keyframe"] is True
    assert clip["use_default_camera_transform"] is False


def test_save_and_read_roundtrip(tmp_path):
    """Save a project and read it back."""
    project = {
        "fps": 30,
        "tracks": [
            {"type": 3, "clips": []},
            {"type": 0, "clips": [
                {"key_frame_track": {"node_list": []}, "camera_transform": {}}
            ]},
        ],
    }

    keyframes = [
        Keyframe(time=0, src_time=0, pan=0.5, tilt=-0.3, roll=0, fov=1.0, distance=0.4),
        Keyframe(time=60, src_time=60, pan=-0.2, tilt=-0.1, roll=0, fov=1.5, distance=0.4),
    ]

    inject_keyframes(project, keyframes)
    path = tmp_path / "test.insproj"
    save_project(project, path, backup=False)

    # Read back
    loaded = read_project(path)
    loaded_kfs = extract_keyframes(loaded)
    assert len(loaded_kfs) == 2
    assert abs(loaded_kfs[0].pan - 0.5) < 0.001
    assert abs(loaded_kfs[1].fov - 1.5) < 0.001


@pytest.mark.skipif(not TEST_INSPROJ.exists(), reason="Test project not available")
def test_inject_into_real_project(tmp_path):
    """Inject keyframes into a copy of the real test project."""
    import shutil

    # Copy project to tmp
    dest = tmp_path / TEST_INSPROJ.name
    shutil.copy2(TEST_INSPROJ, dest)

    project = read_project(dest)
    original_kfs = extract_keyframes(project)

    # Inject new keyframes
    new_kfs = [
        Keyframe(time=0, src_time=0, pan=0.0, tilt=-0.3, roll=0, fov=1.0, distance=0.4),
        Keyframe(time=100, src_time=100, pan=0.5, tilt=-0.2, roll=0, fov=1.2, distance=0.4),
        Keyframe(time=200, src_time=200, pan=-0.3, tilt=-0.4, roll=0, fov=0.8, distance=0.4),
    ]
    inject_keyframes(project, new_kfs)
    save_project(project, dest, backup=False)

    # Verify
    reloaded = read_project(dest)
    result_kfs = extract_keyframes(reloaded)
    assert len(result_kfs) == 3
    assert result_kfs[0].pan == 0.0
    assert result_kfs[2].time == 200


# ── Trajectory conversion tests ──────────────────────────────────────────


def test_trajectory_to_keyframes_count():
    fps = 30.0
    num_frames = 300
    pan = [0.1 * i / num_frames for i in range(num_frames)]
    tilt = [-0.3] * num_frames
    fov = [1.0] * num_frames

    keyframes = trajectory_to_keyframes(pan, tilt, fov, fps, keyframe_interval_frames=6)
    # 300 / 6 = 50 keyframes, plus last frame
    assert 49 <= len(keyframes) <= 52


def test_trajectory_to_keyframes_includes_last_frame():
    num_frames = 100
    pan = [0.0] * num_frames
    tilt = [-0.3] * num_frames
    fov = [1.0] * num_frames

    keyframes = trajectory_to_keyframes(pan, tilt, fov, 30.0, keyframe_interval_frames=7)
    assert keyframes[-1].time == num_frames - 1


def test_trajectory_to_keyframes_values_in_radians():
    """Output keyframes should preserve radian values."""
    keyframes = trajectory_to_keyframes(
        [0.5] * 30, [-0.3] * 30, [1.2] * 30, 30.0, keyframe_interval_frames=10
    )
    assert abs(keyframes[0].pan - 0.5) < 0.001
    assert abs(keyframes[0].tilt - (-0.3)) < 0.001
    assert abs(keyframes[0].fov - 1.2) < 0.001


# ── Interpolation tests ─────────────────────────────────────────────────


def test_interpolate_trajectory_shape():
    keyframes = [
        Keyframe(time=0, src_time=0, pan=0, tilt=0, roll=0, fov=1.0, distance=0.4),
        Keyframe(time=100, src_time=100, pan=0.5, tilt=-0.3, roll=0, fov=1.5, distance=0.4),
    ]
    trajectory = interpolate_trajectory(keyframes, total_frames=200)
    assert trajectory.shape == (200, 3)


def test_interpolate_trajectory_endpoints():
    keyframes = [
        Keyframe(time=0, src_time=0, pan=0.1, tilt=-0.2, roll=0, fov=1.0, distance=0.4),
        Keyframe(time=99, src_time=99, pan=0.5, tilt=-0.4, roll=0, fov=1.5, distance=0.4),
    ]
    trajectory = interpolate_trajectory(keyframes, total_frames=100)
    assert abs(trajectory[0, 0] - 0.1) < 0.01  # pan at start
    assert abs(trajectory[99, 2] - 1.5) < 0.01  # fov at end
