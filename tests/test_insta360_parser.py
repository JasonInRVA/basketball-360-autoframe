"""Tests for the Insta360 .insprj sidecar parser."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from autoframe.insta360_parser import (
    Keyframe,
    interpolate_trajectory,
    parse_insprj,
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


def test_interpolate_trajectory_shape(sample_insprj_path):
    keyframes = parse_insprj(sample_insprj_path)
    trajectory = interpolate_trajectory(keyframes, fps=30.0, total_frames=600)
    assert trajectory.shape == (600, 3)


def test_interpolate_trajectory_start_matches_first_keyframe(sample_insprj_path):
    keyframes = parse_insprj(sample_insprj_path)
    trajectory = interpolate_trajectory(keyframes, fps=30.0, total_frames=600)
    # First frame should be close to first keyframe values
    assert abs(trajectory[0, 0] - 0.0) < 1.0  # yaw ~0
    assert abs(trajectory[0, 1] - 0.0) < 1.0  # pitch ~0
