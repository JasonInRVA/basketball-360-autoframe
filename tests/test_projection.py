"""Tests for equirectangular projection math."""

import numpy as np

from autoframe.projection import pixel_to_spherical, tile_positions


def test_pixel_to_spherical_center():
    """Center of equirectangular image should map to (0, 0)."""
    yaw, pitch = pixel_to_spherical(500, 250, 1000, 500)
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6


def test_pixel_to_spherical_left_edge():
    """Left edge should be -180 degrees yaw."""
    yaw, _ = pixel_to_spherical(0, 250, 1000, 500)
    assert abs(yaw - (-180.0)) < 1e-6


def test_pixel_to_spherical_top():
    """Top of image should be +90 pitch."""
    _, pitch = pixel_to_spherical(500, 0, 1000, 500)
    assert abs(pitch - 90.0) < 1e-6


def test_tile_positions_count():
    """Should generate the requested number of tiles."""
    tiles = tile_positions(6, 90.0)
    assert len(tiles) == 6


def test_tile_positions_cover_360():
    """Tiles should span the full 360 degrees."""
    tiles = tile_positions(4, 90.0)
    yaws = [t[0] for t in tiles]
    diffs = [yaws[i + 1] - yaws[i] for i in range(len(yaws) - 1)]
    for d in diffs:
        assert abs(d - 90.0) < 1e-6


def test_tile_positions_equator():
    """All tiles should be at pitch=0 (equator)."""
    tiles = tile_positions(6, 90.0)
    for _, pitch in tiles:
        assert pitch == 0.0
