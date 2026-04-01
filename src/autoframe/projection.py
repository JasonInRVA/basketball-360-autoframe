"""Equirectangular <-> rectilinear projection math.

NOTE: This module is only used during training (to preprocess frames for
the CNN) and for optional local preview. It is NOT on the critical output
path — Insta360 Studio handles the actual rendering from the .insprj
sidecar we generate. Bugs here affect model training quality, not output.

Provides conversion between equirectangular and perspective views, and
utilities for mapping pixel coordinates to spherical angles.
"""

import numpy as np


def equirect_to_rectilinear(
    equirect: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    """Extract a rectilinear (perspective) crop from an equirectangular image.

    Args:
        equirect: Input equirectangular image (H, W, 3).
        yaw: Horizontal rotation in degrees (-180 to 180).
        pitch: Vertical rotation in degrees (-90 to 90).
        fov: Horizontal field of view in degrees.
        out_width: Output image width.
        out_height: Output image height.

    Returns:
        Rectilinear image (out_height, out_width, 3).
    """
    import cv2

    eq_h, eq_w = equirect.shape[:2]

    # Convert angles to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    # Compute focal length from FOV
    f = out_width / (2.0 * np.tan(np.radians(fov) / 2.0))

    # Build pixel coordinate grid for output image
    u = np.arange(out_width, dtype=np.float64) - out_width / 2.0
    v = np.arange(out_height, dtype=np.float64) - out_height / 2.0
    u, v = np.meshgrid(u, v)

    # Direction vectors in camera space
    x = u
    y = v
    z = np.full_like(u, f)

    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm

    # Rotation matrices (yaw around Y, pitch around X)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)

    # Apply pitch (rotation around X-axis)
    y1 = y * cos_pitch - z * sin_pitch
    z1 = y * sin_pitch + z * cos_pitch

    # Apply yaw (rotation around Y-axis)
    x2 = x * cos_yaw + z1 * sin_yaw
    z2 = -x * sin_yaw + z1 * cos_yaw

    # Convert 3D direction to equirectangular coordinates
    theta = np.arctan2(x2, z2)  # longitude
    phi = np.arcsin(np.clip(y1, -1, 1))  # latitude

    # Map to pixel coordinates in equirectangular image
    src_x = ((theta / np.pi + 1.0) / 2.0 * eq_w).astype(np.float32)
    src_y = ((phi / (np.pi / 2.0) + 1.0) / 2.0 * eq_h).astype(np.float32)

    # Wrap horizontal coordinate
    src_x = src_x % eq_w

    # Remap
    result = cv2.remap(
        equirect,
        src_x,
        src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return result


def pixel_to_spherical(
    px: float,
    py: float,
    eq_width: int,
    eq_height: int,
) -> tuple[float, float]:
    """Convert equirectangular pixel coordinates to yaw/pitch in degrees.

    Args:
        px: X pixel coordinate.
        py: Y pixel coordinate.
        eq_width: Equirectangular image width.
        eq_height: Equirectangular image height.

    Returns:
        (yaw, pitch) in degrees.
    """
    yaw = (px / eq_width * 360.0) - 180.0
    pitch = 90.0 - (py / eq_height * 180.0)
    return yaw, pitch


def tile_positions(num_tiles: int, fov: float) -> list[tuple[float, float]]:
    """Generate evenly-spaced yaw/pitch positions for detection tiles.

    Tiles are arranged in a ring around the equator (pitch=0) since
    basketball courts are at eye level.

    Args:
        num_tiles: Number of tiles to generate.
        fov: FOV of each tile in degrees.

    Returns:
        List of (yaw, pitch) tuples in degrees.
    """
    step = 360.0 / num_tiles
    return [(i * step - 180.0, 0.0) for i in range(num_tiles)]
