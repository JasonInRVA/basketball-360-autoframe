"""Virtual camera controller — decides where to point and smooths movement.

Takes detections each frame and outputs a smooth yaw/pitch/fov trajectory
that keeps the basketball action centered in frame.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from autoframe.config import CameraConfig
from autoframe.detector import Detection


class VirtualCamera:
    """Tracks the action and controls virtual camera orientation."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.yaw = config.initial_yaw
        self.pitch = config.initial_pitch
        self.fov = config.fov

    def update(self, detections: list[Detection]) -> tuple[float, float, float]:
        """Given current frame detections, compute the next camera position.

        Args:
            detections: Detected players and balls in this frame.

        Returns:
            (yaw, pitch, fov) for this frame.
        """
        target_yaw, target_pitch = self._compute_target(detections)

        if target_yaw is not None:
            # Smooth yaw with wraparound handling
            self.yaw = self._smooth_angle(self.yaw, target_yaw, self.config.smoothing_factor)
            self.pitch = (
                self.config.smoothing_factor * self.pitch
                + (1 - self.config.smoothing_factor) * target_pitch
            )

            # Adaptive FOV: zoom in when action is clustered, zoom out when spread
            spread = self._compute_spread(detections)
            target_fov = np.clip(spread * 1.5, self.config.min_fov, self.config.max_fov)
            self.fov = self.config.smoothing_factor * self.fov + (1 - self.config.smoothing_factor) * target_fov

        return self.yaw, self.pitch, self.fov

    def _compute_target(
        self, detections: list[Detection]
    ) -> tuple[float | None, float | None]:
        """Compute where the camera should point based on detections.

        Priority: ball position if detected, otherwise centroid of players.
        """
        if not detections:
            return None, None

        balls = [d for d in detections if d.label == "ball"]
        players = [d for d in detections if d.label == "player"]

        if balls:
            # Weight toward the ball but keep players in view
            ball = max(balls, key=lambda d: d.confidence)
            if players:
                player_yaw = self._circular_mean([p.yaw for p in players])
                player_pitch = np.mean([p.pitch for p in players])
                # 60% ball, 40% player centroid — keeps the action contextual
                target_yaw = self._circular_weighted_mean(
                    [ball.yaw, player_yaw], [0.6, 0.4]
                )
                target_pitch = 0.6 * ball.pitch + 0.4 * player_pitch
            else:
                target_yaw = ball.yaw
                target_pitch = ball.pitch
        elif players:
            target_yaw = self._circular_mean([p.yaw for p in players])
            target_pitch = np.mean([p.pitch for p in players])
        else:
            return None, None

        return target_yaw, target_pitch

    def _compute_spread(self, detections: list[Detection]) -> float:
        """Compute the angular spread of detections (for adaptive FOV)."""
        if len(detections) < 2:
            return self.config.fov

        yaws = [d.yaw for d in detections]
        pitches = [d.pitch for d in detections]

        # Use circular range for yaw
        yaw_spread = self._circular_range(yaws)
        pitch_spread = max(pitches) - min(pitches)

        return max(yaw_spread, pitch_spread) + 20.0  # padding

    @staticmethod
    def _smooth_angle(current: float, target: float, alpha: float) -> float:
        """Exponential smoothing with wraparound handling for angles."""
        diff = target - current
        # Handle wraparound
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return current + (1 - alpha) * diff

    @staticmethod
    def _circular_mean(angles: list[float]) -> float:
        """Compute mean of angles handling wraparound."""
        rads = np.radians(angles)
        mean_sin = np.mean(np.sin(rads))
        mean_cos = np.mean(np.cos(rads))
        return float(np.degrees(np.arctan2(mean_sin, mean_cos)))

    @staticmethod
    def _circular_weighted_mean(angles: list[float], weights: list[float]) -> float:
        """Weighted circular mean."""
        rads = np.radians(angles)
        w = np.array(weights)
        mean_sin = np.sum(w * np.sin(rads))
        mean_cos = np.sum(w * np.cos(rads))
        return float(np.degrees(np.arctan2(mean_sin, mean_cos)))

    @staticmethod
    def _circular_range(angles: list[float]) -> float:
        """Compute the smallest arc containing all angles."""
        sorted_angles = sorted(a % 360 for a in angles)
        if len(sorted_angles) < 2:
            return 0.0
        gaps = [
            sorted_angles[i + 1] - sorted_angles[i] for i in range(len(sorted_angles) - 1)
        ]
        gaps.append(360 - sorted_angles[-1] + sorted_angles[0])
        max_gap = max(gaps)
        return 360 - max_gap


def smooth_trajectory(
    yaws: list[float],
    pitches: list[float],
    fovs: list[float],
    window: int = 15,
) -> tuple[list[float], list[float], list[float]]:
    """Post-process smooth pass over the full trajectory.

    Useful for a second pass after the initial per-frame smoothing.

    Args:
        yaws: Per-frame yaw values.
        pitches: Per-frame pitch values.
        fovs: Per-frame FOV values.
        window: Smoothing window size in frames.

    Returns:
        Smoothed (yaws, pitches, fovs).
    """
    # For yaw, smooth sin/cos components separately to handle wraparound
    rads = np.radians(yaws)
    sin_smooth = uniform_filter1d(np.sin(rads), size=window)
    cos_smooth = uniform_filter1d(np.cos(rads), size=window)
    smooth_yaws = np.degrees(np.arctan2(sin_smooth, cos_smooth)).tolist()

    smooth_pitches = uniform_filter1d(np.array(pitches, dtype=float), size=window).tolist()
    smooth_fovs = uniform_filter1d(np.array(fovs, dtype=float), size=window).tolist()

    return smooth_yaws, smooth_pitches, smooth_fovs
