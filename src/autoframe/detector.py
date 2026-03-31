"""Object detection on equirectangular frames using tiled YOLO inference.

Strategy: extract overlapping perspective tiles from the equirectangular
frame, run YOLO on each tile, then map detections back to equirectangular
coordinates and merge with NMS.
"""

from dataclasses import dataclass

import numpy as np

from autoframe.config import DetectionConfig
from autoframe.projection import (
    equirect_to_rectilinear,
    pixel_to_spherical,
    tile_positions,
)


@dataclass
class Detection:
    """A single detected object in equirectangular coordinates."""

    yaw: float  # degrees
    pitch: float  # degrees
    label: str  # "player" or "ball"
    confidence: float
    bbox_equirect: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2 in equirect


class TiledDetector:
    """Runs YOLO on perspective tiles extracted from equirectangular frames."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_path)
        return self._model

    def detect(self, equirect_frame: np.ndarray) -> list[Detection]:
        """Run tiled detection on an equirectangular frame.

        Args:
            equirect_frame: Full equirectangular image (H, W, 3).

        Returns:
            List of detections in spherical coordinates.
        """
        eq_h, eq_w = equirect_frame.shape[:2]
        tile_size = 640  # YOLO expects square-ish input
        tiles = tile_positions(self.config.num_tiles, self.config.tile_fov)
        all_detections: list[Detection] = []

        for yaw, pitch in tiles:
            # Extract perspective tile
            tile_img = equirect_to_rectilinear(
                equirect_frame, yaw, pitch, self.config.tile_fov, tile_size, tile_size
            )

            # Run YOLO
            results = self.model(tile_img, verbose=False, conf=self.config.confidence_threshold)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == self.config.player_class_id:
                        label = "player"
                    elif cls_id == self.config.ball_class_id:
                        label = "ball"
                    else:
                        continue

                    # Get center of detection in tile pixel coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # Map tile pixel back to approximate equirectangular position
                    # The tile center corresponds to (yaw, pitch)
                    offset_yaw = (cx - tile_size / 2) / tile_size * self.config.tile_fov
                    offset_pitch = -(cy - tile_size / 2) / tile_size * self.config.tile_fov

                    det_yaw = yaw + offset_yaw
                    det_pitch = pitch + offset_pitch

                    all_detections.append(
                        Detection(
                            yaw=det_yaw,
                            pitch=det_pitch,
                            label=label,
                            confidence=conf,
                        )
                    )

        return self._nms_spherical(all_detections)

    def _nms_spherical(
        self, detections: list[Detection], angle_threshold: float = 5.0
    ) -> list[Detection]:
        """Simple NMS in spherical coordinates to remove duplicate detections
        from overlapping tiles.

        Args:
            detections: All detections from all tiles.
            angle_threshold: Minimum angular distance (degrees) between
                detections of the same class.

        Returns:
            Filtered detections.
        """
        if not detections:
            return []

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        for det in detections:
            is_duplicate = False
            for existing in kept:
                if det.label != existing.label:
                    continue
                # Angular distance (approximate)
                dy = det.yaw - existing.yaw
                dp = det.pitch - existing.pitch
                # Handle wraparound
                if dy > 180:
                    dy -= 360
                elif dy < -180:
                    dy += 360
                dist = np.sqrt(dy**2 + dp**2)
                if dist < angle_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)

        return kept
