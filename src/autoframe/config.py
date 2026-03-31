"""Configuration and defaults for the autoframe pipeline."""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """Virtual camera parameters."""

    # Output resolution
    output_width: int = 1920
    output_height: int = 1080

    # Initial virtual camera orientation (degrees)
    initial_yaw: float = 0.0
    initial_pitch: float = 0.0

    # Field of view (degrees) — controls zoom level
    fov: float = 90.0
    min_fov: float = 60.0
    max_fov: float = 120.0

    # Smoothing — higher = smoother but slower to react
    smoothing_factor: float = 0.85

    # How fast the camera can pan (degrees per frame)
    max_pan_speed: float = 3.0


@dataclass
class DetectionConfig:
    """Object detection parameters."""

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    player_class_id: int = 0  # COCO 'person' class
    ball_class_id: int = 32  # COCO 'sports ball' class

    # Tile-based detection for equirectangular input
    # Extract N overlapping perspective tiles, run detection, merge results
    num_tiles: int = 6
    tile_fov: float = 90.0
    tile_overlap: float = 0.15


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    # Video I/O
    input_path: str = ""
    output_path: str = "data/output/reframed.mp4"
    fps: float = 30.0

    # Processing
    detection_interval: int = 3  # Run detection every N frames (perf tradeoff)
    preview: bool = False  # Show live preview window during processing
