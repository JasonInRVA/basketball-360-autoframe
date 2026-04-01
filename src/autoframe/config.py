"""Configuration and defaults for the autoframe pipeline."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Neural network architecture and inference parameters."""

    # Architecture
    backbone: str = "resnet18"  # resnet18, resnet34, efficientnet_b0
    gru_hidden_size: int = 256
    gru_num_layers: int = 2
    dropout: float = 0.1

    # Input preprocessing — frames are downsampled to this size for the CNN
    frame_width: int = 640
    frame_height: int = 320

    # Trained model checkpoint
    checkpoint_path: str = ""


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Data
    data_dir: str = "data/training"  # Contains video + .insprj pairs
    val_split: float = 0.15
    sample_fps: float = 5.0  # Subsample from video FPS (camera motion is smooth)
    sequence_length: int = 60  # Frames of context per training sample

    # Optimization
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"  # cosine, step

    # Loss weights (yaw is most important for basketball side-to-side action)
    weight_yaw: float = 2.0
    weight_pitch: float = 1.0
    weight_fov: float = 1.0

    # Augmentation
    horizontal_flip: bool = True
    temporal_jitter: bool = True

    # Output
    output_dir: str = "runs"
    save_every: int = 10  # Save checkpoint every N epochs


@dataclass
class InferenceConfig:
    """Top-level config for running inference (generating a sidecar)."""

    model: ModelConfig = field(default_factory=ModelConfig)

    input_path: str = ""
    output_path: str = ""  # .insprj output path (defaults to <input>.insprj)

    # Post-processing smoothing window (frames). Applied after model prediction.
    smooth_window: int = 15

    # Keyframe density in the output .insprj (milliseconds between keyframes).
    # 200ms = 5 keyframes/sec — dense enough for smooth motion, sparse enough
    # for Insta360 Studio to handle comfortably.
    keyframe_interval_ms: float = 200.0
