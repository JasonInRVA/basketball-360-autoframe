"""Configuration and defaults for the autoframe pipeline."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Neural network architecture and inference parameters."""

    # Architecture
    backbone: str = "resnet18"
    gru_hidden_size: int = 256
    gru_num_layers: int = 2
    dropout: float = 0.1

    # Input preprocessing
    frame_width: int = 640
    frame_height: int = 320

    # Trained model checkpoint
    checkpoint_path: str = ""


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    data_dir: str = "data/training"
    val_split: float = 0.15
    sample_fps: float = 5.0
    sequence_length: int = 60

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"

    # Loss weights
    weight_yaw: float = 2.0
    weight_pitch: float = 1.0
    weight_fov: float = 1.0

    # Augmentation
    horizontal_flip: bool = True
    temporal_jitter: bool = True

    # Output
    output_dir: str = "runs"
    save_every: int = 10


@dataclass
class InferenceConfig:
    """Config for running inference (generating keyframes for a project)."""

    model: ModelConfig = field(default_factory=ModelConfig)

    input_path: str = ""  # Source equirectangular video
    project_path: str = ""  # Existing .insproj to modify
    output_path: str = ""  # Where to save modified .insproj

    # Post-processing smoothing window (frames)
    smooth_window: int = 15

    # Keyframe density: frames between keyframes in output.
    # 6 frames at 30fps = 5 keyframes/second.
    keyframe_interval_frames: int = 6
