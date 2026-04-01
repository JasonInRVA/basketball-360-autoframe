"""Model-based camera controller for inference.

Loads a trained CameraPredictor checkpoint and runs frame-by-frame
inference to produce a camera trajectory, with optional post-hoc
smoothing.
"""

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from autoframe.model import CameraPredictor


class InferenceCamera:
    """Runs the trained model to predict camera trajectory for a video."""

    def __init__(self, checkpoint_path: str, device: torch.device | None = None):
        if device is None:
            device = _get_device()
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = checkpoint["model_config"]

        self.model = CameraPredictor(
            backbone=model_cfg["backbone"],
            gru_hidden_size=model_cfg["gru_hidden_size"],
            gru_num_layers=model_cfg["gru_num_layers"],
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self._hidden: torch.Tensor | None = None

    def reset(self):
        """Reset GRU hidden state (call at start of each video)."""
        self._hidden = None

    def predict_frame(self, frame: np.ndarray) -> dict[str, float]:
        """Predict camera params for a single frame.

        Args:
            frame: Preprocessed frame as (H, W, 3) uint8 numpy array.

        Returns:
            Dict with yaw_deg, pitch_deg, fov_deg.
        """
        import cv2

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        frame_t = frame_t.to(self.device)

        result, self._hidden = self.model.predict_single(frame_t, self._hidden)
        return result


def smooth_trajectory(
    yaws: list[float],
    pitches: list[float],
    fovs: list[float],
    window: int = 15,
) -> tuple[list[float], list[float], list[float]]:
    """Post-process smooth pass over the full trajectory.

    Uses circular smoothing for yaw to handle wraparound.
    """
    rads = np.radians(yaws)
    sin_smooth = uniform_filter1d(np.sin(rads), size=window)
    cos_smooth = uniform_filter1d(np.cos(rads), size=window)
    smooth_yaws = np.degrees(np.arctan2(sin_smooth, cos_smooth)).tolist()

    smooth_pitches = uniform_filter1d(np.array(pitches, dtype=float), size=window).tolist()
    smooth_fovs = uniform_filter1d(np.array(fovs, dtype=float), size=window).tolist()

    return smooth_yaws, smooth_pitches, smooth_fovs


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
