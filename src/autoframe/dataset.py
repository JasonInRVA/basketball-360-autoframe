"""PyTorch Dataset for training the camera trajectory model.

Pairs equirectangular video frames with per-frame camera parameters
extracted from Insta360 .insprj sidecar files. Handles subsampling,
temporal windowing, and augmentation.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from autoframe.insta360_parser import (
    find_sidecar,
    interpolate_trajectory,
    parse_insprj,
)
from autoframe.video_io import get_video_info


class TrajectoryDataset(Dataset):
    """Dataset of (frame_sequence, camera_params) pairs for behavioral cloning.

    Each sample is a fixed-length sequence of frames paired with the
    corresponding camera parameters (yaw encoded as sin/cos, pitch, fov).
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 60,
        sample_fps: float = 5.0,
        frame_width: int = 640,
        frame_height: int = 320,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing video files and their .insprj sidecars.
            sequence_length: Number of frames per training sample.
            sample_fps: Subsampled frame rate (lower = faster training).
            frame_width: Resize frames to this width.
            frame_height: Resize frames to this height.
            augment: Enable data augmentation (horizontal flip, jitter).
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.sample_fps = sample_fps
        self.frame_size = (frame_width, frame_height)
        self.augment = augment

        # Discover video + sidecar pairs
        self.clips: list[dict] = []
        self._discover_clips()

        # Build index of all possible (clip_idx, start_frame) samples
        self.samples: list[tuple[int, int]] = []
        self._build_sample_index()

    def _discover_clips(self):
        """Find all video files that have matching .insprj sidecars."""
        video_extensions = {".mp4", ".insv", ".mov", ".avi"}
        for video_path in sorted(self.data_dir.iterdir()):
            if video_path.suffix.lower() not in video_extensions:
                continue
            sidecar = find_sidecar(video_path)
            if sidecar is None:
                continue

            info = get_video_info(str(video_path))
            keyframes = parse_insprj(sidecar)
            trajectory = interpolate_trajectory(
                keyframes, info["fps"], info["frame_count"]
            )

            # Compute frame step for subsampling
            frame_step = max(1, round(info["fps"] / self.sample_fps))

            self.clips.append({
                "video_path": str(video_path),
                "trajectory": trajectory,  # (N, 3): yaw_deg, pitch_deg, fov_deg
                "fps": info["fps"],
                "frame_count": info["frame_count"],
                "frame_step": frame_step,
            })

        if not self.clips:
            raise FileNotFoundError(
                f"No video + .insprj pairs found in {self.data_dir}. "
                "Place source videos and their .insprj sidecar files in the same directory."
            )

    def _build_sample_index(self):
        """Create list of (clip_idx, start_frame) for all valid windows."""
        for clip_idx, clip in enumerate(self.clips):
            step = clip["frame_step"]
            # Number of subsampled frames available
            n_subsampled = clip["frame_count"] // step
            # Sliding window with 50% overlap
            stride = max(1, self.sequence_length // 2)
            for start in range(0, n_subsampled - self.sequence_length, stride):
                self.samples.append((clip_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a (frames, targets) pair.

        frames: (T, 3, H, W) float32 tensor, normalized to [0, 1].
        targets: (T, 4) float32 tensor — [sin(yaw), cos(yaw), pitch, fov].
        """
        clip_idx, start = self.samples[idx]
        clip = self.clips[clip_idx]
        step = clip["frame_step"]

        # Frame indices in the original video
        frame_indices = [
            (start + t) * step for t in range(self.sequence_length)
        ]

        # Read frames
        cap = cv2.VideoCapture(clip["video_path"])
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                # Pad with last good frame if we run off the end
                frame = frames[-1] if frames else np.zeros(
                    (self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8
                )
            else:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Extract corresponding trajectory targets
        trajectory = clip["trajectory"]  # (N, 3): yaw_deg, pitch_deg, fov_deg
        targets = []
        for fi in frame_indices:
            fi_clamped = min(fi, len(trajectory) - 1)
            yaw_deg, pitch_deg, fov_deg = trajectory[fi_clamped]
            yaw_rad = np.radians(yaw_deg)
            targets.append([
                np.sin(yaw_rad),
                np.cos(yaw_rad),
                pitch_deg,
                fov_deg,
            ])

        frames_arr = np.stack(frames)  # (T, H, W, 3)
        targets_arr = np.array(targets, dtype=np.float32)  # (T, 4)

        # Augmentation: horizontal flip (mirrors court, negates yaw)
        if self.augment and np.random.random() > 0.5:
            frames_arr = frames_arr[:, :, ::-1, :].copy()  # flip width
            targets_arr[:, 0] *= -1  # negate sin(yaw)
            # cos(yaw) stays the same — this correctly mirrors the angle

        # Convert to tensors
        frames_t = torch.from_numpy(frames_arr).float().permute(0, 3, 1, 2) / 255.0
        targets_t = torch.from_numpy(targets_arr)

        return frames_t, targets_t
