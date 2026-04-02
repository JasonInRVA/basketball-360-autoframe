"""PyTorch Dataset for training the camera trajectory model.

Pairs equirectangular video frames with per-frame camera parameters
extracted from Insta360 Studio project files.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from autoframe.insta360_parser import (
    extract_keyframes,
    find_insproj_in_directory,
    get_clip_info,
    interpolate_trajectory,
    read_project,
)
from autoframe.video_io import get_video_info


class TrajectoryDataset(Dataset):
    """Dataset of (frame_sequence, camera_params) pairs for behavioral cloning.

    Each sample is a fixed-length sequence of frames paired with the
    corresponding camera parameters (yaw encoded as sin/cos, pitch, fov).
    All angle values from the .insproj are in radians.
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
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.sample_fps = sample_fps
        self.frame_size = (frame_width, frame_height)
        self.augment = augment

        self.clips: list[dict] = []
        self._discover_clips()

        self.samples: list[tuple[int, int]] = []
        self._build_sample_index()

    def _discover_clips(self):
        """Find all video files that have matching .insproj project files.

        Expects either:
          - Project directories alongside videos
          - .insproj files alongside videos
          - Videos inside project directories
        """
        video_extensions = {".mp4", ".insv", ".mov", ".avi"}

        # Strategy 1: Look for project directories containing .insproj files
        for item in sorted(self.data_dir.iterdir()):
            if item.is_dir():
                insproj = find_insproj_in_directory(item)
                if insproj is None:
                    continue
                self._try_add_clip_from_project(insproj)

        # Strategy 2: Look for .insproj files directly
        for insproj in sorted(self.data_dir.glob("*.insproj")):
            if "_backup" in insproj.name:
                continue
            self._try_add_clip_from_project(insproj)

        if not self.clips:
            raise FileNotFoundError(
                f"No video + .insproj pairs found in {self.data_dir}. "
                "Place Insta360 Studio project directories (containing .insproj files) "
                "in the data directory."
            )

    def _try_add_clip_from_project(self, insproj_path: Path):
        """Try to add a clip from a project file."""
        try:
            project = read_project(insproj_path)
            clip_info = get_clip_info(project)
            keyframes = extract_keyframes(project)
        except (ValueError, FileNotFoundError):
            return

        total_frames = clip_info["source_frames"]
        if total_frames == 0 or len(keyframes) < 2:
            return

        trajectory = interpolate_trajectory(keyframes, total_frames)
        fps = clip_info["fps"]
        frame_step = max(1, round(fps / self.sample_fps))

        # Try to find the source video
        video_path = self._find_source_video(clip_info, insproj_path)
        if video_path is None:
            return

        self.clips.append({
            "video_path": str(video_path),
            "trajectory": trajectory,  # (N, 3): pan_rad, tilt_rad, fov_rad
            "fps": fps,
            "frame_count": total_frames,
            "frame_step": frame_step,
            "left_trim": clip_info["left_trim"],
            "right_trim": clip_info["right_trim"],
        })

    def _find_source_video(self, clip_info: dict, insproj_path: Path) -> Path | None:
        """Try to locate the source video file."""
        # Check the URL in the clip info
        url = clip_info.get("url", "")
        if url:
            # URL format is like "/path/to/file.insvVID_012.insv" — split on .insv
            for candidate in [Path(url), Path(url.split(".insv")[0] + ".insv")]:
                if candidate.exists():
                    return candidate

        # Look for video files near the project file
        video_extensions = {".mp4", ".insv", ".mov"}
        search_dirs = [insproj_path.parent, insproj_path.parent.parent]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for ext in video_extensions:
                for video in search_dir.glob(f"*{ext}"):
                    return video

        return None

    def _build_sample_index(self):
        for clip_idx, clip in enumerate(self.clips):
            step = clip["frame_step"]
            n_subsampled = clip["frame_count"] // step
            stride = max(1, self.sequence_length // 2)
            for start in range(0, n_subsampled - self.sequence_length, stride):
                self.samples.append((clip_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a (frames, targets) pair.

        frames: (T, 3, H, W) float32 tensor, normalized to [0, 1].
        targets: (T, 4) float32 tensor — [sin(pan), cos(pan), tilt, fov].
            Pan (yaw) encoded as sin/cos for circular continuity.
            Tilt and FOV in radians (matching .insproj).
        """
        clip_idx, start = self.samples[idx]
        clip = self.clips[clip_idx]
        step = clip["frame_step"]

        frame_indices = [(start + t) * step for t in range(self.sequence_length)]

        # Read frames
        cap = cv2.VideoCapture(clip["video_path"])
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                frame = frames[-1] if frames else np.zeros(
                    (self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8
                )
            else:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Extract trajectory targets (already in radians)
        trajectory = clip["trajectory"]  # (N, 3): pan_rad, tilt_rad, fov_rad
        targets = []
        for fi in frame_indices:
            fi_clamped = min(fi, len(trajectory) - 1)
            pan_rad, tilt_rad, fov_rad = trajectory[fi_clamped]
            targets.append([
                np.sin(pan_rad),
                np.cos(pan_rad),
                tilt_rad,
                fov_rad,
            ])

        frames_arr = np.stack(frames)
        targets_arr = np.array(targets, dtype=np.float32)

        # Augmentation: horizontal flip
        if self.augment and np.random.random() > 0.5:
            frames_arr = frames_arr[:, :, ::-1, :].copy()
            targets_arr[:, 0] *= -1  # negate sin(pan)

        frames_t = torch.from_numpy(frames_arr).float().permute(0, 3, 1, 2) / 255.0
        targets_t = torch.from_numpy(targets_arr)

        return frames_t, targets_t
