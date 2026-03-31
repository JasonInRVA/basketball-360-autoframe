"""Video reading and writing utilities.

Handles reading equirectangular frames from Insta360 X4 output and
writing the final reframed video.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

console = Console()


def check_ffmpeg() -> bool:
    """Verify FFmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_video_info(path: str) -> dict:
    """Get video metadata using OpenCV.

    Returns:
        Dict with keys: width, height, fps, frame_count, duration_sec.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


class FrameReader:
    """Reads frames from a video file one at a time."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.info = get_video_info(path)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def __len__(self) -> int:
        return self.info["frame_count"]

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """Writes frames to a video file using FFmpeg for better compression."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray):
        """Write a single frame."""
        self._proc.stdin.write(frame.tobytes())

    def close(self):
        """Finalize the video file."""
        self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            err = self._proc.stderr.read().decode()
            console.print(f"[red]FFmpeg error:[/red] {err}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
