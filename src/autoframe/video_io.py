"""Video reading utilities.

Handles reading equirectangular frames from Insta360 X4 output for
model training and inference. Video writing/rendering is handled by
Insta360 Studio — we inject keyframes into its project files and let
Studio render.
"""

from pathlib import Path

import cv2
import numpy as np


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
