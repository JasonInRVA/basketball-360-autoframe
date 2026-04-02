"""Backend-agnostic access to camera trajectories.

This module provides a thin abstraction over storage formats that can hold
Insta360 camera keyframes. Currently implemented:
  - `insproj` (Insta360 Studio project JSON, timeline-based) via insta360_parser

Planned (not yet implemented):
  - `insprj` legacy XML sidecar
"""

from pathlib import Path
from typing import Protocol, Literal, Any

from autoframe import insta360_parser

Backend = Literal["insproj", "insprj"]


class CameraStore(Protocol):
    def read(self, path: str | Path) -> Any: ...
    def extract_keyframes(self, project: Any, clip_index: int = 0): ...
    def get_clip_info(self, project: Any, clip_index: int = 0) -> dict: ...
    def inject_keyframes(self, project: Any, keyframes, clip_index: int = 0): ...
    def save(self, project: Any, output_path: str | Path, backup: bool = True): ...


class InsprojStore:
    """Studio project JSON backend."""

    def read(self, path: str | Path):
        return insta360_parser.read_project(path)

    def extract_keyframes(self, project: dict, clip_index: int = 0):
        return insta360_parser.extract_keyframes(project, clip_index)

    def get_clip_info(self, project: dict, clip_index: int = 0) -> dict:
        return insta360_parser.get_clip_info(project, clip_index)

    def inject_keyframes(self, project: dict, keyframes, clip_index: int = 0):
        return insta360_parser.inject_keyframes(project, keyframes, clip_index)

    def save(self, project: dict, output_path: str | Path, backup: bool = True):
        return insta360_parser.save_project(project, output_path, backup)


def detect_backend(path: str | Path, preferred: Backend | Literal["auto"] = "auto") -> Backend:
    """Choose backend based on hint or file extension."""
    if preferred and preferred != "auto":
        return preferred  # trust caller
    suffix = Path(path).suffix.lower()
    if suffix == ".insprj":
        return "insprj"
    return "insproj"


def get_store(path: str | Path, preferred: Backend | Literal["auto"] = "auto") -> CameraStore:
    backend = detect_backend(path, preferred)
    if backend == "insproj":
        return InsprojStore()
    raise NotImplementedError(
        "Legacy .insprj XML sidecar backend is not yet implemented. "
        "Use .insproj projects (Studio timeline) for now."
    )
