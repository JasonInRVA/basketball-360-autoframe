"""CLI entry point for basketball-360-autoframe."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from autoframe import __version__
from autoframe.config import CameraConfig, DetectionConfig, PipelineConfig

app = typer.Typer(
    name="autoframe",
    help="Auto-reframe 360° basketball video into standard rectangular video.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"basketball-360-autoframe v{__version__}")
        raise typer.Exit()


@app.command()
def reframe(
    input_video: Path = typer.Argument(
        ...,
        help="Path to 360° equirectangular video (e.g., from Insta360 X4).",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output video path. Defaults to <input>_reframed.mp4.",
    ),
    model: str = typer.Option(
        "yolov8n.pt",
        "--model", "-m",
        help="YOLO model path or name (downloads automatically if not found).",
    ),
    fov: float = typer.Option(
        90.0,
        "--fov",
        help="Virtual camera field of view in degrees.",
        min=30.0,
        max=150.0,
    ),
    width: int = typer.Option(
        1920,
        "--width", "-w",
        help="Output video width.",
    ),
    height: int = typer.Option(
        1080,
        "--height", "-h",
        help="Output video height.",
    ),
    smoothing: float = typer.Option(
        0.85,
        "--smoothing", "-s",
        help="Camera smoothing factor (0=instant, 1=frozen).",
        min=0.0,
        max=0.99,
    ),
    confidence: float = typer.Option(
        0.4,
        "--confidence", "-c",
        help="Detection confidence threshold.",
        min=0.1,
        max=1.0,
    ),
    detection_interval: int = typer.Option(
        3,
        "--detect-every",
        help="Run detection every N frames (higher = faster but less accurate).",
        min=1,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Show live preview window during processing.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """Reframe a 360° basketball video into a standard rectangular video.

    Takes equirectangular video from an Insta360 X4 (or similar 360° camera),
    detects the basketball action, and outputs a smoothly-panned rectangular
    video suitable for YouTube upload.
    """
    from autoframe.pipeline import run

    if output is None:
        output = input_video.with_stem(input_video.stem + "_reframed")

    config = PipelineConfig(
        input_path=str(input_video),
        output_path=str(output),
        detection_interval=detection_interval,
        preview=preview,
        camera=CameraConfig(
            output_width=width,
            output_height=height,
            fov=fov,
            smoothing_factor=smoothing,
        ),
        detection=DetectionConfig(
            model_path=model,
            confidence_threshold=confidence,
        ),
    )

    run(config)


@app.command()
def info(
    input_video: Path = typer.Argument(
        ...,
        help="Path to video file.",
        exists=True,
    ),
):
    """Show video file information."""
    from autoframe.video_io import get_video_info

    vid_info = get_video_info(str(input_video))
    console.print(f"[bold]{input_video.name}[/bold]")
    console.print(f"  Resolution: {vid_info['width']}x{vid_info['height']}")
    console.print(f"  FPS:        {vid_info['fps']:.2f}")
    console.print(f"  Frames:     {vid_info['frame_count']}")
    console.print(f"  Duration:   {vid_info['duration_sec']:.1f}s")

    # Check if likely equirectangular (2:1 aspect ratio)
    ratio = vid_info["width"] / vid_info["height"] if vid_info["height"] > 0 else 0
    if 1.9 < ratio < 2.1:
        console.print("  Format:     [green]Equirectangular (2:1)[/green]")
    else:
        console.print(f"  Format:     [yellow]Aspect ratio {ratio:.2f}:1 (may not be equirectangular)[/yellow]")


if __name__ == "__main__":
    app()
