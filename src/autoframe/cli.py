"""CLI entry point for basketball-360-autoframe."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from autoframe import __version__

app = typer.Typer(
    name="autoframe",
    help="Auto-reframe 360° basketball video using learned camera trajectories.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"basketball-360-autoframe v{__version__}")
        raise typer.Exit()


# ── Train ────────────────────────────────────────────────────────────────


@app.command()
def train(
    data_dir: Path = typer.Argument(
        ...,
        help="Directory containing video files and their .insprj sidecar files.",
        exists=True,
    ),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs."),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size."),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate."),
    sequence_length: int = typer.Option(60, "--seq-len", help="Frames per training sample."),
    sample_fps: float = typer.Option(5.0, "--sample-fps", help="Subsample video to this FPS."),
    backbone: str = typer.Option("resnet18", "--backbone", help="CNN backbone (resnet18, resnet34, efficientnet_b0)."),
    output_dir: str = typer.Option("runs", "--output-dir", help="Directory for checkpoints and logs."),
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version.",
    ),
):
    """Train the camera trajectory model from video + .insprj pairs.

    Place your Insta360 source videos alongside their .insprj sidecar files
    (from Insta360 Studio) in DATA_DIR, then run this command.
    """
    from autoframe.config import ModelConfig, TrainingConfig
    from autoframe.train import train as run_training

    training_config = TrainingConfig(
        data_dir=str(data_dir),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        sequence_length=sequence_length,
        sample_fps=sample_fps,
        output_dir=output_dir,
    )
    model_config = ModelConfig(backbone=backbone)

    run_training(training_config, model_config)


# ── Reframe ──────────────────────────────────────────────────────────────


@app.command()
def reframe(
    input_video: Path = typer.Argument(
        ...,
        help="Path to 360° equirectangular video.",
        exists=True,
        readable=True,
    ),
    model: Path = typer.Argument(
        ...,
        help="Path to trained model checkpoint (.pt file).",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output video path. Defaults to <input>_reframed.mp4.",
    ),
    width: int = typer.Option(1920, "--width", "-w", help="Output video width."),
    height: int = typer.Option(1080, "--height", "-h", help="Output video height."),
    smooth: int = typer.Option(15, "--smooth", help="Post-prediction smoothing window (frames)."),
    preview: bool = typer.Option(False, "--preview", help="Show live preview during rendering."),
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version.",
    ),
):
    """Reframe a 360° video using a trained model.

    The model predicts where a human camera operator would point the virtual
    camera at each frame, then extracts a standard rectangular crop.
    """
    from autoframe.config import CameraConfig, InferenceConfig, ModelConfig
    from autoframe.pipeline import run

    if output is None:
        output = input_video.with_stem(input_video.stem + "_reframed")

    config = InferenceConfig(
        input_path=str(input_video),
        output_path=str(output),
        preview=preview,
        smooth_window=smooth,
        camera=CameraConfig(output_width=width, output_height=height),
        model=ModelConfig(checkpoint_path=str(model)),
    )

    run(config)


# ── Utility commands ─────────────────────────────────────────────────────


@app.command()
def info(
    input_video: Path = typer.Argument(..., help="Path to video file.", exists=True),
):
    """Show video file information."""
    from autoframe.video_io import get_video_info

    vid_info = get_video_info(str(input_video))
    console.print(f"[bold]{input_video.name}[/bold]")
    console.print(f"  Resolution: {vid_info['width']}x{vid_info['height']}")
    console.print(f"  FPS:        {vid_info['fps']:.2f}")
    console.print(f"  Frames:     {vid_info['frame_count']}")
    console.print(f"  Duration:   {vid_info['duration_sec']:.1f}s")

    ratio = vid_info["width"] / vid_info["height"] if vid_info["height"] > 0 else 0
    if 1.9 < ratio < 2.1:
        console.print("  Format:     [green]Equirectangular (2:1)[/green]")
    else:
        console.print(f"  Format:     [yellow]Aspect ratio {ratio:.2f}:1 (may not be equirectangular)[/yellow]")


@app.command()
def parse_sidecar(
    sidecar: Path = typer.Argument(..., help="Path to .insprj file.", exists=True),
):
    """Parse and display keyframes from an Insta360 .insprj sidecar file."""
    from autoframe.insta360_parser import parse_insprj

    keyframes = parse_insprj(sidecar)
    console.print(f"[bold]{sidecar.name}[/bold] — {len(keyframes)} keyframes\n")

    for kf in keyframes:
        console.print(
            f"  {kf.time_ms / 1000:.2f}s | "
            f"pan={kf.pan:+7.2f}° | "
            f"tilt={kf.tilt:+6.2f}° | "
            f"roll={kf.roll:+6.2f}° | "
            f"fov={kf.fov:.3f}rad ({kf.fov * 180 / 3.14159:.0f}°)"
        )


if __name__ == "__main__":
    app()
