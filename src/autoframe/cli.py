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
        help="Directory containing Insta360 Studio project directories.",
        exists=True,
    ),
    epochs: int = typer.Option(100, "--epochs", "-e"),
    batch_size: int = typer.Option(16, "--batch-size", "-b"),
    lr: float = typer.Option(1e-4, "--lr"),
    sequence_length: int = typer.Option(60, "--seq-len"),
    sample_fps: float = typer.Option(5.0, "--sample-fps"),
    backbone: str = typer.Option("resnet18", "--backbone"),
    output_dir: str = typer.Option("runs", "--output-dir"),
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
    ),
):
    """Train the camera trajectory model.

    DATA_DIR should contain Insta360 Studio project directories (each
    containing a .insproj file with keyframes) and the corresponding
    source video files.
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
    ),
    model: Path = typer.Argument(
        ...,
        help="Path to trained model checkpoint (.pt file).",
        exists=True,
    ),
    project: Path = typer.Option(
        ..., "--project", "-p",
        help="Path to existing Insta360 Studio .insproj file (used as template).",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output .insproj path. Defaults to overwriting the project (with backup).",
    ),
    smooth: int = typer.Option(15, "--smooth"),
    keyframe_interval: int = typer.Option(
        6, "--keyframe-interval",
        help="Frames between keyframes (6 at 30fps = 5/sec).",
    ),
    project_format: str = typer.Option(
        "auto",
        "--format",
        help="Storage backend: auto|insproj|insprj (legacy sidecar not yet supported).",
    ),
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
    ),
):
    """Inject AI-predicted keyframes into an Insta360 Studio project.

    Creates a backup of the original project, then replaces the keyframe
    track with the model's predicted camera trajectory. Open the modified
    project in Insta360 Studio to preview and export.
    """
    from autoframe.config import InferenceConfig, ModelConfig
    from autoframe.pipeline import run

    if output is None:
        output = project

    config = InferenceConfig(
        input_path=str(input_video),
        project_path=str(project),
        output_path=str(output),
        smooth_window=smooth,
        keyframe_interval_frames=keyframe_interval,
        model=ModelConfig(checkpoint_path=str(model)),
        project_format=project_format,
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
def inspect_project(
    project: Path = typer.Argument(
        ...,
        help="Path to .insproj file or project directory.",
        exists=True,
    ),
    project_format: str = typer.Option(
        "auto",
        "--format",
        help="Storage backend: auto|insproj|insprj (legacy sidecar not yet supported).",
    ),
):
    """Inspect an Insta360 Studio project and display its keyframes."""
    import math
    from autoframe.storage import get_store

    store = get_store(project, project_format)
    proj = store.read(project)
    clip_info = store.get_clip_info(proj)

    console.print(f"[bold]{project.name}[/bold]")
    console.print(f"  Source:  {clip_info['name']}")
    console.print(f"  Frames:  {clip_info['source_frames']}")
    console.print(f"  FPS:     {clip_info['fps']}")
    console.print(f"  Trim:    {clip_info['left_trim']} → {clip_info['right_trim']}")

    try:
        keyframes = store.extract_keyframes(proj)
    except ValueError as e:
        console.print(f"\n  [yellow]{e}[/yellow]")
        return

    console.print(f"\n  [bold]{len(keyframes)} keyframes:[/bold]\n")
    for kf in keyframes:
        console.print(
            f"    frame {kf.time:5d} | "
            f"pan={math.degrees(kf.pan):+7.2f}° | "
            f"tilt={math.degrees(kf.tilt):+6.2f}° | "
            f"fov={math.degrees(kf.fov):5.1f}° | "
            f"dist={kf.distance:.3f}"
        )


if __name__ == "__main__":
    app()
