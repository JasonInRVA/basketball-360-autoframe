"""Main processing pipeline for inference.

Single-pass pipeline: reads equirectangular video frames, runs the trained
model to predict camera trajectory, smooths the result, and writes an
.insprj sidecar file. The actual video rendering is handled by Insta360
Studio, which reads the sidecar and produces the final output with full
quality, audio, lens correction, and stabilization.
"""

import cv2
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from autoframe.camera import InferenceCamera, smooth_trajectory
from autoframe.config import InferenceConfig
from autoframe.insta360_parser import trajectory_to_keyframes, write_insprj
from autoframe.video_io import FrameReader

console = Console()


def run(config: InferenceConfig) -> str:
    """Execute the inference pipeline.

    Reads video → predicts camera trajectory → writes .insprj sidecar.

    Args:
        config: Inference configuration.

    Returns:
        Path to the output .insprj file.
    """
    if not config.model.checkpoint_path:
        console.print(
            "[red]No model checkpoint specified.[/red]\n"
            "Train a model first with: autoframe train data/training/\n"
            "Then run: autoframe reframe video.mp4 runs/<run>/best.pt"
        )
        raise SystemExit(1)

    console.print(f"[bold]Input:[/bold]  {config.input_path}")
    console.print(f"[bold]Model:[/bold]  {config.model.checkpoint_path}")

    # Initialize
    reader = FrameReader(config.input_path)
    camera = InferenceCamera(config.model.checkpoint_path)
    camera.reset()

    fps = reader.info["fps"]
    console.print(
        f"[dim]Video: {reader.info['width']}x{reader.info['height']} "
        f"@ {fps:.1f}fps, "
        f"{reader.info['frame_count']} frames "
        f"({reader.info['duration_sec']:.1f}s)[/dim]"
    )

    # === Predict camera trajectory ===
    console.print("\n[bold cyan]Predicting camera trajectory...[/bold cyan]")
    raw_yaws: list[float] = []
    raw_pitches: list[float] = []
    raw_fovs: list[float] = []

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Predicting...", total=len(reader))

        for frame in reader:
            small = cv2.resize(
                frame,
                (config.model.frame_width, config.model.frame_height),
            )
            params = camera.predict_frame(small)
            raw_yaws.append(params["yaw_deg"])
            raw_pitches.append(params["pitch_deg"])
            raw_fovs.append(params["fov_deg"])
            progress.advance(task)

    # === Smooth trajectory ===
    console.print("[dim]Smoothing trajectory...[/dim]")
    yaws, pitches, fovs = smooth_trajectory(
        raw_yaws, raw_pitches, raw_fovs, config.smooth_window
    )

    # === Write .insprj sidecar ===
    keyframes = trajectory_to_keyframes(
        yaw_deg=yaws,
        pitch_deg=pitches,
        fov_deg=fovs,
        fps=fps,
        keyframe_interval_ms=config.keyframe_interval_ms,
    )

    output_path = write_insprj(
        keyframes=keyframes,
        output_path=config.output_path,
        source_video_path=config.input_path,
    )

    console.print(f"\n[bold green]Done![/bold green] Sidecar: {output_path}")
    console.print(
        f"[dim]Generated {len(keyframes)} keyframes "
        f"({config.keyframe_interval_ms:.0f}ms intervals)[/dim]"
    )
    console.print(
        "\n[bold]Next step:[/bold] Open Insta360 Studio, import your source video,\n"
        "load this sidecar file, and export the reframed video."
    )

    return str(output_path)
