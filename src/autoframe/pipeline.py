"""Main processing pipeline for inference.

Loads a trained model, runs it over the input video to predict camera
trajectory, smooths the result, then renders the reframed output.
"""

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from autoframe.camera import InferenceCamera, smooth_trajectory
from autoframe.config import InferenceConfig
from autoframe.projection import equirect_to_rectilinear
from autoframe.video_io import FrameReader, VideoWriter, check_ffmpeg

console = Console()


def run(config: InferenceConfig) -> str:
    """Execute the reframing pipeline using a trained model.

    Args:
        config: Inference configuration.

    Returns:
        Path to the output video file.
    """
    if not check_ffmpeg():
        console.print("[red]FFmpeg not found.[/red] Install it: brew install ffmpeg")
        raise SystemExit(1)

    if not config.model.checkpoint_path:
        console.print(
            "[red]No model checkpoint specified.[/red]\n"
            "Train a model first with: autoframe train data/training/\n"
            "Then run: autoframe reframe video.mp4 --model runs/<run>/best.pt"
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

    # === Pass 1: Predict camera trajectory ===
    console.print("\n[bold cyan]Pass 1/2:[/bold cyan] Predicting camera trajectory...")
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
            # Resize for model input
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

    # === Pass 2: Render reframed output ===
    console.print(f"\n[bold cyan]Pass 2/2:[/bold cyan] Rendering to {config.output_path}")
    reader2 = FrameReader(config.input_path)

    with (
        VideoWriter(
            config.output_path,
            config.camera.output_width,
            config.camera.output_height,
            fps,
        ) as writer,
        Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress,
    ):
        task = progress.add_task("Rendering...", total=len(reader2))

        for i, frame in enumerate(reader2):
            crop = equirect_to_rectilinear(
                frame,
                yaws[i],
                pitches[i],
                fovs[i],
                config.camera.output_width,
                config.camera.output_height,
            )
            writer.write(crop)

            if config.preview:
                cv2.imshow("Preview", crop)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    console.print("[yellow]Cancelled by user.[/yellow]")
                    break

            progress.advance(task)

    if config.preview:
        cv2.destroyAllWindows()

    console.print(f"\n[bold green]Done![/bold green] Output: {config.output_path}")
    return config.output_path
