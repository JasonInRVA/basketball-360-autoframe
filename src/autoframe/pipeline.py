"""Main processing pipeline — ties everything together.

Reads equirectangular video → detects action → computes virtual camera path
→ extracts reframed perspective crops → writes output video.
"""

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from autoframe.camera import VirtualCamera, smooth_trajectory
from autoframe.config import PipelineConfig
from autoframe.detector import Detection, TiledDetector
from autoframe.projection import equirect_to_rectilinear
from autoframe.video_io import FrameReader, VideoWriter, check_ffmpeg

console = Console()


def run(config: PipelineConfig) -> str:
    """Execute the full reframing pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Path to the output video file.
    """
    # Preflight checks
    if not check_ffmpeg():
        console.print("[red]FFmpeg not found.[/red] Install it: brew install ffmpeg")
        raise SystemExit(1)

    console.print(f"[bold]Input:[/bold] {config.input_path}")

    # Initialize components
    reader = FrameReader(config.input_path)
    detector = TiledDetector(config.detection)
    camera = VirtualCamera(config.camera)

    console.print(
        f"[dim]Video: {reader.info['width']}x{reader.info['height']} "
        f"@ {reader.info['fps']:.1f}fps, "
        f"{reader.info['frame_count']} frames "
        f"({reader.info['duration_sec']:.1f}s)[/dim]"
    )

    fps = reader.info["fps"] or config.fps

    # === Pass 1: Detect and compute raw camera trajectory ===
    console.print("\n[bold cyan]Pass 1/2:[/bold cyan] Detecting action...")
    raw_yaws: list[float] = []
    raw_pitches: list[float] = []
    raw_fovs: list[float] = []
    last_detections: list[Detection] = []

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Detecting...", total=len(reader))

        for i, frame in enumerate(reader):
            if i % config.detection_interval == 0:
                last_detections = detector.detect(frame)

            yaw, pitch, fov = camera.update(last_detections)
            raw_yaws.append(yaw)
            raw_pitches.append(pitch)
            raw_fovs.append(fov)

            progress.advance(task)

    # === Smooth the trajectory ===
    console.print("[dim]Smoothing camera trajectory...[/dim]")
    smooth_window = max(1, int(fps / 2))  # ~0.5 second window
    yaws, pitches, fovs = smooth_trajectory(raw_yaws, raw_pitches, raw_fovs, smooth_window)

    # === Pass 2: Extract reframed crops ===
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
                import cv2

                cv2.imshow("Preview", crop)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    console.print("[yellow]Preview cancelled by user.[/yellow]")
                    break

            progress.advance(task)

    if config.preview:
        import cv2

        cv2.destroyAllWindows()

    console.print(f"\n[bold green]Done![/bold green] Output: {config.output_path}")
    return config.output_path
