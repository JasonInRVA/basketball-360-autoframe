"""Main processing pipeline for inference.

Single-pass pipeline: reads equirectangular video frames, runs the trained
model to predict camera trajectory, smooths the result, and injects the
keyframes into an Insta360 Studio project file (.insproj).

The actual video rendering is handled by Insta360 Studio.
"""

import cv2
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from autoframe.camera import InferenceCamera, smooth_trajectory
from autoframe.config import InferenceConfig
from autoframe.storage import get_store
from autoframe.insta360_parser import trajectory_to_keyframes
from autoframe.video_io import FrameReader

console = Console()


def run(config: InferenceConfig) -> str:
    """Execute the inference pipeline.

    Reads video, predicts camera trajectory, injects keyframes into
    an existing Insta360 Studio project.

    Args:
        config: Inference configuration.

    Returns:
        Path to the modified .insproj file.
    """
    if not config.model.checkpoint_path:
        console.print(
            "[red]No model checkpoint specified.[/red]\n"
            "Train a model first with: autoframe train data/training/\n"
            "Then run: autoframe reframe video.mp4 runs/<run>/best.pt --project <project>"
        )
        raise SystemExit(1)

    if not config.project_path:
        console.print(
            "[red]No Insta360 Studio project specified.[/red]\n"
            "Create a project in Insta360 Studio with your source video,\n"
            "then pass it with: --project /path/to/project.insproj"
        )
        raise SystemExit(1)

    console.print(f"[bold]Input:[/bold]   {config.input_path}")
    console.print(f"[bold]Model:[/bold]   {config.model.checkpoint_path}")
    console.print(f"[bold]Project:[/bold] {config.project_path}")
    console.print(f"[bold]Format:[/bold]  {config.project_format}")

    # Load the existing project as template via store
    store = get_store(config.project_path, config.project_format)
    project = store.read(config.project_path)

    # Initialize video reader and model
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
    raw_pans: list[float] = []
    raw_tilts: list[float] = []
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
            # Model outputs radians (matching .insproj format)
            raw_pans.append(params["pan_rad"])
            raw_tilts.append(params["tilt_rad"])
            raw_fovs.append(params["fov_rad"])
            progress.advance(task)

    # === Smooth trajectory ===
    console.print("[dim]Smoothing trajectory...[/dim]")
    pans, tilts, fovs = smooth_trajectory(
        raw_pans, raw_tilts, raw_fovs, config.smooth_window
    )

    # === Convert to keyframes and inject into project ===
    keyframes = trajectory_to_keyframes(
        pan_rad=pans,
        tilt_rad=tilts,
        fov_rad=fovs,
        fps=fps,
        keyframe_interval_frames=config.keyframe_interval_frames,
    )

    store.inject_keyframes(project, keyframes)

    output_path = store.save(project, config.output_path)

    console.print(f"\n[bold green]Done![/bold green] Modified project: {output_path}")
    console.print(
        f"[dim]Injected {len(keyframes)} keyframes "
        f"(every {config.keyframe_interval_frames} frames)[/dim]"
    )
    console.print(
        "\n[bold]Next step:[/bold] Open the project in Insta360 Studio and export."
    )

    return str(output_path)
