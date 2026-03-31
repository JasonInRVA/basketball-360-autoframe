"""Training loop for the camera trajectory model.

Behavioral cloning: supervised regression from equirectangular frames
to camera parameters, learning to replicate human camera operation.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from torch.utils.data import DataLoader, random_split

from autoframe.config import ModelConfig, TrainingConfig
from autoframe.dataset import TrajectoryDataset
from autoframe.model import CameraPredictor

console = Console()


class WeightedCameraLoss(nn.Module):
    """Smooth-L1 loss with per-component weighting.

    Allows emphasizing yaw accuracy (most important for basketball
    side-to-side camera tracking) over pitch and FOV.
    """

    def __init__(self, weight_yaw: float = 2.0, weight_pitch: float = 1.0, weight_fov: float = 1.0):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        # Weights for [sin_yaw, cos_yaw, pitch, fov]
        self.register_buffer(
            "weights",
            torch.tensor([weight_yaw, weight_yaw, weight_pitch, weight_fov]),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_component = self.loss_fn(pred, target)  # (B, T, 4)
        weighted = per_component * self.weights
        return weighted.mean()


def train(
    training_config: TrainingConfig,
    model_config: ModelConfig,
) -> str:
    """Run the full training loop.

    Args:
        training_config: Training hyperparameters.
        model_config: Model architecture config.

    Returns:
        Path to the best checkpoint.
    """
    device = _get_device()
    console.print(f"[bold]Device:[/bold] {device}")

    # === Data ===
    console.print(f"\n[bold cyan]Loading data from {training_config.data_dir}...[/bold cyan]")
    full_dataset = TrajectoryDataset(
        data_dir=training_config.data_dir,
        sequence_length=training_config.sequence_length,
        sample_fps=training_config.sample_fps,
        frame_width=model_config.frame_width,
        frame_height=model_config.frame_height,
        augment=True,
    )

    # Train/val split
    val_size = max(1, int(len(full_dataset) * training_config.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Disable augmentation for validation
    val_dataset.dataset = TrajectoryDataset(
        data_dir=training_config.data_dir,
        sequence_length=training_config.sequence_length,
        sample_fps=training_config.sample_fps,
        frame_width=model_config.frame_width,
        frame_height=model_config.frame_height,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    console.print(f"  Train samples: {train_size}")
    console.print(f"  Val samples:   {val_size}")
    console.print(f"  Sequence len:  {training_config.sequence_length} frames")

    # === Model ===
    model = CameraPredictor(
        backbone=model_config.backbone,
        gru_hidden_size=model_config.gru_hidden_size,
        gru_num_layers=model_config.gru_num_layers,
        dropout=model_config.dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"  Parameters:    {trainable:,} trainable / {total:,} total")

    # === Optimizer & scheduler ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    if training_config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )

    criterion = WeightedCameraLoss(
        weight_yaw=training_config.weight_yaw,
        weight_pitch=training_config.weight_pitch,
        weight_fov=training_config.weight_fov,
    ).to(device)

    # === Training loop ===
    output_dir = Path(training_config.output_dir) / f"run_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configs for reproducibility
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config.__dict__, indent=2)
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config.__dict__, indent=2)
    )

    best_val_loss = float("inf")
    best_checkpoint = ""
    history: list[dict] = []

    console.print(f"\n[bold cyan]Training for {training_config.epochs} epochs...[/bold cyan]")
    console.print(f"  Output: {output_dir}\n")

    for epoch in range(1, training_config.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        for frames, targets in train_loader:
            frames = frames.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions, _ = model(frames)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(train_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        yaw_errors: list[float] = []

        with torch.no_grad():
            for frames, targets in val_loader:
                frames = frames.to(device)
                targets = targets.to(device)

                predictions, _ = model(frames)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_batches += 1

                # Compute yaw error in degrees for interpretability
                pred_yaw = torch.atan2(predictions[..., 0], predictions[..., 1])
                true_yaw = torch.atan2(targets[..., 0], targets[..., 1])
                diff = torch.abs(pred_yaw - true_yaw)
                diff = torch.min(diff, 2 * np.pi - diff)  # Handle wraparound
                yaw_errors.extend(torch.degrees(diff).flatten().cpu().numpy().tolist())

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_yaw_error = np.mean(yaw_errors) if yaw_errors else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "yaw_error_deg": avg_yaw_error,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # --- Logging ---
        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = str(output_dir / "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "model_config": model_config.__dict__,
            }, best_checkpoint)
            marker = " [green]*best*[/green]"

        console.print(
            f"  Epoch {epoch:3d}/{training_config.epochs} | "
            f"train {avg_train_loss:.4f} | "
            f"val {avg_val_loss:.4f} | "
            f"yaw err {avg_yaw_error:.1f}°{marker}"
        )

        # Periodic checkpoint
        if epoch % training_config.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "model_config": model_config.__dict__,
            }, str(output_dir / f"epoch_{epoch:03d}.pt"))

    # Save training history
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Best val loss: {best_val_loss:.4f}")
    console.print(f"  Best checkpoint: {best_checkpoint}")

    return best_checkpoint


def _get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
