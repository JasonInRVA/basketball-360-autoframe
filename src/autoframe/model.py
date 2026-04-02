"""Neural network for camera trajectory prediction.

Architecture: pretrained CNN backbone (feature extractor) → GRU (temporal
context) → linear head → (sin_yaw, cos_yaw, pitch, fov).

The model learns to predict where a human camera operator would point the
virtual camera given the current equirectangular frame and recent history.
Yaw is encoded as (sin, cos) to avoid discontinuity at the ±180° boundary.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CameraPredictor(nn.Module):
    """Predicts virtual camera parameters from equirectangular video frames.

    Input: sequence of equirectangular frames (B, T, C, H, W)
    Output: camera params per frame (B, T, 4) — [sin_yaw, cos_yaw, pitch, fov]
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # === CNN backbone (feature extractor) ===
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.feature_dim = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.feature_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze early layers — only fine-tune later blocks
        self._freeze_early_layers()

        # === Temporal model (GRU) ===
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        # === Prediction head ===
        # Output: sin(yaw), cos(yaw), pitch, fov = 4 values
        self.head = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def _freeze_early_layers(self):
        """Freeze the first ~75% of backbone layers.

        We only fine-tune the last residual block, since early features
        (edges, textures) transfer well from ImageNet.
        """
        all_params = list(self.backbone.parameters())
        freeze_count = int(len(all_params) * 0.75)
        for param in all_params[:freeze_count]:
            param.requires_grad = False

    def forward(
        self,
        frames: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            frames: (B, T, C, H, W) — batch of frame sequences.
            hidden: Optional GRU hidden state for streaming inference.

        Returns:
            predictions: (B, T, 4) — [sin_yaw, cos_yaw, pitch, fov] per frame.
            hidden: Updated GRU hidden state.
        """
        B, T, C, H, W = frames.shape

        # Extract per-frame features with CNN
        x = frames.reshape(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, feature_dim)
        features = features.reshape(B, T, self.feature_dim)

        # Temporal modeling
        gru_out, hidden = self.gru(features, hidden)  # (B, T, hidden_size)

        # Predict camera params
        predictions = self.head(gru_out)  # (B, T, 4)

        return predictions, hidden

    def predict_single(
        self,
        frame: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Convenience method for single-frame inference.

        Args:
            frame: (C, H, W) single frame tensor.
            hidden: GRU hidden state from previous frame.

        Returns:
            Dict with pan_rad, tilt_rad, fov_rad (all radians) and
            updated hidden state.
        """
        self.eval()
        with torch.no_grad():
            x = frame.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
            pred, hidden = self.forward(x, hidden)
            pred = pred.squeeze()  # (4,)

            sin_pan, cos_pan, tilt, fov = pred.cpu().numpy()
            pan_rad = float(torch.atan2(pred[0], pred[1]).cpu().numpy())

            return {
                "pan_rad": pan_rad,
                "tilt_rad": float(tilt),
                "fov_rad": float(fov),
            }, hidden
