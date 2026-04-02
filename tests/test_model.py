"""Tests for the CameraPredictor model architecture."""

import pytest
import torch

from autoframe.model import CameraPredictor


@pytest.fixture
def model():
    return CameraPredictor(
        backbone="resnet18",
        gru_hidden_size=64,
        gru_num_layers=1,
        dropout=0.0,
    )


def test_model_output_shape(model):
    """Model should output (B, T, 4) predictions."""
    frames = torch.randn(2, 5, 3, 320, 640)
    pred, hidden = model(frames)
    assert pred.shape == (2, 5, 4)


def test_model_hidden_state_shape(model):
    frames = torch.randn(1, 3, 3, 320, 640)
    _, hidden = model(frames)
    assert hidden.shape[0] == 1  # num_layers
    assert hidden.shape[1] == 1  # batch
    assert hidden.shape[2] == 64  # hidden_size


def test_model_streaming_inference(model):
    """Hidden state should carry across sequential calls."""
    frame = torch.randn(3, 320, 640)
    result1, hidden1 = model.predict_single(frame, None)
    result2, hidden2 = model.predict_single(frame, hidden1)

    # Output should be in radians
    assert "pan_rad" in result1
    assert "tilt_rad" in result1
    assert "fov_rad" in result1
    assert not torch.equal(hidden1, hidden2)
