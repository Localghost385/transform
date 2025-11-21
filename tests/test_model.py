import os
import torch
import pytest
from src.model import DrumTransformer

@pytest.fixture
def dummy_input():
    B, T, D = 2, 512, 23
    x = torch.randint(0, 2, (B, T, D)).float()
    y = torch.randint(0, 2, (B, T, D)).float()
    return x, y

def test_forward_output_shape(dummy_input):
    x, _ = dummy_input
    model = DrumTransformer(num_classes=23, seq_len=512)
    out = model(x)
    # Check output shape
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    # Probabilities are in [0,1]
    assert torch.all((out >= 0) & (out <= 1)), "Output not in [0,1] range"

def test_loss_computation(dummy_input):
    x, y = dummy_input
    model = DrumTransformer(num_classes=23, seq_len=512)
    preds = model(x)
    loss = model.compute_loss(preds, y)
    # Loss should be scalar and positive
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_parameter_count():
    model = DrumTransformer(num_classes=23, seq_len=512)
    n_params = model.count_parameters()
    assert isinstance(n_params, int)
    assert n_params > 0

def test_save_load(tmp_path, dummy_input):
    x, _ = dummy_input
    model = DrumTransformer(num_classes=23, seq_len=512)
    # Save model
    save_path = tmp_path / "test_model.pt"
    model.save(save_path)
    assert os.path.exists(save_path)
    # Load into new model
    new_model = DrumTransformer(num_classes=23, seq_len=512)
    new_model.load(save_path)
    # Forward pass with loaded model should give same shape
    out = new_model(x)
    assert out.shape == x.shape

def test_autoregressive_mask_dummy(dummy_input):
    # Simple test: forward works with seq_len < max
    x, _ = dummy_input
    short_seq = x[:, :128, :]
    model = DrumTransformer(num_classes=23, seq_len=512)
    out = model(short_seq)
    assert out.shape == short_seq.shape
