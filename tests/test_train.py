import os
import sys
import numpy as np
import shutil
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.dataset import DrumDataset, get_dataloader
from src.model import DrumTransformer
from src.train import train_epoch, eval_epoch, save_checkpoint

# Temporary directory for dummy data
TEST_DIR = "test_train_npz"

@pytest.fixture(scope="module")
def setup_dummy_data():
    os.makedirs(TEST_DIR, exist_ok=True)
    # Create small dummy .npz files
    for i in range(3):
        seq = torch.randint(0, 2, (100, 23)).numpy().astype("float32")
        npz_path = os.path.join(TEST_DIR, f"song_{i}.npz")
        np.savez(npz_path, sequence=seq)
    yield TEST_DIR
    shutil.rmtree(TEST_DIR)

@pytest.fixture
def small_model():
    model = DrumTransformer(
        num_classes=23,
        seq_len=50,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128
    )
    return model

def test_dataloader_runs(setup_dummy_data):
    loader = get_dataloader(setup_dummy_data, seq_len=50, batch_size=2)
    batch = next(iter(loader))
    x, y = batch
    assert x.shape[1] <= 50 and x.shape[2] == 23
    assert y.shape[1] <= 50 and y.shape[2] == 23

def test_train_epoch_runs(setup_dummy_data, small_model):
    loader = get_dataloader(setup_dummy_data, seq_len=50, batch_size=2)
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
    device = "cpu"
    small_model.to(device)
    loss = train_epoch(small_model, loader, optimizer, device)
    assert loss >= 0
    # Forward pass produces correct shape
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    preds = small_model(x)
    assert preds.shape == x.shape

def test_eval_epoch_runs(setup_dummy_data, small_model):
    loader = get_dataloader(setup_dummy_data, seq_len=50, batch_size=2)
    device = "cpu"
    small_model.to(device)
    val_loss = eval_epoch(small_model, loader, device)
    assert val_loss >= 0

def test_save_checkpoint_creates_file(tmp_path, small_model):
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(small_model, optimizer, epoch=1, path=checkpoint_path)
    assert os.path.exists(checkpoint_path)
    # Load checkpoint to ensure it's valid
    ckpt = torch.load(checkpoint_path)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt["epoch"] == 1
