import os
import sys
import argparse
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.dataset import DrumDataset
from src.train import (
    setup_device_and_perf,
    build_optimizer,
    build_scheduler_with_warmup,
    save_checkpoint,
    maybe_load_checkpoint,
    train_epoch,
    eval_epoch,
)
from model import DrumTransformer

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def dummy_npz_dir(tmp_path):
    """Create dummy .npz files for DrumDataset"""
    for i in range(3):
        data = torch.randint(0, 2, (10, 23)).numpy().astype("float32")
        npz_path = tmp_path / f"seq_{i}.npz"
        with open(npz_path, "wb") as f:
            import numpy as np
            np.savez(f, sequence=data)
    return tmp_path

@pytest.fixture
def device():
    return setup_device_and_perf(argparse.Namespace(num_threads=1))

@pytest.fixture
def dummy_model():
    model = DrumTransformer(
        num_classes=23,
        seq_len=10,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.1
    )
    return model

@pytest.fixture
def dummy_dataloader(dummy_npz_dir):
    dataset = DrumDataset(str(dummy_npz_dir), seq_len=5, augment=False)
    return DataLoader(dataset, batch_size=2, shuffle=False)

# -------------------------
# Tests
# -------------------------
def test_dataset_loading(dummy_npz_dir):
    dataset = DrumDataset(str(dummy_npz_dir), seq_len=5)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 23  # num_classes
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

def test_optimizer_and_scheduler(dummy_model):
    optimizer = build_optimizer(dummy_model, lr=1e-3)
    scheduler = build_scheduler_with_warmup(optimizer, warmup_steps=2, total_steps=10)
    # Check optimizer has parameters
    assert len(optimizer.param_groups) > 0
    # Check scheduler returns values
    lr_values = [scheduler.get_last_lr()[0] for _ in range(3)]
    assert all(isinstance(lr, float) for lr in lr_values)

def test_train_step(dummy_model, dummy_dataloader, device):
    model = dummy_model.to(device)
    optimizer = build_optimizer(model, lr=1e-3)
    scaler = None
    avg_loss, step = train_epoch(
        model, dummy_dataloader, optimizer, scaler, device,
        grad_accum=1, use_amp=False
    )
    assert avg_loss >= 0
    assert step > 0

def test_eval_step(dummy_model, dummy_dataloader, device):
    model = dummy_model.to(device)
    avg_loss = eval_epoch(model, dummy_dataloader, device, use_amp=False)
    assert avg_loss >= 0

def test_checkpoint_save_and_load(dummy_model, tmp_path):
    optimizer = build_optimizer(dummy_model, lr=1e-3)
    scheduler = build_scheduler_with_warmup(optimizer, warmup_steps=1, total_steps=5)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    ckpt_path = tmp_path / "test_ckpt.pt"
    save_checkpoint(str(ckpt_path), dummy_model, optimizer, scaler, scheduler, epoch=1, step=1)
    assert os.path.exists(ckpt_path)

    start_epoch, global_step, loaded = maybe_load_checkpoint(
        str(ckpt_path), dummy_model, optimizer, scaler, scheduler, map_location="cpu"
    )
    assert loaded
    assert start_epoch == 2
    assert global_step == 1
