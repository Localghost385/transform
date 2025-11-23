import os
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader
from src.dataset import CompleteHierarchicalDrumDataset, get_complete_rl_dataloader  # Adjust import

# -------------------- Fixtures --------------------
@pytest.fixture
def temp_npz_dir(tmp_path):
    """Create temporary NPZ files with synthetic drum sequences."""
    npz_dir = tmp_path
    for i in range(3):
        seq_len = np.random.randint(64, 128)
        num_drums = 5
        sequence = np.random.randint(0, 2, size=(seq_len, num_drums)).astype(np.float32)
        np.savez(npz_dir / f"seq_{i}.npz", sequence=sequence)
    return npz_dir

# -------------------- Dataset Tests --------------------
def test_dataset_loading(temp_npz_dir):
    dataset = CompleteHierarchicalDrumDataset(temp_npz_dir, seq_len=32)
    assert len(dataset) > 0, "Dataset should have samples"
    sample = dataset[0]
    assert "step" in sample and "bar" in sample and "phrase" in sample and "metrics" in sample
    assert isinstance(sample["step"], torch.Tensor)
    assert isinstance(sample["metrics"], dict)
    # Check step sequence length
    assert sample["step"].shape[0] <= 32

def test_augmentation_shift(temp_npz_dir):
    dataset = CompleteHierarchicalDrumDataset(temp_npz_dir, seq_len=32, augment=True)
    sample1 = dataset[0]["step"].numpy()
    sample2 = dataset[0]["step"].numpy()
    # With augmentation, sequences may differ due to shift
    assert sample1.shape == sample2.shape

def test_metric_shapes(temp_npz_dir):
    dataset = CompleteHierarchicalDrumDataset(temp_npz_dir, seq_len=32, steps_per_bar=8, bars_per_phrase=2)
    sample = dataset[0]
    step_metrics = sample["metrics"]["step"]
    bar_metrics = sample["metrics"]["bar"]
    phrase_metrics = sample["metrics"]["phrase"]
    
    # IOI stats shape
    assert step_metrics["ioi_stats"].shape[1] == 2 or step_metrics["ioi_stats"].shape[1] == len(step_metrics["fast"])
    
    # Hits per bar
    assert phrase_metrics["hits_per_bar"].shape[1] == sample["step"].shape[1]

def test_empty_sequence_handling(tmp_path):
    empty_file = tmp_path / "empty.npz"
    np.savez(empty_file, sequence=np.zeros((0, 5), dtype=np.float32))
    # Should raise error if all sequences invalid
    with pytest.raises(ValueError):
        CompleteHierarchicalDrumDataset(tmp_path, seq_len=32)

# -------------------- Dataloader Tests --------------------
def test_dataloader_batch_shapes(temp_npz_dir):
    dataloader = get_complete_rl_dataloader(temp_npz_dir, seq_len=32, batch_size=2, num_workers=0)
    batch = next(iter(dataloader))
    # Check tensor shapes
    assert batch["step"].shape[0] == 2
    assert batch["bar"].shape[0] == 2
    assert batch["phrase"].shape[0] == 2
    # Metrics list length
    assert len(batch["metrics"]) == 2
    # Check metrics contents
    for metrics in batch["metrics"]:
        assert "step" in metrics and "bar" in metrics and "phrase" in metrics and "song" in metrics

def test_collate_padding(temp_npz_dir):
    # Create sequences of different lengths
    for i in range(2):
        seq_len = 16 + i*4
        sequence = np.random.randint(0, 2, size=(seq_len, 5)).astype(np.float32)
        np.savez(temp_npz_dir / f"seq_pad_{i}.npz", sequence=sequence)
    
    dataloader = get_complete_rl_dataloader(temp_npz_dir, seq_len=16, batch_size=2)
    batch = next(iter(dataloader))
    # Check padding
    assert batch["step"].shape[1] >= 16

