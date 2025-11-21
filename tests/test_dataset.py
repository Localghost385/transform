import os
import shutil
import sys
import pytest
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.dataset import DrumDataset, get_dataloader

# Temporary directory for test .npz files
TEST_DIR = "test_npz"

@pytest.fixture(scope="module")
def setup_test_data():
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Create 3 test sequences with different lengths
    seq1 = np.random.randint(0, 2, size=(600, 10)).astype(np.float32)  # longer than seq_len
    seq2 = np.random.randint(0, 2, size=(512, 10)).astype(np.float32)  # equal to seq_len
    seq3 = np.random.randint(0, 2, size=(200, 10)).astype(np.float32)  # shorter than seq_len

    np.savez(os.path.join(TEST_DIR, "seq1.npz"), sequence=seq1)
    np.savez(os.path.join(TEST_DIR, "seq2.npz"), sequence=seq2)
    np.savez(os.path.join(TEST_DIR, "seq3.npz"), sequence=seq3)

    yield TEST_DIR

    # Cleanup
    shutil.rmtree(TEST_DIR)

def test_dataset_length(setup_test_data):
    dataset = DrumDataset(setup_test_data, seq_len=512)
    # seq1 -> 600-512=88 samples, seq2 -> 1 sample, seq3 -> 1 sample (too short)
    expected_len = 88 + 1 + 1
    assert len(dataset) == expected_len, f"Expected length {expected_len}, got {len(dataset)}"

def test_dataset_item_shape(setup_test_data):
    dataset = DrumDataset(setup_test_data, seq_len=512)
    x, y = dataset[0]
    # Input shape: (seq_len, D) or smaller if sequence shorter
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[1] == 10
    assert y.shape[1] == 10

def test_dataset_short_sequence(setup_test_data):
    dataset = DrumDataset(setup_test_data, seq_len=512)
    short_sample = [i for i, (seq_idx, _) in enumerate(dataset.sample_indices)
                    if dataset.sequences[seq_idx].shape[0] < 512]
    x, y = dataset[short_sample[0]]
    seq_len_actual = dataset.sequences[dataset.sample_indices[short_sample[0]][0]].shape[0]
    assert x.shape[0] == seq_len_actual - 1
    assert y.shape[0] == seq_len_actual - 1

def test_dataset_augmentation(setup_test_data):
    dataset = DrumDataset(setup_test_data, seq_len=512, augment=True)
    x1, y1 = dataset[0]
    x2, y2 = dataset[0]  # same sample, augmentation may change
    assert x1.shape == x2.shape
    assert y1.shape == y2.shape

def test_dataloader_integration(setup_test_data):
    loader = get_dataloader(setup_test_data, seq_len=512, batch_size=2)
    for x_batch, y_batch in loader:
        assert x_batch.shape[1] <= 512
        assert y_batch.shape[1] <= 512
        assert x_batch.shape[0] <= 2
        assert y_batch.shape[0] <= 2
        break
