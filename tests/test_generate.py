import os
import sys
import tempfile
import numpy as np
import torch
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.model import DrumTransformer
from src.generate import generate_sequence

@pytest.fixture
def dummy_model():
    # Small model for testing
    model = DrumTransformer(num_classes=5, seq_len=10, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    return model

def test_generate_sequence_shape(dummy_model):
    seq_len = 8  # 8 bars * 1 step per bar for simplicity
    steps_per_bar = 1
    length_bars = seq_len
    sequence = generate_sequence(dummy_model, seed=42, length_bars=length_bars, steps_per_bar=steps_per_bar, temperature=1.0, device="cpu")
    assert isinstance(sequence, np.ndarray)
    assert sequence.shape == (length_bars * steps_per_bar, dummy_model.num_classes)

def test_generate_sequence_values(dummy_model):
    sequence = generate_sequence(dummy_model, seed=123, length_bars=5, steps_per_bar=2, temperature=1.0, device="cpu")
    # Values should be 0 or 1 (after Bernoulli sampling)
    unique_values = np.unique(sequence)
    assert set(unique_values).issubset({0.0, 1.0})

def test_generate_sequence_deterministic(dummy_model):
    # Same seed should produce same output
    seq1 = generate_sequence(dummy_model, seed=999, length_bars=4, steps_per_bar=2, temperature=1.0, device="cpu")
    seq2 = generate_sequence(dummy_model, seed=999, length_bars=4, steps_per_bar=2, temperature=1.0, device="cpu")
    np.testing.assert_array_equal(seq1, seq2)

def test_generate_sequence_save_npz(dummy_model):
    sequence = generate_sequence(dummy_model, seed=7, length_bars=3, steps_per_bar=2, temperature=1.0, device="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gen.npz")
        np.savez(save_path, sequence=sequence)
        assert os.path.exists(save_path)
        loaded = np.load(save_path)["sequence"]
        np.testing.assert_array_equal(sequence, loaded)
