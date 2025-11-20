# Drum Transformer Development To-Do List

## Phase 0: Environment Setup
- Set up Python environment (Python 3.10+ recommended)
- Install libraries:
  - `torch`, `torchvision` (PyTorch)
  - `mido` or `pretty_midi` for MIDI processing
  - `numpy`
  - `PyYAML` for drum map
  - `tqdm` for progress bars
- Set up project directory:

```
drum-transformer/
├── data/
├── src/
├── saved_models/
├── checkpoints/
└── final.pt
```

- Create virtual environment and version control (`git init`)

---

## Phase 1: Data Preparation
- Collect all input MIDI files into `data/raw_midi/`
- Create input drum map YAML (`src/drum_mapping.py` or `.yaml`) defining pitch → class mapping
- Write MIDI preprocessing script (`preprocess.py`):
- Load MIDI files
- Filter drum channels only
- Quantize notes to 16 steps/bar grid
- Map pitches → drum classes using YAML
- Output per-song matrices (T x D) as `.npz` in `data/processed_npz/`
- Optional: implement velocity extraction for future extension

---

## Phase 2: Dataset & Loader
- Implement PyTorch Dataset class (`dataset.py`):
- Load `.npz` files
- Generate context windows (length L=512)
- Return (input_seq, target_seq) pairs
- Optional: data augmentation (shift, masking)
- Create DataLoader with batching

---

## Phase 3: Model Implementation
- Implement decoder-only Transformer (`model.py`):
- Embedding layer (multi-hot → d_model)
- Positional embeddings
- Transformer blocks (masked self-attention)
- Output layer → D with sigmoid
- Define binary cross-entropy loss
- Add utility functions: save/load model, count parameters

---

## Phase 4: Training Pipeline
- Create `train.py`:
- Setup optimizer (AdamW) and learning rate scheduler
- Implement training loop with BCE loss
- Add validation loop and logging
- Save checkpoints, implement early stopping
- Test on small subset for sanity check

---

## Phase 5: Generation Pipeline
- Create `generate.py`:
- Load trained model
- Initialize sequence (zeros or BOS)
- Use random seed
- Autoregressive generation
- Map class indices → MIDI pitches (output drum map)
- Export MIDI
- Implement optional parameters: temperature, sequence length
- Test generation for short sequences first

---

## Phase 6: Humanization & Postprocessing
- Implement microtiming jitter
- Implement velocity jitter
- Implement swing adjustments
- Test postprocessing to ensure playable MIDI

---

## Phase 7: Evaluation
- Listen to generated sequences
- Optional metrics:
- Hit accuracy per drum class
- Pattern similarity
- Diversity of generated sequences
- Save evaluation logs

---

## Phase 8: Advanced / Optional Features
- Add velocity prediction (multi-class)
- Add structural tokens (VERSE, CHORUS, FILL)
- Multi-drummer training with style tokens
- Convert model to ONNX/TorchScript for faster inference
- Implement VAE post-processor for expressiveness

---

## Phase 9: Documentation & Maintenance
- Write README with usage instructions
- Document YAML drum maps (input/output)
- Version control checkpoints
- Optional: scripts for batch MIDI generation

---

### First Actionable Step
- Create the **input drum map YAML** and verify MIDI preprocessing can convert raw MIDI into class matrices using it. This is required before dataset creation and model training.
