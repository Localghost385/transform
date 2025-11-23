# Drum Transformer Model — Full Design Document (Updated Final Integrated Version)

## Overview
This document defines the **complete**, **finalized**, **integrated** design for a hierarchical Transformer-based drum pattern generation system. It merges the original early design with all updates made during development, including:

- Hierarchical dataset architecture (step/bar/phrase/section/song)
- Rich musical metrics for RL and evaluation
- NPZ preprocessing pipeline
- YAML-based drum mapping
- Drum Transformer model architecture
- Full training pipeline
- MIDI generation systems
- Humanization and output control
- Unified project structure

This represents the canonical specification for the system.

---

# 1. Input Data

## 1.1 Data assumptions
- Dataset consists of MIDI drum transcriptions from a single drummer.
- All MIDI files are **quantized**, grid-aligned, and contain only drum/percussion tracks.
- System supports expressive variants later, but phase 1 assumes quantized notation-style data.

## 1.2 Drum pitch mapping via YAML
All incoming MIDI pitches are mapped to fixed drum classes using an input YAML file.

### Example Input YAML Drum Map (with ``` instead of backticks)
``` 
cymbal_splash:
  class_id: 0
  pitches: [43, 83]

hi_hat_closed:
  class_id: 1
  pitches: [54]

hi_hat_open_1:
  class_id: 2
  pitches: [80]

hi_hat_open_2:
  class_id: 3
  pitches: [46]

hi_hat_pedal:
  class_id: 4
  pitches: [56]

kick_left:
  class_id: 5
  pitches: [23]

kick_center:
  class_id: 6
  pitches: [24]

racktom_1:
  class_id: 7
  pitches: [36]

racktom_2:
  class_id: 8
  pitches: [35]

racktom_3:
  class_id: 9
  pitches: [33]

racktom_4:
  class_id: 10
  pitches: [43]

floortom_1:
  class_id: 11
  pitches: [38]

floortom_2:
  class_id: 12
  pitches: [41]

cymbal_china:
  class_id: 13
  pitches: [49, 84]

cymbal_crash_1:
  class_id: 14
  pitches: [49]

cymbal_crash_2:
  class_id: 15
  pitches: [45, 50]

ride_1_bell:
  class_id: 16
  pitches: [53, 108]

ride_1_bow:
  class_id: 17
  pitches: [51, 102]

ride_1_edge:
  class_id: 18
  pitches: [59, 81]

ride_1_choke:
  class_id: 19
  pitches: [82]

snare_hit:
  class_id: 20
  pitches: [38]

snare_rimshot:
  class_id: 21
  pitches: [80]

snare_sidestick:
  class_id: 22
  pitches: [31, 25]
```

---

# 2. Drum Representation (Time Grid)

## 2.1 Quantization
- Default: **16 steps per bar** (16th notes)
- System supports 32 steps/bar if needed.

## 2.2 Representation
At each time step t, the drum hit vector is:

``` X[t] ∈ {0,1}^D ```

Where:
- D = drum class count from YAML file
- X[t][i] = 1 if class i hits on step t

Multi-hot, no timing offsets.

## 2.3 Velocity (phase 2)
Velocity is optional and disabled in phase 1. Future expansion uses categorical velocity bins.

---

# 3. Preprocessing Pipeline (preprocess.py)

## 3.1 Steps
1. Load raw MIDI.
2. Extract drum notes only.
3. Quantize timestamps.
4. Map pitches → drum class IDs via YAML.
5. Build matrix S of shape (T, D).
6. Save to NPZ with key ```sequence```.

## 3.2 Purpose
This stage produces standardized NPZ files for the dataset loader.

---

# 4. Hierarchical Dataset (dataset.py)

## 4.1 Hierarchical structure
The dataset computes features at multiple temporal resolutions:

- **Step** (fine)
- **Bar** (16 steps)
- **Phrase** (bars_per_phrase, default 4 bars)
- **Section** (optional ranges)
- **Song** (entire window)

## 4.2 Aggregation
- Bar-level: mean over each 16-step chunk
- Phrase-level: mean over bars_per_phrase × steps
- Song-level: metrics over the sampled window

## 4.3 Metrics computed
### Step / Bar / Phrase metrics:
- Hit density
- Drum density per class
- Max hits per chunk
- Dense fraction (threshold-based)
- IOI statistics (mean, variance, histogram)
- Co-occurrence rates
- Fast-pattern detection (interval <= threshold)
- Transition matrices
- Syncopation
- N-gram counts
- Hits-per-bar (phrase only)

### Section metrics:
Computed only if sections are provided.

### Song metrics:
- Hit density
- Entropy (step-level and phrase-level)
- Ngram diversity
- Density and distribution properties

## 4.4 Outputs
Each ```__getitem__``` returns:
- Step, bar, phrase tensors
- Full hierarchical metrics dictionary for optional RL use

## 4.5 Dataloader
Pads sequences and batches hierarchical tensors and metric dicts.

---

# 5. Model Architecture (model.py)

## 5.1 High-level design
A **hierarchical-aware Transformer**:

- Input = step-level embeddings from multi-hot vectors
- Additional bar/phrase positional encodings optional
- Transformer decoder (GPT-style)
- Predicts next-step multi-hot vector via sigmoid output

## 5.2 Embeddings
- Linear projection: D → d_model
- Positional embeddings
- Optional bar-position and phrase-position embeddings

## 5.3 Transformer configuration
- Default layers: 8
- Heads: 8
- d_model: 512
- FFN: 2048
- Dropout: 0.1
- Context window: 512 steps
- Masked self-attention

## 5.4 Output head
- Linear(d_model → D)
- Sigmoid for independent drum activation probabilities

## 5.5 Loss
Binary Cross-Entropy:
``` Loss = mean(BCE(pred[t], target[t])) ```

## 5.6 RL compatibility
The model can also train via:
- Combined BCE + metric reward
- Pure RL optimizing syncopation, IOI variance, density, etc.

---

# 6. Training Pipeline (train.py)

## 6.1 Optimizer and schedule
- AdamW
- Warmup + cosine decay
- Gradient clipping

## 6.2 Batch sampling
- Random windows of length 512
- Respect song boundaries
- Augmentation (temporal shifts) optional

## 6.3 Logging
- Train/val loss
- Per-class precision/recall
- Optional RL reward curves

## 6.4 Checkpointing
- Save every epoch
- Best checkpoint logic
- Final ```final.pt```

---

# 7. Generation Pipeline (generate.py)

## 7.1 Inputs
- Random seed
- Sequence length in bars
- Temperature
- Optional constraints (avoid dense bars, force fills, etc.)

## 7.2 Generation loop
For each step:
1. Feed context to model
2. Compute ```p = sigmoid(logits / temperature)```
3. Sample per-class from Bernoulli(p)
4. Append vector and continue

## 7.3 MIDI rendering
- Map class IDs to output drum-map pitches (GM or custom)
- Write NOTE_ON/OFF events
- Apply microtiming and velocity humanization if enabled

---

# 8. Humanization (humanize.py)

## Optional processes:
- Microtiming jitter ± few ms
- Velocity jitter
- Swing feel
- Optional groove templates

---

# 9. Project Structure

```
drum-transformer/
│
├── data/
│   ├── raw_midi/
│   └── processed_npz/
│
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── utils/
│   │   ├── midi_io.py
│   │   ├── drum_mapping.py
│   │   ├── humanize.py
│   │   └── sampling.py
│
├── saved_models/
├── checkpoints/
└── final.pt
```

---

# 10. Default Parameter Summary

| Component | Value |
|----------|--------|
| Quantization | 16 steps/bar |
| Drum classes | 23 |
| Model type | Transformer decoder |
| Layers | 8 |
| Heads | 8 |
| d_model | 512 |
| FFN | 2048 |
| Context | 512 steps |
| Loss | BCE |
| Generation input | seed, length_bars |
| Default length | 64 bars |
| Humanization | off by default |

---

# 11. Future Extensions
- Velocity modeling
- Structural tokens for large-scale form
- Multi-drummer style tokens
- VAE expressive postprocessing
- ONNX export
- Real-time inference

---

# 12. Conclusion
This document defines the **complete, unified architecture** for a hierarchical drum-Transformer system, including:

- Preprocessing → hierarchical dataset → Transformer → training → RL → MIDI generation
- YAML-driven drum mapping
- Full-step/bar/phrase metric pipeline
- High-level musical evaluation
- Modular system design

This design is the final reference specification for implementation.
