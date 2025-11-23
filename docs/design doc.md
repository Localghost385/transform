# Drum Transformer Model — Full Design Document (Updated with YAML Drum Map)

## Overview
This document defines the architecture, data representation, training pipeline, and generation process for a Transformer-based drum pattern generator trained on quantized MIDI drum transcriptions from a single drummer.

The goal is to create a system that:
- Learns the drummer’s stylistic tendencies from full-song MIDI files.
- Uses a Transformer decoder (GPT-like) trained on quantized drum-grid sequences.
- Generates completely new full-length drum tracks.
- Takes only a **random seed** (and optionally a target length in bars) as input.
- Outputs a quantized MIDI file with optional humanization.

---

# 1. Input Data

## 1.1 Data assumptions
- All input MIDI files are **quantized** drum transcriptions (e.g., drum notation exports).
- Timing is precise (no expressive microtiming).
- All tracks only contain percussion parts.

## 1.2 Drum pitch mapping
MIDI drum pitches must be mapped to a fixed set of **drum classes** using a YAML file.  
This mapping ensures consistent representation across all input MIDI files regardless of differing articulations or pitch formats.

### Example Input YAML Drum Map (`input_drum_map.yaml`)
```
cymbal_splash:
  class_id: 0
  pitches: [43, 83]  # G2, B5 (hit, choke)

hi_hat_closed:
  class_id: 1
  pitches: [54]  # F#1

hi_hat_open_1:
  class_id: 2
  pitches: [80]  # G#5

hi_hat_open_2:
  class_id: 3
  pitches: [46]  # A#1

hi_hat_pedal:
  class_id: 4
  pitches: [56]  # G#1 (pedal close foot)

kick_left:
  class_id: 5
  pitches: [23]  # B0

kick_center:
  class_id: 6
  pitches: [24]  # C1

racktom_1:
  class_id: 7
  pitches: [36]  # C2

racktom_2:
  class_id: 8
  pitches: [35]  # B1

racktom_3:
  class_id: 9
  pitches: [33]  # A1

racktom_4:
  class_id: 10
  pitches: [43]  # G1

floortom_1:
  class_id: 11
  pitches: [38]  # D2

floortom_2:
  class_id: 12
  pitches: [41]  # F1

cymbal_china:
  class_id: 13
  pitches: [49, 84]  # E2, C6 (hit, choke)

cymbal_crash_1:
  class_id: 14
  pitches: [49, 49]  # C#2, C#6 (hit, choke)

cymbal_crash_2:
  class_id: 15
  pitches: [45, 50]  # A2, D6 (hit, choke)

ride_1_bell:
  class_id: 16
  pitches: [53, 108]  # F2, G8

ride_1_bow:
  class_id: 17
  pitches: [51, 102]  # D#2, F#8

ride_1_edge:
  class_id: 18
  pitches: [59, 81]  # B2, A5

ride_1_choke:
  class_id: 19
  pitches: [82]  # A#5 (mute/choke)

snare_hit:
  class_id: 20
  pitches: [38]  # D1

snare_rimshot:
  class_id: 21
  pitches: [80]  # G5

snare_sidestick:
  class_id: 22
  pitches: [31, 25]  # G0, C#1
```

---

# 2. Drum Representation (Quantized Time Grid)

## 2.1 Quantization resolution
- Default: **16 steps per bar** (16th-note resolution).
- Optional: 32 steps/bar for more detailed transcriptions.

## 2.2 Per-step representation
For each time step ```t```, define a multi-hot vector:

```
X[t] ∈ {0,1}^D
```

Where:
- ```D``` = number of drum classes.
- ```X[t][i] = 1``` if drum class ```i``` is played at step ```t```.

No timing offsets or explicit durations are used; hits are assumed to have standard short durations typical of drums.

## 2.3 Optional velocity modeling
Velocity may be ignored (default) or represented as:
- Quantized bins (e.g., 4 levels)
- A separate categorical prediction per active drum

For initial implementation, **velocity is disabled** for simplicity.

---

# 3. Dataset Construction

## 3.1 MIDI preprocessing
1. Load each MIDI file.
2. Extract only drum notes.
3. Quantize each note to the nearest grid step.
4. Map MIDI pitch → drum class index using the input YAML map.
5. Construct a matrix:  

```
Song S: shape (T, D)
T = total number of time steps in the song
D = number of drum classes
```

6. Save each song as compressed NumPy ```.npz``` files.

## 3.2 Sequence chunking for training
- Transformer context window ```L = 512``` steps (32 bars at 16 steps/bar).
- Sample random windows from songs:
  - Input: ```X[t ... t+L-1]```
  - Target: ```X[t+1 ... t+L]```
- Songs are **not concatenated**; boundaries are respected.

---

# 4. Model Architecture

## 4.1 High-level structure
A **decoder-only Transformer** (GPT-like) operating on sequences of drum-grid embeddings.

## 4.2 Input embedding
Each multi-hot vector ```X[t]``` is projected through:
- A linear layer: ```Linear(D → d_model)```
- Positional embeddings (learned)
- Optional bar position embedding (modulo the grid)

## 4.3 Transformer configuration
Default architecture:

- Layers: **8**
- Heads: **8**
- Model dimension: **512**
- Feedforward dimension: **2048**
- Dropout: **0.1**
- Context window: **512 tokens**
- Activation: GELU
- Masked self-attention for next-step prediction

## 4.4 Output head
- Linear layer: ```d_model → D```
- Sigmoid activation
- Output probability for each drum class at next step

This is a **multi-label prediction**.

## 4.5 Loss function
- Binary cross-entropy loss (BCE) over all drum classes per time step:

```
Loss = mean(BCE(pred[t], X_true[t+1]))
```

---

# 5. Training Pipeline

## 5.1 Optimizer
- AdamW  
- Learning rate warmup followed by cosine decay  

## 5.2 Training parameters
- Batch size: **8–32** depending on VRAM  
- Epochs: **50–200** depending on dataset size  
- Gradient clipping to avoid spikes  
- Checkpoints saved every epoch  
- Early stopping based on validation loss  

## 5.3 Logging
Track:
- Training loss
- Validation loss
- Hit accuracy per drum class
- Overall precision/recall (optional)

---

# 6. Generation Process

## 6.1 Inputs
- **Random seed** (integer)
- Optional: ```length_bars``` (default 64)
- Optional: sampling temperature (default 1.0)

## 6.2 Initialization
- Convert the seed into a random number generator state.
- Initialize sequence with:
  - Either a learned BOS vector  
  - Or zeros (empty step)  

## 6.3 Autoregressive generation
For each time step:
1. Feed current context into Transformer.
2. Get predicted probabilities for next step:  
   ```p = sigmoid(logits / temperature)```
3. Sample each drum class independently using Bernoulli(```p```), or apply optional constraints.
4. Append next-step vector to sequence.

Continue until:
- Number of generated steps = ```length_bars * steps_per_bar```.

## 6.4 Convert steps → MIDI
- Map each active drum class to an **output drum map** pitch (GM or desired MIDI pitch).
- Create NOTE_ON / NOTE_OFF events.
- Use fixed or user-specified tempo.
- Export standard MIDI.

---

# 7. Optional Humanization (Postprocessing)

## 7.1 Microtiming
For each hit, add a slight random offset:
- Δt ∈ uniform(-5 ms, +5 ms)

## 7.2 Velocity jitter
- Add jitter ±5–15 velocity units
- Clamp to 1–127 range

## 7.3 Swing
- Push even 16th notes slightly late if desired (+10 ms)

---

# 8. Project Structure

```
drum-transformer/
│
├─┬── data/
│ ├── raw_midi/
│ └── processed_npz/
│
├─┬── src/
│ ├── preprocess.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── generate.py
│ ├── utils/
│ │ ├── midi_io.py
│ │ ├── drum_mapping.py
│ │ ├── humanize.py
│ │ └── sampling.py
│
├── saved_models/
├── checkpoints/
└── final.pt
```

---

# 9. Final Default Parameters Summary

| Component               | Value / Setting                       |
|------------------------|----------------------------------------|
| Quantization           | 16 steps/bar                           |
| Drum classes           | 23 (per input YAML)                    |
| Velocity               | Disabled (binary-only)                 |
| Model type             | Transformer decoder                    |
| Layers                 | 8                                      |
| Heads                  | 8                                      |
| Embedding dimension    | 512                                    |
| FFN dimension          | 2048                                   |
| Context window         | 512 steps                              |
| Loss                   | BCE                                    |
| Generation input       | seed, length_bars                      |
| Default length         | 64 bars                                |
| Humanization           | Optional (off by default)              |

---

# 10. Future Extensions

## 10.1 Add velocity modeling
- Multi-class per-hit velocity prediction
- More realistic output

## 10.2 Add structural tokens
- SECTION_VERSE
- SECTION_CHORUS
- SECTION_FILL
…to improve long-form structure.

## 10.3 Multi-drummer training
- Add “style tokens” to indicate drummer identity.

## 10.4 Output drum map normalization
- Map internal classes to a standard GM kit or any target MIDI format

## 10.5 VAE postprocessor
- Improve expressiveness (GrooVAE-like).

## 10.6 Convert model to ONNX or TorchScript
- Faster inference.

---

# 11. Conclusion

This design implements a practical Transformer-based drum generator that:
- Uses quantized MIDI representations suitable for drum transcription.
- Learns long-range drummer style from full transcriptions.
- Generates entire drum tracks with only a random seed.
- Supports input/output drum maps for clean mapping and output flexibility.
- Maintains simplicity while allowing future expansion.
