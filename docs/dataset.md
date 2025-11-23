# Technical Documentation: CompleteHierarchicalDrumDataset with Optimized Training Integration

## Overview

The ```CompleteHierarchicalDrumDataset``` is a fully hierarchical dataset designed for reinforcement learning (RL) and modeling of drum sequences. It supports **multi-level analysis** (step, bar, phrase, section, and song) with extensive metrics computation for rhythm structure, density, diversity, and temporal patterns.  

The dataset can be integrated with PyTorch ```DataLoader``` for batched processing in RL training pipelines or generative models. It is fully compatible with **multi-GPU training** and **hyperparameter optimization** for automatic tuning of RL reward weights and other training parameters.

---

## Directory Structure and File Format

- **Input Directory:** A folder containing .npz files. Each .npz file must contain a NumPy array stored under the key ```sequence```.
- **Sequence Shape:** ```[time_steps, num_drums]```
  - ```time_steps``` – number of discrete time steps in the sequence
  - ```num_drums``` – number of drum instruments/tracks
- **File Naming:** Any .npz file in the folder is loaded; others are ignored.

---

## Class: ```CompleteHierarchicalDrumDataset```

### Initialization

```CompleteHierarchicalDrumDataset(
    npz_dir,
    seq_len=512,
    steps_per_bar=16,
    bars_per_phrase=4,
    sections=None,
    augment=False,
    fast_threshold=4
)```

**Parameters:**
- ```npz_dir``` – directory containing .npz sequences
- ```seq_len``` – length of subsequences sampled from full sequences
- ```steps_per_bar``` – number of steps per musical bar
- ```bars_per_phrase``` – number of bars in a musical phrase
- ```sections``` – optional list of ```(start, end)``` indices for song sections
- ```augment``` – if True, randomly shift sequences for data augmentation
- ```fast_threshold``` – threshold in steps to identify fast drum hits

### Internal Variables

- ```self.files``` – list of .npz files
- ```self.sequences``` – loaded sequences from all files
- ```self.phrase_len``` – ```steps_per_bar * bars_per_phrase```
- ```self.sample_indices``` – precomputed indices ```(sequence_idx, start_idx)``` for slicing

---

## Sequence Sampling

The dataset supports **sliding window sampling**:

- For a sequence of length ```L``` and ```seq_len = S```:
  ```text
  max_start = L - S
  sample_indices = [(seq_idx, start_idx) for start_idx in range(max_start)]
  ```
- This allows training on all overlapping sub-sequences.
- Optional **augmentation**: random shift along time axis (```np.roll```).

---

## Hierarchical Aggregation

The dataset aggregates sequences at multiple hierarchical levels:

| Level   | Length / Steps                    | Function                                      |
|---------|----------------------------------|-----------------------------------------------|
| Step    | ```seq_len```                         | Raw stepwise sequence                          |
| Bar     | ```steps_per_bar```                   | Aggregated average over each bar              |
| Phrase  | ```steps_per_bar * bars_per_phrase``` | Aggregated over entire phrase                 |
| Section | user-defined ```(start, end)```       | Aggregated per song section                   |
| Song    | entire sequence                   | Aggregated across whole song                  |

Aggregation is handled using ```_aggregate(seq, step_size)```:

```text
num_units = seq.shape[0] // step_size
aggregated = seq[:num_units*step_size].reshape(num_units, step_size, -1).mean(axis=1)
```

---

## Metrics Computed

The dataset computes a **wide range of metrics** at each hierarchy level, optimized for RL reward shaping.

### 1. Hit Density

```text
def _hit_density(seq):
    total_hits = seq.sum(axis=1)
    avg_hits = total_hits.mean()
    drum_density = seq.mean(axis=0)
    max_hits = total_hits.max()
```

- **avg_hits:** average number of hits per step
- **drum_density:** fraction of hits per drum
- **max_hits:** max simultaneous hits in a step

---

### 2. Fraction of Dense Bars

```text
def _fraction_dense_bars(seq, threshold=None)
```

- Fraction of bars with hits above a threshold
- Default: half of number of drums

---

### 3. Inter-Onset Interval (IOI) Statistics

```text
def _ioi_stats(seq)
```

- Computes **mean**, **variance**, and **histogram** of intervals between hits per drum
- Useful for analyzing rhythmic motifs and spacing patterns

---

### 4. Co-occurrence

```text
def _co_occurrence(seq)
```

- Fraction of steps where more than one drum plays simultaneously
- Captures polyphonic density

---

### 5. N-grams

```text
def _n_grams(seq, n=4)
```

- Extracts n-step drum patterns (flattened across drums)
- Returns a ```Counter``` for pattern frequency analysis
- Supports diversity analysis at phrase/song level

---

### 6. Fast Patterns

```text
def _fast_patterns(seq)
```

- Counts number of hits separated by ≤ ```fast_threshold``` steps
- Measures rhythmic intensity

---

### 7. Transitions

```text
def _transitions(seq)
```

- Computes **drum-to-drum conditional probability matrix** between consecutive steps
- Useful for capturing sequential structure

---

### 8. Syncopation

```text
def _syncopation(seq)
```

- Fraction of hits on **weak beats** (not downbeat)
- Weak beats are determined by modulo operation with ```steps_per_bar```

---

### 9. Phrase Structure

```text
def _phrase_structure(seq)
```

- Returns number of hits per bar in a phrase
- Ensures short sequences are handled safely

---

### 10. Section Metrics

```text
def _section_metrics(seq)
```

- Aggregates metrics for predefined sections
- Returns dictionary with:
  - avg_hits
  - drum_density
  - max_hits
  - dense_frac

---

### 11. Song Metrics

```text
def _song_metrics(seq)
```

- Aggregates metrics across the entire sequence:
  - avg_hits
  - drum_density
  - max_hits
  - step_entropy – entropy of hits per step
  - phrase_entropy – average entropy per phrase
  - ngram_diversity – count of unique n-grams

---

## __getitem__ Output

Each sample from the dataset returns a dictionary:

```text
{
    "step": torch.Tensor,   # [seq_len, num_drums]
    "bar": torch.Tensor,    # [num_bars, num_drums]
    "phrase": torch.Tensor, # [num_phrases, num_drums]
    "metrics": {
        "step": { ... },
        "bar": { ... },
        "phrase": { ... },
        "section": [...],
        "song": { ... }
    }
}
```

- Metrics are **precomputed for RL reward shaping or analysis**
- Allows **multi-level evaluation** in generative or RL tasks

---

## DataLoader Integration

```text
get_complete_rl_dataloader(
    npz_dir,
    seq_len=512,
    steps_per_bar=16,
    bars_per_phrase=4,
    batch_size=16,
    shuffle=True,
    augment=False,
    num_workers=0
)
```

- Wraps the dataset in a PyTorch DataLoader
- Uses **custom collate_fn** with ```pad_sequence``` for variable-length sequences
- Returns dictionary with:
  - ```"step"```, ```"bar"```, ```"phrase"``` tensors ```[batch, seq_len, num_drums]```
  - ```"metrics"``` – list of per-sample metric dictionaries
- Fully compatible with **single-node multi-GPU (```nn.DataParallel```) and DDP (```DistributedDataParallel```)**

---

## Optimizations for Training

1. **Mixed-Precision Training:** ```torch.cuda.amp.autocast()``` and ```GradScaler``` to reduce memory and increase throughput.
2. **Gradient Clipping:** prevents exploding gradients for long sequences.
3. **Multi-GPU / DDP Support:** automatic data distribution and gradient synchronization for large-scale training.
4. **Checkpointing & Resume:** save model, optimizer, and scheduler states for reproducibility.
5. **RL Reward Shaping:** supports hierarchical metrics and composite reward functions.
6. **Dynamic Hyperparameter Optimization:**
   - Use frameworks like ```Optuna``` or ```Ray Tune```.
   - Tune ```RL_weight```, learning rate, dropout, hidden dimensions automatically.
   - Optionally treat reward weights as **trainable parameters** updated along with model.
7. **Reward Normalization:** stabilize RL training by normalizing per-batch or per-epoch metric values.
8. **Early Pruning:** stop trials or gradient updates that are unlikely to improve performance.

---

## Design Considerations

- **Hierarchical modeling** – step → bar → phrase → section → song
- **Multi-level metrics** – allows RL reward shaping at different levels
- **Flexible augmentation** – optional sequence shifts
- **Efficient indexing** – precomputes all sample start indices
- **Robust to short sequences** – handles sequences shorter than a bar/phrase
- **Extensible metrics** – easy to add new statistics (e.g., swing, polyrhythm)
- **DDP-Compatible** – designed for multi-node or multi-GPU distributed training

---

## Example Usage

```text
from your_module import get_complete_rl_dataloader

npz_dir = "./train/"
dataloader = get_complete_rl_dataloader(npz_dir, seq_len=512, batch_size=2)

for batch in dataloader:
    print(batch["step"].shape)  # [batch, seq_len, num_drums]
    print(batch["metrics"][0]["song"]["ngram_diversity"])
    break
```

---

## Testing Recommendations

- **Unit tests** for:
  - ```_hit_density```, ```_ioi_stats```, ```_fast_patterns```, ```_transitions```
  - Edge cases: empty sequences, short sequences
- **Dataloader tests**:
  - Correct batch shapes
  - Padding correctness for variable-length sequences
- **Integration tests**:
  - Verify hierarchical metrics are consistent across levels
  - Test multi-GPU training and DDP synchronization
  - Validate reward normalization and RL weighting adjustments

---

## Summary

The ```CompleteHierarchicalDrumDataset``` provides a **robust, multi-level framework** for drum sequence modeling. Its main strengths:

- Hierarchical aggregation
- Rich, interpretable metrics
- Augmentation support
- Easy integration with PyTorch for RL and generative tasks
- Fully compatible with **optimized multi-GPU and DDP training**
- Supports **automated hyperparameter optimization** for reward shaping and training stability

Ideal for:

- Reinforcement learning on symbolic drums
- Rhythm pattern analysis
- Generative modeling of polyphonic drum sequences
