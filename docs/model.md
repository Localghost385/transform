# Technical Documentation: HierarchicalDrumModel

## Overview

The ```HierarchicalDrumModel``` is a PyTorch neural network designed to model drum sequences at multiple hierarchical levels (step, bar, phrase). It can be used for **generative modeling** or **reinforcement learning** with symbolic drum sequences. The architecture is modular, allowing for stepwise, barwise, and phrasewise processing, and optionally supports multi-track drums.

This model is compatible with the ```CompleteHierarchicalDrumDataset``` and its hierarchical metrics.

---

## Directory Structure and File Format

- **Input:** Batched tensors from ```CompleteHierarchicalDrumDataset``` DataLoader
  - ```"step"```: [batch_size, seq_len, num_drums]  
  - ```"bar"```: [batch_size, num_bars, num_drums]  
  - ```"phrase"```: [batch_size, num_phrases, num_drums]  
- **Output:** Predictions at the same hierarchical level as input
  - Step-level: next-step drum probabilities  
  - Bar/phrase-level: aggregated drum patterns for higher-level generation

---

## Class: ```HierarchicalDrumModel```

### Initialization

```python
HierarchicalDrumModel(
    num_drums,
    step_hidden_dim=128,
    bar_hidden_dim=128,
    phrase_hidden_dim=128,
    num_layers=2,
    dropout=0.1,
    use_transformer=False
)
```

**Parameters:**
- ```num_drums``` – number of drum instruments/tracks
- ```step_hidden_dim``` – hidden size for step-level RNN/MLP
- ```bar_hidden_dim``` – hidden size for bar-level RNN/MLP
- ```phrase_hidden_dim``` – hidden size for phrase-level RNN/MLP
- ```num_layers``` – number of layers per hierarchical level
- ```dropout``` – dropout rate
- ```use_transformer``` – if True, replaces RNNs with Transformer encoders

### Internal Variables

- ```self.step_rnn``` / ```self.bar_rnn``` / ```self.phrase_rnn``` – hierarchical RNN modules
- ```self.step_to_bar``` / ```self.bar_to_phrase``` – linear projection layers for hierarchical aggregation
- ```self.output_layer``` – final output layer producing drum probabilities per step
- ```self.activation``` – activation function (default: Sigmoid or Softmax)

---

## Forward Pass

The model supports hierarchical forward computation:

1. **Step-level processing**:
    - Input: step sequence [batch, seq_len, num_drums]
    - Processed with step-level RNN/Transformer
    - Output: hidden states for aggregation

2. **Bar-level processing**:
    - Aggregates step-level hidden states over each bar
    - Input: [batch, num_bars, step_hidden_dim]
    - Processed with bar-level RNN/Transformer
    - Output: bar-level representations

3. **Phrase-level processing**:
    - Aggregates bar-level hidden states over each phrase
    - Input: [batch, num_phrases, bar_hidden_dim]
    - Processed with phrase-level RNN/Transformer
    - Output: phrase-level representations

4. **Prediction**:
    - Combines step, bar, and phrase representations
    - Produces drum probabilities for the next step (or sequence generation)
    - Optional: hierarchical loss computation for RL rewards

```python
def forward(self, step_seq, bar_seq=None, phrase_seq=None):
    # step-level RNN -> bar aggregation -> phrase aggregation
    # return step_preds, bar_preds, phrase_preds
```

---

## Loss Functions

Supports multiple objectives:

1. **Step-level binary cross-entropy** for drum hits
2. **Bar/phrase-level BCE or MSE** for aggregated drum patterns
3. **Optional RL reward shaping**:
    - Metrics from ```CompleteHierarchicalDrumDataset``` (density, IOI, syncopation)
    - Hierarchical reward at step, bar, or phrase level

```python
def compute_loss(preds, targets, metrics=None):
    # step_loss + bar_loss + phrase_loss + optional metric-based reward
```

---

## Hierarchical Sampling / Generation

- Supports **autoregressive sampling**:
    - Step-level: predicts next drum hits sequentially
    - Bar-level: aggregates step predictions for bar summary
    - Phrase-level: aggregates bar predictions for phrase summary
- Can optionally **condition on previous phrases** or full-song context
- Can generate sequences of arbitrary length using sliding window approach

```python
def generate(self, start_seq, length=512, temperature=1.0):
    # autoregressive generation with hierarchical aggregation
```

---

## Design Considerations

1. **Hierarchical Modeling**:
    - Step → Bar → Phrase levels
    - Each level has independent hidden states for capturing temporal patterns

2. **Modular Architecture**:
    - RNNs or Transformers at each level
    - Linear projections for step→bar and bar→phrase aggregation

3. **Compatibility with RL**:
    - Allows integration of hierarchical metrics as rewards
    - Supports policy-gradient or value-based RL

4. **Regularization**:
    - Dropout layers between hierarchical levels
    - Gradient clipping for long sequences

5. **Flexibility**:
    - Can extend to multi-track (drums + bass) sequences
    - Supports optional section-level conditioning

---

## Example Usage

```python
from your_module import HierarchicalDrumModel, get_complete_rl_dataloader

npz_dir = "./train/"
dataloader = get_complete_rl_dataloader(npz_dir, seq_len=512, batch_size=2)

model = HierarchicalDrumModel(num_drums=10, step_hidden_dim=128)

for batch in dataloader:
    step_seq = batch["step"]
    bar_seq = batch["bar"]
    phrase_seq = batch["phrase"]

    step_preds, bar_preds, phrase_preds = model(step_seq, bar_seq, phrase_seq)
    # Compute loss using batch["metrics"] if needed
```
---

## Testing Recommendations

- **Unit tests**:
    - Forward pass shapes at each hierarchy level
    - Step/bar/phrase hidden states and output consistency
- **Integration tests**:
    - Compatibility with ```CompleteHierarchicalDrumDataset``` outputs
    - Autoregressive generation for short sequences
- **RL tests**:
    - Metric-based reward signals correctly backpropagated

---

## Summary

```HierarchicalDrumModel``` is a flexible, hierarchical neural network for drum sequence modeling. Its main features:

- Step → Bar → Phrase hierarchical modeling
- Optional Transformer or RNN backbones
- Compatible with multi-track drum sequences
- Supports RL reward integration via hierarchical metrics
- Modular design for generation, training, and analysis

Ideal for:

- Symbolic drum generation
- Reinforcement learning with hierarchical reward shaping
- Multi-scale rhythm analysis
