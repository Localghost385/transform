
# Technical Design Document: train.py — Hierarchical Drum Transformer Training Pipeline

## Overview
The training script implements the **full training loop** for the Hierarchical Drum Transformer, including:

- Step-level supervised training with BCE loss
- Optional Reinforcement Learning (RL) reward shaping
- Checkpointing (model, optimizer, scheduler)
- Logging (loss, RL metrics)
- Mixed-precision training
- Multi-GPU support (DataParallel / DistributedDataParallel compatible)

The design aligns with the higher-level project specification and ensures modular, reproducible, and scalable training.

---

## 1. Input / Data

### 1.1 Dataset
- ```CompleteHierarchicalDrumDataset``` NPZ files
- Returns:
  - ```step``` tensor: [batch_size, seq_len, num_drums]
  - ```bar``` tensor (optional): [batch_size, num_bars, num_drums]
  - ```phrase``` tensor (optional): [batch_size, num_phrases, num_drums]
  - ```metrics``` dict (optional): hierarchical metrics for RL reward shaping

### 1.2 Dataloader
- Batch size configurable
- Supports shuffling, drop_last=True for full batch consistency
- Optionally supports distributed sampling for multi-GPU

---

## 2. Model

- ```HierarchicalDrumModel``` (step/bar/phrase hierarchy)
- Configurable parameters:
  - ```num_drums```, ```d_model```, ```num_layers```, ```nhead```
- Multi-GPU via:
  - ```nn.DataParallel``` for single-node multi-GPU
  - Optionally ```DistributedDataParallel``` (future extension)

---

## 3. Loss Function

### 3.1 Supervised Loss
- Step-level BCEWithLogitsLoss:
  ```
  Loss_step = mean(BCE(logits[t], step_target[t]))
  ```

### 3.2 Reinforcement Learning Loss (Optional)
- Reward computed from hierarchical metrics:
  - density
  - syncopation
  - IOI variance
  - user-defined composite reward
- Combined loss:
  ```
  Total_loss = Step_loss + RL_weight * (-reward)
  ```
- RL is optional and can be toggled with command-line flag

---

## 4. Training Loop

### 4.1 Per-Epoch Flow
1. Set model to ```train()```
2. Iterate over batches:
   - Move data to device
   - Forward pass through model
   - Compute step BCE loss
   - Compute RL reward if enabled
   - Backward pass with gradient scaling
   - Gradient clipping
   - Optimizer step
   - Scheduler step
   - Log metrics (loss, RL reward)

### 4.2 Checkpointing
- Saves every epoch:
  - ```epoch``` number
  - ```model.state_dict()```
  - ```optimizer.state_dict()```
  - ```scheduler.state_dict()```
- Directory configurable
- Checkpoint naming: ```epoch_{epoch}.pt```

### 4.3 Logging
- Uses modular logger
- Tracks:
  - epoch
  - batch step
  - step BCE loss
  - RL reward (if enabled)
- Log interval configurable

---

## 5. Optimizer & Scheduler

- Optimizer: ```AdamW``` with weight decay
- Learning rate scheduler: ```cosine``` or user-defined
- Mixed-precision training supported with ```torch.cuda.amp.GradScaler```
- Gradient clipping to stabilize long-sequence training

---

## 6. Multi-GPU Support

### 6.1 Single-node Multi-GPU
- ```nn.DataParallel(model)``` wraps the model
- Automatic device allocation
- Loss backward passes accumulate across GPUs

### 6.2 Distributed / Future Extension
- DDP recommended for large datasets/models
- Each GPU handles subset of data
- Synchronizes gradients across nodes
- Requires ```torch.distributed.launch``` or ```torchrun```

---

## 7. Configuration / CLI

Command-line arguments include:

- ```--data_dir```: path to processed NPZ files
- ```--batch_size```: training batch size
- ```--seq_len```: context window length
- ```--epochs```: number of training epochs
- ```--lr```: learning rate
- ```--weight_decay```: optimizer weight decay
- ```--grad_clip```: gradient clipping value
- ```--checkpoint_dir```: directory to save checkpoints
- ```--rl_enabled```: toggle RL reward shaping
- ```--rl_weight```: weight of RL term in loss
- ```--device```: 'cuda' or 'cpu'
- ```--num_gpus```: number of GPUs for training
- ```--log_interval```: logging frequency per batch

---

## 8. Mixed-Precision Training

- Uses ```torch.cuda.amp.autocast()``` for forward pass
- Uses ```torch.cuda.amp.GradScaler()``` for scaling gradients
- Reduces memory usage and increases throughput on modern GPUs

---

## 9. Reward Computation (RL)

- User-defined function
- Receives:
  - ```logits```: model outputs
  - ```metrics```: hierarchical features
- Returns scalar reward tensor
- Can combine multiple metrics into a weighted reward

---

## 10. Checkpoint Loading / Resume

- ```save_checkpoint(model, optimizer, scheduler, epoch, path)``` saves model state
- Checkpoints can be reloaded for:
  - resuming training
  - evaluation
  - generation

---

## 11. Example Training Invocation

```
python train.py --data_dir ./data/processed --batch_size 4 --seq_len 512 --epochs 100 --lr 1e-4 --checkpoint_dir ./checkpoints --rl_enabled True --rl_weight 0.1 --num_gpus 2
```
---

## 12. Summary

The training pipeline supports:

- Supervised BCE training for hierarchical drum sequences
- Optional RL reward shaping from hierarchical metrics
- Checkpointing and resume support
- Mixed-precision and multi-GPU training
- Configurable logging and training hyperparameters

This design ensures a **scalable, reproducible, and modular training system** aligned with the full drum Transformer project specification.
