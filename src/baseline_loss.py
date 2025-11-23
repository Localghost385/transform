# expects dataset.cache or dataloader that yields 'step' tensors shaped (B,T,D)
import torch, numpy as np
from dataset import CachedHierarchicalDrumDataset

dataset = CachedHierarchicalDrumDataset(
    npz_dir="data/train",
    seq_len=512,
    augment=False
)

# get full dataset steps as numpy array (N_samples, T, D)
all_steps = []
for i in range(len(dataset)):
    s = dataset[i]['step'].numpy()  # or use your cached array
    all_steps.append(s)
all_steps = np.stack(all_steps, axis=0)  # (N, T, D)
p = all_steps.mean(axis=(0,1))  # per-drum probability, shape (D,)

# baseline BCE
eps = 1e-12
p_clipped = np.clip(p, eps, 1-eps)
bce_per_drum = -(p_clipped * np.log(p_clipped) + (1-p_clipped) * np.log(1-p_clipped))
baseline_loss = bce_per_drum.mean()
print("Baseline BCE (mean-predictor):", baseline_loss)
