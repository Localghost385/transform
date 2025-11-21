import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class DrumDataset(Dataset):
    """
    PyTorch Dataset for drum sequences stored as .npz files.
    Each .npz file should contain a 'sequence' array of shape (T, D)
    where T is the number of timesteps and D is the number of drum classes.
    """
    def __init__(self, npz_dir, seq_len=512, augment=False):
        self.npz_dir = npz_dir
        self.seq_len = seq_len
        self.augment = augment
        
        # Load all .npz files
        self.files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        
        self.sequences = []
        for f in tqdm(self.files, desc="Loading NPZ files"):
            try:
                data = np.load(f)['sequence']  # shape: (T, D)
            except KeyError:
                print(f"Warning: 'sequence' not found in {f}, skipping.")
                continue

            if data.shape[0] < 1:
                print(f"Warning: sequence in {f} is empty, skipping.")
                continue
            elif data.shape[0] < 2:
                print(f"Warning: sequence in {f} is very short ({data.shape[0]} timesteps).")

            self.sequences.append(data.astype(np.float32))

        if not self.sequences:
            raise ValueError(f"No valid sequences found in {npz_dir}.")

        # Precompute sample indices for slicing sequences
        self.sample_indices = []
        for i, seq in enumerate(self.sequences):
            if seq.shape[0] <= seq_len:
                self.sample_indices.append((i, 0))  # one sample
            else:
                for start_idx in range(seq.shape[0] - seq_len):
                    self.sample_indices.append((i, start_idx))

        if not self.sample_indices:
            raise ValueError("No samples available: all sequences are too short.")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.sample_indices[idx]
        seq = self.sequences[seq_idx]
        
        if seq.shape[0] <= self.seq_len:
            if seq.shape[0] == 1:
                input_seq = seq.copy()
                target_seq = seq.copy()
            else:
                input_seq = seq[:-1]
                target_seq = seq[1:]
        else:
            input_seq = seq[start_idx:start_idx + self.seq_len]
            target_seq = seq[start_idx + 1:start_idx + self.seq_len + 1]

        # Optional augmentation: small random temporal shift
        if self.augment:
            shift = np.random.randint(-2, 3)
            input_seq = np.roll(input_seq, shift, axis=0)

        # Convert to torch tensors
        input_seq = torch.from_numpy(input_seq).float()
        target_seq = torch.from_numpy(target_seq).float()
        return input_seq, target_seq


def get_dataloader(npz_dir, seq_len=512, batch_size=16, shuffle=True, augment=False, num_workers=0):
    dataset = DrumDataset(npz_dir, seq_len=seq_len, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    # Quick test
    npz_dir = "../data/processed_npz"
    dataloader = get_dataloader(npz_dir, seq_len=512, batch_size=4)
    for x, y in dataloader:
        print("Input:", x.shape)
        print("Target:", y.shape)
        break
