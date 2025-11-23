import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter

class CachedHierarchicalDrumDataset(Dataset):
    """
    Fully cached hierarchical drum dataset.
    Precomputes step/bar/phrase sequences and all metrics in memory.
    """
    def __init__(self, npz_dir, seq_len=512, steps_per_bar=16, bars_per_phrase=4,
                 augment=False, fast_threshold=4, verbose=True):
        self.seq_len = seq_len
        self.steps_per_bar = steps_per_bar
        self.bars_per_phrase = bars_per_phrase
        self.phrase_len = steps_per_bar * bars_per_phrase
        self.augment = augment
        self.fast_threshold = fast_threshold

        # Load NPZ files
        files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        sequences = []
        for f in tqdm(files, desc="Loading NPZ files"):
            try:
                data = np.load(f)['sequence'].astype(np.float32)
                if data.shape[0] < 1:
                    continue
                sequences.append(data)
            except KeyError:
                continue

        if len(sequences) == 0:
            raise ValueError(f"No valid sequences found in {npz_dir}.")

        # -------------------- Precompute cache --------------------
        self.cache = []
        for seq_idx, seq in enumerate(tqdm(sequences, desc="Caching sequences")):
            max_start = max(1, seq.shape[0] - seq_len)
            for start_idx in range(max_start):
                step_seq = seq[start_idx:start_idx + seq_len]
                bar_seq = self._aggregate(step_seq, steps_per_bar)
                phrase_seq = self._aggregate(step_seq, self.phrase_len)
                metrics = self.compute_all_metrics(step_seq, bar_seq, phrase_seq)
                # Convert sequences to torch tensors and cache
                self.cache.append({
                    "step": torch.from_numpy(step_seq).float(),
                    "bar": torch.from_numpy(bar_seq).float(),
                    "phrase": torch.from_numpy(phrase_seq).float(),
                    "metrics": metrics
                })
        if verbose:
            print(f"Cached {len(self.cache)} samples in memory.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        sample = self.cache[idx]
        # Apply augmentation if enabled
        if self.augment:
            step = sample["step"]
            shift = np.random.randint(-2, 3)
            step = torch.roll(step, shifts=shift, dims=0)
            # Recompute bar/phrase from shifted step
            bar = self._aggregate(step.numpy(), self.steps_per_bar)
            phrase = self._aggregate(step.numpy(), self.phrase_len)
            bar = torch.from_numpy(bar).float()
            phrase = torch.from_numpy(phrase).float()
            return {"step": step, "bar": bar, "phrase": phrase, "metrics": sample["metrics"]}
        return sample

    # -------------------- Aggregation --------------------
    def _aggregate(self, seq, step_size):
        num_units = seq.shape[0] // step_size
        if num_units == 0:
            return seq.mean(axis=0, keepdims=True)
        return seq[:num_units*step_size].reshape(num_units, step_size, -1).mean(axis=1)

    # -------------------- Metric computation --------------------
    def compute_all_metrics(self, step_seq, bar_seq, phrase_seq):
        def _hit_density(seq):
            total_hits = seq.sum(axis=1)
            avg_hits = total_hits.mean()
            drum_density = seq.mean(axis=0)
            max_hits = total_hits.max()
            return avg_hits, drum_density, max_hits

        def _fraction_dense_bars(seq, threshold=None):
            if threshold is None:
                threshold = seq.shape[1] / 2
            dense = (seq.sum(axis=1) > threshold).mean()
            return dense

        def _ioi_stats(seq):
            ioi_list = []
            ioi_hist = []
            for d in range(seq.shape[1]):
                hits = np.where(seq[:, d] > 0)[0]
                if len(hits) < 2:
                    ioi_list.append((0.0, 0.0))
                    ioi_hist.append(np.array([]))
                else:
                    intervals = np.diff(hits)
                    ioi_list.append((intervals.mean(), intervals.var()))
                    hist, _ = np.histogram(intervals, bins=np.arange(1, 33))
                    ioi_hist.append(hist)
            return np.array(ioi_list), ioi_hist

        def _co_occurrence(seq):
            return (seq.sum(axis=1) > 1).mean()

        def _fast_patterns(seq):
            counts = []
            for d in range(seq.shape[1]):
                hits = np.where(seq[:, d] > 0)[0]
                if len(hits) < 2:
                    counts.append(0)
                else:
                    intervals = np.diff(hits)
                    counts.append((intervals <= self.fast_threshold).sum())
            return np.array(counts)

        def _transitions(seq):
            trans = np.zeros((seq.shape[1], seq.shape[1]))
            for t in range(seq.shape[0]-1):
                prev = seq[t]
                nxt = seq[t+1]
                for i in range(seq.shape[1]):
                    if prev[i] > 0:
                        trans[i] += nxt
            return trans / (trans.sum(axis=1, keepdims=True) + 1e-6)

        def _syncopation(seq):
            weak_steps = np.arange(seq.shape[0]) % self.steps_per_bar != 0
            hits = seq[weak_steps].sum()
            total = seq.sum()
            return hits / (total + 1e-6)

        def _phrase_structure(seq):
            num_bars = seq.shape[0] // self.steps_per_bar
            if num_bars == 0:
                return np.zeros((1, seq.shape[1]), dtype=seq.dtype)
            bars = seq[:num_bars*self.steps_per_bar].reshape(num_bars, self.steps_per_bar, -1)
            return bars.sum(axis=1)

        # Step metrics
        step_avg_hits, step_drum_density, step_max_hits = _hit_density(step_seq)
        step_dense_frac = _fraction_dense_bars(step_seq)
        step_ioi_stats, step_ioi_hist = _ioi_stats(step_seq)
        step_co = _co_occurrence(step_seq)
        step_fast = _fast_patterns(step_seq)
        step_trans = _transitions(step_seq)
        step_sync = _syncopation(step_seq)

        # Bar metrics
        bar_avg_hits, bar_drum_density, bar_max_hits = _hit_density(bar_seq)
        bar_dense_frac = _fraction_dense_bars(bar_seq)
        bar_ioi_stats, bar_ioi_hist = _ioi_stats(bar_seq)
        bar_co = _co_occurrence(bar_seq)
        bar_fast = _fast_patterns(bar_seq)
        bar_trans = _transitions(bar_seq)
        bar_sync = _syncopation(bar_seq)

        # Phrase metrics
        phrase_avg_hits, phrase_drum_density, phrase_max_hits = _hit_density(phrase_seq)
        phrase_dense_frac = _fraction_dense_bars(phrase_seq)
        phrase_ioi_stats, phrase_ioi_hist = _ioi_stats(phrase_seq)
        phrase_co = _co_occurrence(phrase_seq)
        phrase_fast = _fast_patterns(phrase_seq)
        phrase_trans = _transitions(phrase_seq)
        phrase_sync = _syncopation(phrase_seq)
        phrase_hits_per_bar = _phrase_structure(phrase_seq)

        return {
            "step": {
                "avg_hits": step_avg_hits, "drum_density": step_drum_density,
                "max_hits": step_max_hits, "dense_frac": step_dense_frac,
                "ioi_stats": step_ioi_stats, "ioi_hist": step_ioi_hist,
                "co": step_co, "fast": step_fast, "trans": step_trans,
                "sync": step_sync
            },
            "bar": {
                "avg_hits": bar_avg_hits, "drum_density": bar_drum_density,
                "max_hits": bar_max_hits, "dense_frac": bar_dense_frac,
                "ioi_stats": bar_ioi_stats, "ioi_hist": bar_ioi_hist,
                "co": bar_co, "fast": bar_fast, "trans": bar_trans,
                "sync": bar_sync
            },
            "phrase": {
                "avg_hits": phrase_avg_hits, "drum_density": phrase_drum_density,
                "max_hits": phrase_max_hits, "dense_frac": phrase_dense_frac,
                "ioi_stats": phrase_ioi_stats, "ioi_hist": phrase_ioi_hist,
                "co": phrase_co, "fast": phrase_fast, "trans": phrase_trans,
                "sync": phrase_sync,
                "hits_per_bar": phrase_hits_per_bar
            }
        }
