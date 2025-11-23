import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

class CompleteHierarchicalDrumDataset(Dataset):
    """
    Fully hierarchical drum dataset for RL.
    Computes metrics at step, bar, phrase, section, and song levels.
    """
    def __init__(self, npz_dir, seq_len=512, steps_per_bar=16, bars_per_phrase=4,
                 sections=None, augment=False, fast_threshold=4):
        self.npz_dir = npz_dir
        self.seq_len = seq_len
        self.steps_per_bar = steps_per_bar
        self.bars_per_phrase = bars_per_phrase
        self.phrase_len = steps_per_bar * bars_per_phrase
        self.sections = sections
        self.augment = augment
        self.fast_threshold = fast_threshold

        self.files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.sequences = []

        for f in tqdm(self.files, desc="Loading NPZ files"):
            try:
                data = np.load(f)['sequence'].astype(np.float32)
            except KeyError:
                print(f"Warning: 'sequence' not found in {f}, skipping.")
                continue
            if data.shape[0] < 1:
                print(f"Warning: sequence in {f} is empty, skipping.")
                continue
            self.sequences.append(data)

        if not self.sequences:
            raise ValueError(f"No valid sequences found in {npz_dir}.")

        # Precompute sample indices
        self.sample_indices = []
        for i, seq in enumerate(self.sequences):
            max_start = max(1, seq.shape[0] - seq_len)
            for start_idx in range(max_start):
                self.sample_indices.append((i, start_idx))

    def __len__(self):
        return len(self.sample_indices)

    # -------------------- Aggregation Helpers --------------------
    def _aggregate(self, seq, step_size):
        num_units = seq.shape[0] // step_size
        if num_units == 0:
            return seq.mean(axis=0, keepdims=True)
        return seq[:num_units*step_size].reshape(num_units, step_size, -1).mean(axis=1)

    def _hit_density(self, seq):
        total_hits = seq.sum(axis=1)
        avg_hits = total_hits.mean()
        drum_density = seq.mean(axis=0)
        max_hits = total_hits.max()
        return avg_hits, drum_density, max_hits

    def _fraction_dense_bars(self, seq, threshold=None):
        if threshold is None:
            threshold = seq.shape[1] / 2
        dense = (seq.sum(axis=1) > threshold).mean()
        return dense

    def _ioi_stats(self, seq):
        """Mean, variance, and motif histogram per drum."""
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

    def _co_occurrence(self, seq):
        # Fraction of steps with >1 drum hit
        return (seq.sum(axis=1) > 1).mean()

    def _n_grams(self, seq, n=4):
        ngrams = Counter()
        seq_int = (seq > 0).astype(int)
        for i in range(len(seq) - n + 1):
            key = tuple(seq_int[i:i+n].flatten())
            ngrams[key] += 1
        return ngrams

    def _fast_patterns(self, seq):
        counts = []
        for d in range(seq.shape[1]):
            hits = np.where(seq[:, d] > 0)[0]
            if len(hits) < 2:
                counts.append(0)
            else:
                intervals = np.diff(hits)
                counts.append((intervals <= self.fast_threshold).sum())
        return np.array(counts)

    def _transitions(self, seq):
        trans = np.zeros((seq.shape[1], seq.shape[1]))
        for t in range(seq.shape[0]-1):
            prev = seq[t]
            nxt = seq[t+1]
            for i in range(seq.shape[1]):
                if prev[i] > 0:
                    trans[i] += nxt
        return trans / (trans.sum(axis=1, keepdims=True) + 1e-6)

    def _syncopation(self, seq):
        weak_steps = np.arange(seq.shape[0]) % self.steps_per_bar != 0
        hits = seq[weak_steps].sum()
        total = seq.sum()
        return hits / (total + 1e-6)

    def _phrase_structure(self, seq):
        """Hits per bar within phrase. Handles short sequences."""
        num_bars = seq.shape[0] // self.steps_per_bar
        if num_bars == 0:
            # Return zeros for all drums if sequence too short
            return np.zeros((1, seq.shape[1]), dtype=seq.dtype)
        bars = seq[:num_bars*self.steps_per_bar].reshape(num_bars, self.steps_per_bar, -1)
        return bars.sum(axis=1)  # shape: (bars, drums)


    def _section_metrics(self, seq):
        if self.sections is None:
            return None
        metrics = []
        for start, end in self.sections:
            sec_seq = seq[start:end]
            avg_hits, drum_density, max_hits = self._hit_density(sec_seq)
            dense_frac = self._fraction_dense_bars(sec_seq)
            metrics.append({
                "avg_hits": avg_hits,
                "drum_density": drum_density,
                "max_hits": max_hits,
                "dense_frac": dense_frac
            })
        return metrics

    def _song_metrics(self, seq):
        avg_hits, drum_density, max_hits = self._hit_density(seq)
        step_entropy = -np.sum(seq.mean(axis=0) * np.log(seq.mean(axis=0)+1e-6))
        phrase_len = self.phrase_len
        # Phrase-wise entropy
        num_phrases = seq.shape[0] // phrase_len
        phrase_entropy = []
        for i in range(num_phrases):
            p = seq[i*phrase_len:(i+1)*phrase_len].mean(axis=0)
            phrase_entropy.append(-np.sum(p*np.log(p+1e-6)))
        phrase_entropy = np.mean(phrase_entropy) if phrase_entropy else 0.0
        ngram_diversity = len(self._n_grams(seq, n=4))
        return {
            "avg_hits": avg_hits,
            "drum_density": drum_density,
            "max_hits": max_hits,
            "step_entropy": step_entropy,
            "phrase_entropy": phrase_entropy,
            "ngram_diversity": ngram_diversity
        }

    # -------------------- Get Item --------------------
    def __getitem__(self, idx):
        seq_idx, start_idx = self.sample_indices[idx]
        seq = self.sequences[seq_idx]
        step_seq = seq[start_idx:start_idx + self.seq_len]

        if self.augment:
            shift = np.random.randint(-2,3)
            step_seq = np.roll(step_seq, shift, axis=0)

        bar_seq = self._aggregate(step_seq, self.steps_per_bar)
        phrase_seq = self._aggregate(step_seq, self.phrase_len)

        # Step metrics
        step_avg_hits, step_drum_density, step_max_hits = self._hit_density(step_seq)
        step_dense_frac = self._fraction_dense_bars(step_seq)
        step_ioi_stats, step_ioi_hist = self._ioi_stats(step_seq)
        step_co = self._co_occurrence(step_seq)
        step_fast = self._fast_patterns(step_seq)
        step_trans = self._transitions(step_seq)
        step_sync = self._syncopation(step_seq)
        step_ngrams = self._n_grams(step_seq, n=4)

        # Bar metrics
        bar_avg_hits, bar_drum_density, bar_max_hits = self._hit_density(bar_seq)
        bar_dense_frac = self._fraction_dense_bars(bar_seq)
        bar_ioi_stats, bar_ioi_hist = self._ioi_stats(bar_seq)
        bar_co = self._co_occurrence(bar_seq)
        bar_fast = self._fast_patterns(bar_seq)
        bar_trans = self._transitions(bar_seq)
        bar_sync = self._syncopation(bar_seq)
        bar_ngrams = self._n_grams(bar_seq, n=4)  # Can extend to 2–4 bars

        # Phrase metrics
        phrase_avg_hits, phrase_drum_density, phrase_max_hits = self._hit_density(phrase_seq)
        phrase_dense_frac = self._fraction_dense_bars(phrase_seq)
        phrase_ioi_stats, phrase_ioi_hist = self._ioi_stats(phrase_seq)
        phrase_co = self._co_occurrence(phrase_seq)
        phrase_fast = self._fast_patterns(phrase_seq)
        phrase_trans = self._transitions(phrase_seq)
        phrase_sync = self._syncopation(phrase_seq)
        phrase_ngrams = self._n_grams(phrase_seq, n=self.bars_per_phrase*4)
        phrase_hits_per_bar = self._phrase_structure(phrase_seq)

        # Section metrics
        section_metrics = self._section_metrics(step_seq)

        # Song metrics
        song_metrics = self._song_metrics(step_seq)

        return {
            "step": torch.from_numpy(step_seq).float(),
            "bar": torch.from_numpy(bar_seq).float(),
            "phrase": torch.from_numpy(phrase_seq).float(),
            "metrics": {
                "step": {
                    "avg_hits": step_avg_hits, "drum_density": step_drum_density,
                    "max_hits": step_max_hits, "dense_frac": step_dense_frac,
                    "ioi_stats": step_ioi_stats, "ioi_hist": step_ioi_hist,
                    "co": step_co, "fast": step_fast, "trans": step_trans,
                    "sync": step_sync, "ngrams": step_ngrams
                },
                "bar": {
                    "avg_hits": bar_avg_hits, "drum_density": bar_drum_density,
                    "max_hits": bar_max_hits, "dense_frac": bar_dense_frac,
                    "ioi_stats": bar_ioi_stats, "ioi_hist": bar_ioi_hist,
                    "co": bar_co, "fast": bar_fast, "trans": bar_trans,
                    "sync": bar_sync, "ngrams": bar_ngrams
                },
                "phrase": {
                    "avg_hits": phrase_avg_hits, "drum_density": phrase_drum_density,
                    "max_hits": phrase_max_hits, "dense_frac": phrase_dense_frac,
                    "ioi_stats": phrase_ioi_stats, "ioi_hist": phrase_ioi_hist,
                    "co": phrase_co, "fast": phrase_fast, "trans": phrase_trans,
                    "sync": phrase_sync, "ngrams": phrase_ngrams,
                    "hits_per_bar": phrase_hits_per_bar
                },
                "section": section_metrics,
                "song": song_metrics
            }
        }

# -------------------- Dataloader --------------------
def get_complete_rl_dataloader(npz_dir, seq_len=512, steps_per_bar=16, bars_per_phrase=4,
                               batch_size=16, shuffle=True, augment=False, num_workers=0):
    dataset = CompleteHierarchicalDrumDataset(npz_dir, seq_len=seq_len,
                                              steps_per_bar=steps_per_bar,
                                              bars_per_phrase=bars_per_phrase,
                                              augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=lambda x: {
                          "step": torch.nn.utils.rnn.pad_sequence([d["step"] for d in x], batch_first=True),
                          "bar": torch.nn.utils.rnn.pad_sequence([d["bar"] for d in x], batch_first=True),
                          "phrase": torch.nn.utils.rnn.pad_sequence([d["phrase"] for d in x], batch_first=True),
                          "metrics": [d["metrics"] for d in x]
                      })

# -------------------- Quick Test --------------------
if __name__ == "__main__":
    npz_dir = "./train/"
    dataloader = get_complete_rl_dataloader(npz_dir, seq_len=512, batch_size=2)
    for batch in dataloader:
        print("Step:", batch["step"].shape)
        print("Bar:", batch["bar"].shape)
        print("Phrase:", batch["phrase"].shape)
        print("Metrics (step avg hits):", batch["metrics"][0]["step"]["avg_hits"])
        print("Metrics (phrase hits per bar):", batch["metrics"][0]["phrase"]["hits_per_bar"])
        print("Metrics (song ngram diversity):", batch["metrics"][0]["song"]["ngram_diversity"])
        break
