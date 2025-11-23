import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from model import DrumTransformer

# === Utilities ===
def load_drum_map(path="drum_map.yml"):
    with open(path, "r") as f:
        drum_map = yaml.safe_load(f)
    return drum_map

def group_drum_classes(drum_map):
    """
    Define groups based on instrument correlations
    """
    groups = {
        "splash": [drum_map['cymbal_splash']['class_id']],
        "hi_hat": [drum_map[k]['class_id'] for k in ['hi_hat_closed','hi_hat_open_1','hi_hat_open_2','hi_hat_pedal']],
        "kick": [drum_map[k]['class_id'] for k in ['kick_left','kick_center']],
        "rack_toms": [drum_map[f'racktom_{i}']['class_id'] for i in range(1,5)],
        "floor_toms": [drum_map[f'floortom_{i}']['class_id'] for i in range(1,3)],
        "crash": [drum_map[k]['class_id'] for k in ['cymbal_china','cymbal_crash_1','cymbal_crash_2']],
        "ride": [drum_map[k]['class_id'] for k in ['ride_1_bell','ride_1_bow','ride_1_edge','ride_1_choke']],
        "snare": [drum_map[k]['class_id'] for k in ['snare_hit','snare_rimshot','snare_sidestick']]
    }
    return groups

def top_k_logits(logits, k=8):
    if k <= 0:
        return logits
    v, ix = torch.topk(logits, k, dim=-1)
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, ix, v)
    return out

def sample_group(logits, group_ids, stochastic=True, threshold=0.2, mutually_exclusive=True):
    group_logits = logits[:, group_ids]

    # Handle NaNs
    if torch.isnan(group_logits).any():
        group_logits = torch.zeros_like(group_logits)

    if mutually_exclusive:
        probs = torch.softmax(group_logits, dim=-1)
        if stochastic:
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            idx = torch.argmax(probs, dim=-1)
        one_hot = torch.zeros_like(group_logits)
        one_hot[0, idx] = 1.0
        return one_hot, probs
    else:
        probs = torch.sigmoid(group_logits)
        if stochastic:
            one_hot = torch.bernoulli(probs)
        else:
            one_hot = (probs > threshold).float()
        return one_hot, probs

# === Generation Function ===
def generate_sequence(
    model,
    drum_map,
    seed_pattern=None,
    length_bars=8,
    steps_per_bar=16,
    temperature=0.7,
    top_k=2,
    prob_threshold=0.2,
    device="cpu",
    stochastic=True,
    visualize=True,
    max_hits_per_step=3,  # new: limit hits per step
):
    """
    Generate drum sequence with more realistic density and seed-aware patterns.
    """
    model.eval()
    model.to(device)
    D = model.num_classes
    total_steps = length_bars * steps_per_bar
    groups = group_drum_classes(drum_map)
    group_order = ["kick","snare","hi_hat","rack_toms","floor_toms","crash","ride","splash"]

    # Initialize context
    if seed_pattern is not None:
        seed_len = seed_pattern.shape[0]
        context = torch.tensor(seed_pattern, dtype=torch.float32).unsqueeze(0).to(device)
        remaining_steps = total_steps - seed_len
    else:
        context = torch.zeros((1, min(model.seq_len, total_steps), D), device=device)
        remaining_steps = total_steps

    generated = []

    for t in tqdm(range(remaining_steps), desc="Generating"):
        input_seq = context[:, -model.seq_len:, :]
        next_step = torch.zeros((1, D), device=device)

        with torch.no_grad():
            logits = model(input_seq)[:, -1, :]  # (1, D)
            logits = logits / max(temperature, 1e-5)
            logits = top_k_logits(logits, top_k)

            # Seed-aware boosting: encourage repeating previous hits, but modest
            if seed_pattern is not None and t < seed_pattern.shape[0]:
                prev_hits = context[:, t % seed_pattern.shape[0], :]
                logits += prev_hits * 1.0  # smaller boost

            for group_name in group_order:
                ids = groups[group_name]
                mutually_exclusive = group_name in ["kick","snare","hi_hat","crash","ride","splash"]

                sampled, _ = sample_group(
                    logits, ids, stochastic=stochastic,
                    threshold=prob_threshold, mutually_exclusive=mutually_exclusive
                )
                next_step[:, ids] = sampled

            # Enforce maximum number of hits per step
            hits = torch.sum(next_step)
            if hits > max_hits_per_step:
                ones_idx = (next_step > 0).nonzero(as_tuple=True)[1]
                remove_count = int(hits - max_hits_per_step)
                remove_idx = np.random.choice(ones_idx.cpu(), remove_count, replace=False)
                next_step[0, remove_idx] = 0

        generated.append(next_step.cpu().numpy())
        context = torch.cat([context, next_step.unsqueeze(1)], dim=1)

    # Combine seed + generated
    if seed_pattern is not None:
        sequence = np.vstack([seed_pattern, np.vstack(generated)])
    else:
        sequence = np.vstack(generated)

    if visualize:
        plt.figure(figsize=(12,4))
        plt.imshow(sequence.T, aspect='auto', cmap='hot', origin='lower')
        plt.colorbar(label='Hit')
        plt.xlabel('Step')
        plt.ylabel('Drum Track')
        plt.title('Generated Drum Sequence')
        plt.show()

    return sequence

# === Fine-tuning Utilities ===
def set_temperature(model, new_temp):
    model.temperature = new_temp

def load_model(model_path, num_classes, seq_len, device="cpu"):
    model = DrumTransformer(num_classes=num_classes, seq_len=seq_len)
    model.load(model_path, map_location=device)
    model.to(device)
    return model

# === CLI Entrypoint ===
def main(args):
    drum_map = load_drum_map(args.drum_map_path)
    model = load_model(args.model_path, args.num_classes, args.seq_len, device=args.device)

    # Optional seed
    seed_pattern = None
    if getattr(args, "use_seed", False):
        seed_steps = min(args.seq_len, args.steps_per_bar*2)
        seed_pattern = np.zeros((seed_steps, args.num_classes), dtype=np.float32)
        seed_pattern[0, 0] = 1  # Example: kick
        seed_pattern[seed_steps//2, 20] = 1  # Example: snare

    sequence = generate_sequence(
        model,
        drum_map,
        length_bars=args.length_bars,
        steps_per_bar=args.steps_per_bar,
        temperature=getattr(args,"temperature",0.7),
        top_k=getattr(args,"top_k",0),
        prob_threshold=getattr(args,"prob_threshold",0.2),
        device=args.device,
        seed_pattern=seed_pattern,
        stochastic=getattr(args,"stochastic",True),
        visualize=getattr(args,"visualize",True),
        # debug=getattr(args,"debug",False)
    )

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.savez(args.output_path, sequence=sequence)
    print(f"Generated sequence saved to {args.output_path}")
    print(f"Sequence shape: {sequence.shape}")
