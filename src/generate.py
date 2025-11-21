import os
import torch
import numpy as np
from model import DrumTransformer

def generate_sequence(
    model,
    seed: int = 42,
    length_bars: int = 64,
    steps_per_bar: int = 16,
    temperature: float = 1.0,
    device: str = "cpu"
):
    """
    Generate a drum sequence autoregressively.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    model.to(device)

    D = model.num_classes
    T_total = length_bars * steps_per_bar

    context_len = model.seq_len
    context = torch.zeros((1, min(context_len, T_total), D), device=device)

    generated = []

    for t in range(T_total):
        input_seq = context[:, -context_len:, :]
        with torch.no_grad():
            preds = model(input_seq)
        next_step_prob = preds[:, -1, :] / temperature
        next_step = torch.bernoulli(next_step_prob).to(torch.float32)
        generated.append(next_step.cpu().numpy())
        context = torch.cat([context, next_step.unsqueeze(1)], dim=1)
        if context.shape[1] > context_len:
            context = context[:, -context_len:, :]

    return np.vstack(generated)


def main(args):
    """
    Generate drum sequence using args passed from CLI.
    """
    model = DrumTransformer(num_classes=args.num_classes, seq_len=args.seq_len)
    model.load(args.model_path, map_location=args.device)

    sequence = generate_sequence(
        model,
        seed=args.seed,
        length_bars=args.length_bars,
        steps_per_bar=args.steps_per_bar,
        temperature=args.temperature,
        device=args.device
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.savez(args.output_path, sequence=sequence)
    print(f"Generated sequence saved to {args.output_path}")
    print(f"Sequence shape: {sequence.shape}")
