import argparse
import os
import torch
from preprocess import process_all_files
from train import main as train_main
from generate import main as generate_main


def main():
    parser = argparse.ArgumentParser(description="Drum Transformer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------------
    # PREPROCESS
    # -------------------------
    preprocess_parser = subparsers.add_parser("preprocess", help="Convert MIDI files to NPZ")
    preprocess_parser.add_argument("--input_dir", type=str, required=True)
    preprocess_parser.add_argument("--output_dir", type=str, required=True)
    preprocess_parser.add_argument("--drum_map", type=str, required=True)
    preprocess_parser.add_argument("--steps_per_bar", type=int, default=16)

    # -------------------------
    # TRAIN
    # -------------------------
    train_parser = subparsers.add_parser("train", help="Train Drum Transformer")

    # Required
    train_parser.add_argument("--train_dir", type=str, required=True)
    train_parser.add_argument("--val_dir", type=str, required=True)

    # Model architecture
    train_parser.add_argument("--num_classes", type=int, default=23)
    train_parser.add_argument("--seq_len", type=int, default=512)
    train_parser.add_argument("--d_model", type=int, default=512)
    train_parser.add_argument("--nhead", type=int, default=8)
    train_parser.add_argument("--num_layers", type=int, default=8)
    train_parser.add_argument("--ff_dim", type=int, default=2048)
    train_parser.add_argument("--dropout", type=float, default=0.1)

    # Training settings
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.01)
    train_parser.add_argument("--clip_grad", type=float, default=1.0)

    # Performance settings
    train_parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision")
    train_parser.add_argument("--use_compile", action="store_true", help="Enable torch.compile()")
    train_parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--num_threads", type=int, default=4)

    # LR warmup
    train_parser.add_argument("--warmup_steps", type=int, default=None)
    train_parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # Checkpoints
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    train_parser.add_argument("--save_every_steps", type=int, default=500)
    train_parser.add_argument("--resume_from", type=str, default=None)

    # Device
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    train_parser.add_argument("--device", type=str, default=default_device)

    # -------------------------
    # GENERATE
    # -------------------------
    generate_parser = subparsers.add_parser("generate", help="Generate drum sequences")
    generate_parser.add_argument("--model_path", type=str, required=True)
    generate_parser.add_argument("--output_path", type=str, default="generated_sequence.npz")
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--length_bars", type=int, default=64)
    generate_parser.add_argument("--steps_per_bar", type=int, default=16)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--device", type=str, default=default_device)
    generate_parser.add_argument("--num_classes", type=int, default=23)
    generate_parser.add_argument("--seq_len", type=int, default=512)

    args = parser.parse_args()

    # -------------------------
    # COMMAND ROUTER
    # -------------------------
    if args.command == "preprocess":
        os.makedirs(args.output_dir, exist_ok=True)
        process_all_files(args.input_dir, args.output_dir, args.drum_map, steps_per_bar=args.steps_per_bar)

    elif args.command == "train":
        train_main(args)

    elif args.command == "generate":
        generate_main(args)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
