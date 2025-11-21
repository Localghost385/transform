import argparse
import os
from preprocess import process_all_files
from train import main as train_main
from generate import main as generate_main

def main():
    parser = argparse.ArgumentParser(description="Drum Transformer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command: preprocess / train / generate")

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Convert MIDI files to NPZ")
    preprocess_parser.add_argument("--input_dir", type=str, required=True)
    preprocess_parser.add_argument("--output_dir", type=str, required=True)
    preprocess_parser.add_argument("--drum_map", type=str, required=True)
    preprocess_parser.add_argument("--steps_per_bar", type=int, default=16)

    # Train
    train_parser = subparsers.add_parser("train", help="Train Drum Transformer")
    train_parser.add_argument("--train_dir", type=str, required=True)
    train_parser.add_argument("--val_dir", type=str, required=True)
    train_parser.add_argument("--num_classes", type=int, default=23)
    train_parser.add_argument("--seq_len", type=int, default=512)
    train_parser.add_argument("--d_model", type=int, default=512)
    train_parser.add_argument("--nhead", type=int, default=8)
    train_parser.add_argument("--num_layers", type=int, default=8)
    train_parser.add_argument("--ff_dim", type=int, default=2048)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--clip_grad", type=float, default=1.0)
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    train_parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    # Generate
    generate_parser = subparsers.add_parser("generate", help="Generate drum sequences")
    generate_parser.add_argument("--model_path", type=str, required=True)
    generate_parser.add_argument("--output_path", type=str, default="generated_sequence.npz")
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--length_bars", type=int, default=64)
    generate_parser.add_argument("--steps_per_bar", type=int, default=16)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    generate_parser.add_argument("--num_classes", type=int, default=23)
    generate_parser.add_argument("--seq_len", type=int, default=512)

    args = parser.parse_args()

    if args.command == "preprocess":
        os.makedirs(args.output_dir, exist_ok=True)
        process_all_files(args.input_dir, args.output_dir, args.drum_map, steps_per_bar=args.steps_per_bar)
    elif args.command == "train":
        from train import main as train_main
        train_main(args)  # pass parsed args
    elif args.command == "generate":
        from generate import main as generate_main
        generate_main(args)  # pass parsed args
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
