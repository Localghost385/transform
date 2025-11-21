import os
import numpy as np

def inspect_npz_dir(npz_dir, n_preview=5):
    """
    Inspect all .npz files in a directory.
    
    Args:
        npz_dir (str): Path to folder containing .npz files.
        n_preview (int): Number of timesteps to print from each sequence.
    """
    files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    if not files:
        print(f"No .npz files found in {npz_dir}")
        return

    for f in files:
        path = os.path.join(npz_dir, f)
        try:
            data = np.load(path)
            if 'sequence' not in data:
                print(f"{f}: WARNING - no 'sequence' key found!")
                continue
            seq = data['sequence']
            print(f"{f}: shape={seq.shape}, dtype={seq.dtype}")
            print(f"First {n_preview} timesteps:\n{seq[:n_preview]}")
            print("-" * 40)
        except Exception as e:
            print(f"{f}: ERROR loading file: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug NPZ drum dataset files")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing NPZ files")
    parser.add_argument("--preview", type=int, default=5, help="Number of timesteps to preview")
    args = parser.parse_args()

    inspect_npz_dir(args.dir, n_preview=args.preview)
