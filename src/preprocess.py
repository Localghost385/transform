"""
Preprocessing script for Drum Transformer.
Converts raw MIDI drum tracks into quantized, class-mapped NumPy arrays for training.
"""

import os
import numpy as np
import mido
import yaml

# -------------------
# Core Functions
# -------------------

def load_drum_map(yaml_path: str) -> dict:
    """Load the input drum map from YAML file."""
    with open(yaml_path, 'r') as f:
        drum_map_yaml = yaml.safe_load(f)
    
    pitch_to_class = {}
    for drum_name, info in drum_map_yaml.items():
        class_id = info['class_id']
        for pitch in info['pitches']:
            pitch_to_class[pitch] = class_id
    return pitch_to_class


def load_midi_file(midi_path: str) -> mido.MidiFile:
    """Load a MIDI file using mido."""
    return mido.MidiFile(midi_path)


def extract_drum_notes(mid: mido.MidiFile) -> list:
    """Extract all drum note events (channel 0)."""
    drum_notes = []
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.channel == 0 and msg.velocity > 0:
                drum_notes.append((abs_time, msg.note, msg.velocity))
    return drum_notes


def quantize_notes(drum_notes: list, ticks_per_bar: int, steps_per_bar: int) -> list:
    """Quantize note timings to fixed steps per bar."""
    step_size = ticks_per_bar / steps_per_bar
    quantized = []
    for abs_tick, pitch, velocity in drum_notes:
        step_idx = int(round(abs_tick / step_size))
        quantized.append((step_idx, pitch, velocity))
    return quantized


def notes_to_multi_hot(quantized_notes: list, pitch_to_class: dict) -> np.ndarray:
    """Convert quantized notes to a multi-hot 2D array."""
    if not quantized_notes:
        num_classes = max(pitch_to_class.values()) + 1
        return np.zeros((0, num_classes), dtype=np.uint8)
    
    max_step = max(step for step, _, _ in quantized_notes)
    num_classes = max(pitch_to_class.values()) + 1
    X = np.zeros((max_step + 1, num_classes), dtype=np.uint8)
    
    for step, pitch, _ in quantized_notes:
        class_id = pitch_to_class.get(pitch)
        if class_id is not None:
            if class_id >= num_classes:
                print(f"Warning: class_id {class_id} exceeds num_classes {num_classes}")
                continue
            X[step, class_id] = 1
    return X


def save_npz(output_path: str, X: np.ndarray):
    """Save the multi-hot array as compressed NPZ."""
    np.savez_compressed(output_path, sequence=X)


def process_single_file(midi_path: str, output_dir: str, pitch_to_class: dict, steps_per_bar: int = 16):
    """Process one MIDI file: load → extract → quantize → multi-hot → save."""
    mid = load_midi_file(midi_path)
    drum_notes = extract_drum_notes(mid)
    ticks_per_bar = mid.ticks_per_beat * 4  # assume 4/4
    quantized = quantize_notes(drum_notes, ticks_per_bar, steps_per_bar)
    X = notes_to_multi_hot(quantized, pitch_to_class)  # <-- FIXED here
    
    filename = os.path.splitext(os.path.basename(midi_path))[0] + ".npz"
    output_path = os.path.join(output_dir, filename)
    save_npz(output_path, X)


def process_all_files(input_dir: str, output_dir: str, drum_map_path: str, steps_per_bar: int = 16):
    """Process all MIDI files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    pitch_to_class = load_drum_map(drum_map_path)
    
    for file in os.listdir(input_dir):
        if file.lower().endswith((".mid", ".midi")):
            midi_path = os.path.join(input_dir, file)
            process_single_file(midi_path, output_dir, pitch_to_class, steps_per_bar)


# -------------------
# CLI
# -------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess MIDI drum files for Drum Transformer")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--drum_map", type=str, required=True)
    parser.add_argument("--steps_per_bar", type=int, default=16)

    args = parser.parse_args()

    process_all_files(args.input_dir, args.output_dir, args.drum_map, args.steps_per_bar)
