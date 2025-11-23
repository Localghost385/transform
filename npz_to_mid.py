"""
Utility: Convert a multi-hot NPZ drum sequence back into a MIDI file.
"""

import os
import yaml
import numpy as np
import mido


def load_class_to_pitch_map(yaml_path: str) -> dict:
    """
    Invert the preprocessing drum map:
    drum_name:
        class_id: int
        pitches: [list of MIDI pitches]
    
    Returns:
        class_id -> representative MIDI pitch
    """
    with open(yaml_path, "r") as f:
        drum_map = yaml.safe_load(f)

    class_to_pitch = {}
    for drum_name, info in drum_map.items():
        class_id = info["class_id"]
        # choose the first pitch as canonical output pitch
        class_to_pitch[class_id] = info["pitches"][0]

    return class_to_pitch


def npz_to_midi(
    npz_path: str,
    output_path: str,
    drum_map_path: str,
    steps_per_bar: int = 16,
    ticks_per_beat: int = 480,
):
    data = np.load(npz_path)
    X = data["sequence"]
    num_steps, num_classes = X.shape

    class_to_pitch = load_class_to_pitch_map(drum_map_path)
    ticks_per_step = int((ticks_per_beat * 4) / steps_per_bar)

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    prev_tick = 0

    for step_idx in range(num_steps):
        step_vector = X[step_idx]
        active_classes = np.where(step_vector > 0.5)[0]

        if len(active_classes) == 0:
            continue  # just wait; delta_time will accumulate automatically

        abs_tick = step_idx * ticks_per_step
        delta_tick = abs_tick - prev_tick
        prev_tick = abs_tick

        first = True
        for class_id in active_classes:
            pitch = class_to_pitch.get(class_id)
            if pitch is None:
                continue

            # time only on the first note in this step
            time = delta_tick if first else 0
            first = False

            track.append(
                mido.Message(
                    "note_on",
                    note=pitch,
                    velocity=100,
                    time=time,
                    channel=9  # standard drum channel
                )
            )
            track.append(
                mido.Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    time=0,
                    channel=9
                )
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mid.save(output_path)
    print(f"Saved MIDI to {output_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert NPZ drum sequence to MIDI")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_midi", type=str, required=True)
    parser.add_argument("--drum_map", type=str, required=True)
    parser.add_argument("--steps_per_bar", type=int, default=16)
    parser.add_argument("--ticks_per_beat", type=int, default=480)

    args = parser.parse_args()

    npz_to_midi(
        args.input_npz,
        args.output_midi,
        args.drum_map,
        args.steps_per_bar,
        args.ticks_per_beat,
    )
