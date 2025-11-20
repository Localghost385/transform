# test_preprocess.py

import os
import tempfile
import numpy as np
import pytest
import mido
import yaml

from src.preprocess import (
    load_drum_map,
    load_midi_file,
    extract_drum_notes,
    quantize_notes,
    notes_to_multi_hot,
    save_npz,
    process_single_file,
    process_all_files
)

# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def drum_map_file():
    drum_map = {
        "kick": {"class_id": 0, "pitches": [36]},
        "snare": {"class_id": 1, "pitches": [38]}
    }
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        yaml.dump(drum_map, f)
        path = f.name
    yield path
    os.remove(path)


@pytest.fixture
def simple_midi_file():
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # add note_on messages for channel 9 (drums)
    track.append(mido.Message('note_on', channel=9, note=36, velocity=100, time=0))
    track.append(mido.Message('note_on', channel=9, note=38, velocity=100, time=480))
    
    fd, path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    mid.save(path)
    yield path
    os.remove(path)


# ---------------------------
# Test load_drum_map
# ---------------------------

def test_load_drum_map(drum_map_file):
    pitch_to_class = load_drum_map(drum_map_file)
    assert pitch_to_class[36] == 0
    assert pitch_to_class[38] == 1
    assert len(pitch_to_class) == 2


# ---------------------------
# Test load_midi_file
# ---------------------------

def test_load_midi_file(simple_midi_file):
    mid = load_midi_file(simple_midi_file)
    assert isinstance(mid, mido.MidiFile)
    assert len(mid.tracks) == 1


# ---------------------------
# Test extract_drum_notes
# ---------------------------

def test_extract_drum_notes(simple_midi_file):
    mid = mido.MidiFile(simple_midi_file)
    notes = extract_drum_notes(mid)
    assert len(notes) == 2
    assert notes[0][1] == 36
    assert notes[1][1] == 38


# ---------------------------
# Test quantize_notes
# ---------------------------

def test_quantize_notes():
    drum_notes = [(0, 36, 100), (480, 38, 100)]
    ticks_per_bar = 480 * 4  # assume 4 beats/bar
    steps_per_bar = 16
    quantized = quantize_notes(drum_notes, ticks_per_bar, steps_per_bar)
    # step indices should be integer
    for step, pitch, vel in quantized:
        assert isinstance(step, int)
    assert quantized[0][0] == 0


# ---------------------------
# Test notes_to_multi_hot
# ---------------------------

def test_notes_to_multi_hot():
    quantized_notes = [(0, 36, 100), (1, 38, 100)]
    pitch_to_class = {36: 0, 38: 1}
    X = notes_to_multi_hot(quantized_notes, num_classes=2, pitch_to_class=pitch_to_class)
    assert X.shape == (2, 2)
    assert X[0, 0] == 1
    assert X[1, 1] == 1


# ---------------------------
# Test save_npz
# ---------------------------

def test_save_npz():
    X = np.array([[1,0],[0,1]], dtype=np.uint8)
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    save_npz(path, X)
    data = np.load(path)['X']
    assert np.array_equal(data, X)
    os.remove(path)


# ---------------------------
# Test process_single_file
# ---------------------------

def test_process_single_file(simple_midi_file, drum_map_file):
    output_dir = tempfile.mkdtemp()
    pitch_to_class = load_drum_map(drum_map_file)
    process_single_file(simple_midi_file, output_dir, pitch_to_class)
    
    files = os.listdir(output_dir)
    assert len(files) == 1
    npz_path = os.path.join(output_dir, files[0])
    data = np.load(npz_path)['X']
    assert data.shape[1] == len(set(pitch_to_class.values()))
    
    # cleanup
    for f in files:
        os.remove(os.path.join(output_dir, f))
    os.rmdir(output_dir)


# ---------------------------
# Test process_all_files
# ---------------------------

def test_process_all_files(simple_midi_file, drum_map_file):
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    # copy MIDI file into input_dir
    midi_copy_path = os.path.join(input_dir, "test.mid")
    import shutil
    shutil.copy(simple_midi_file, midi_copy_path)
    
    process_all_files(input_dir, output_dir, drum_map_file)
    
    files = os.listdir(output_dir)
    assert len(files) == 1
    npz_path = os.path.join(output_dir, files[0])
    data = np.load(npz_path)['X']
    assert data.shape[1] == 2  # kick+snare
    
    # cleanup
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)
