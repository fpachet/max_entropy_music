"""
Copyright (c) 2025 Ynosound.
All rights reserved.

Unauthorized copying, modification, or distribution of this software, in whole or in part,
is strictly prohibited without prior written consent from MyCompany.

See LICENSE file in the project root for full license information.
"""

from pathlib import Path
from typing import Iterable

import mido


def save_midi(
    midi_pitch_sequence: Iterable[int], output_file: Path | str = "generated.mid"
):
    """
    Save a sequence of MIDI pitches to a MIDI file.
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for midi_pitch in midi_pitch_sequence:
        track.append(mido.Message("note_on", note=midi_pitch, velocity=64, time=0))
        track.append(mido.Message("note_off", note=midi_pitch, velocity=64, time=120))
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    while path.exists():
        path = path.with_name(f"{path.stem}-1{path.suffix}")
    mid.save(path)


def extract_pitches_from_midi(midi_file: Path | str) -> tuple[int, ...]:
    """
    Extracts MIDI note sequence from a MIDI file.
    """
    mid = mido.MidiFile(Path(midi_file))
    track = mid.tracks[min(1, len(mid.tracks) - 1)]
    return tuple(
        msg.note for msg in track if msg.type == "note_on" and msg.velocity > 0
    )
