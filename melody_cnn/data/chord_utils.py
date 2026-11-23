"""POP909 chord annotation parsing.

chord_midi.txt rows are ``start_sec end_sec chord`` with labels such as
``B:maj``, ``F#:maj7/5``, ``Bb:sus4`` or ``N`` (no chord).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from melody_cnn.data.key_utils import PITCH_NAME_TO_PC

# (quality name, semitone intervals from root). Ordered longest-first at match time.
CHORD_TYPES: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("minmaj7", (0, 3, 7, 11)),
    ("hdim7", (0, 3, 6, 10)),
    ("dim7", (0, 3, 6, 9)),
    ("maj7", (0, 4, 7, 11)),
    ("min7", (0, 3, 7, 10)),
    ("dom7", (0, 4, 7, 10)),
    ("maj6", (0, 4, 7, 9)),
    ("min6", (0, 3, 7, 9)),
    ("sus4", (0, 5, 7)),
    ("sus2", (0, 2, 7)),
    ("aug", (0, 4, 8)),
    ("dim", (0, 3, 6)),
    ("maj", (0, 4, 7)),
    ("min", (0, 3, 7)),
)
CHORD_TYPE_INDEX = {name: i for i, (name, _) in enumerate(CHORD_TYPES)}


def parse_chord_label(label: str) -> Tuple[int, int]:
    """Return (root_pc, chord_type_index); (-1, -1) for N / unparsable."""
    s = (label or "").strip()
    if not s or s.upper() == "N":
        return -1, -1
    main = s.split("/")[0]
    if ":" in main:
        root_s, quality = main.split(":", 1)
    else:
        root_s, quality = main, "maj"
    root_s = root_s.strip()
    quality = quality.strip().lower()

    tonic = -1
    for name in sorted(PITCH_NAME_TO_PC, key=len, reverse=True):
        if root_s.startswith(name):
            tonic = PITCH_NAME_TO_PC[name]
            break
    if tonic < 0:
        return -1, -1

    type_idx = -1
    for name in sorted(CHORD_TYPE_INDEX, key=len, reverse=True):
        if quality.startswith(name):
            type_idx = CHORD_TYPE_INDEX[name]
            break
    if type_idx < 0:
        type_idx = CHORD_TYPE_INDEX["maj"]
    return int(tonic), int(type_idx)


def load_chord_segments(midi_path: str) -> List[Tuple[float, float, str]]:
    """Read chord_midi.txt next to the MIDI; timestamps stay in seconds."""
    path = Path(midi_path).parent / "chord_midi.txt"
    if not path.exists():
        return []
    try:
        # comments=None: otherwise numpy strips "C#:maj" at the '#' character
        arr = np.loadtxt(str(path), dtype=str, ndmin=2, comments=None)
    except Exception:
        return []
    out: List[Tuple[float, float, str]] = []
    for row in arr:
        if row.size < 3:
            continue
        try:
            out.append((float(row[0]), float(row[1]), str(row[2])))
        except ValueError:
            continue
    return out


def build_chord_arrays(
    segments: Sequence[Tuple[float, float, str]],
    bpm: float,
    n_steps: int,
    steps_per_beat: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-dense-step chord arrays: root pc, chord type index, phase in segment (0..1)."""
    root = np.full(n_steps, -1, dtype=np.int8)
    ctype = np.full(n_steps, -1, dtype=np.int8)
    phase = np.zeros(n_steps, dtype=np.float32)
    if not segments:
        return root, ctype, phase
    scale = bpm / 60.0 * steps_per_beat
    for start_sec, end_sec, label in segments:
        root_pc, type_idx = parse_chord_label(label)
        if root_pc < 0:
            continue
        i0 = int(np.floor(start_sec * scale + 1e-6))
        i1 = int(np.ceil(end_sec * scale - 1e-6))
        i0 = max(i0, 0)
        i1 = min(i1, n_steps)
        if i1 <= i0:
            continue
        span = i1 - i0
        root[i0:i1] = root_pc
        ctype[i0:i1] = type_idx
        phase[i0:i1] = np.arange(span, dtype=np.float32) / float(span)
    return root, ctype, phase
