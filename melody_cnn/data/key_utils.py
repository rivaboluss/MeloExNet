"""Key / scale helpers for piano-roll conditioning."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PITCH_NAME_TO_PC = {
    "B#": 0,
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11], dtype=np.int32)
MINOR_SCALE = np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.int32)


def parse_key_string(key: str) -> Tuple[int, int, str]:
    """
    Return (tonic_pc 0-11, mode 0=major/1=minor, normalized like 'GbM').
    Accepts 'Gb:maj', 'GbM', 'Gbm', 'F# minor', etc.
    """
    if not key:
        return 0, 0, "CM"
    s = key.strip().replace(":", "")
    s = s.replace("major", "M").replace("minor", "m")
    s = s.replace("maj", "M").replace("min", "m")
    s = s.replace("-", "b")
    mode = 1 if s.endswith("m") and not s.endswith("M") else 0
    if s[-1] in ("M", "m"):
        root = s[:-1]
    else:
        root = s
        mode = 0
    root = root.strip()
    # try longest pitch names first
    tonic = 0
    for name in sorted(PITCH_NAME_TO_PC.keys(), key=len, reverse=True):
        if root.startswith(name):
            tonic = PITCH_NAME_TO_PC[name]
            root = name
            break
    norm = f"{root}{'m' if mode else 'M'}"
    return int(tonic), int(mode), norm


def scale_pitch_classes(tonic: int, mode: int) -> set:
    deg = MINOR_SCALE if mode == 1 else MAJOR_SCALE
    return set(int((tonic + d) % 12) for d in deg)


def load_key_for_midi(midi_path: str) -> Tuple[int, int, str]:
    """Prefer POP909 key_audio.txt; fallback C major."""
    folder = Path(midi_path).parent
    key_file = folder / "key_audio.txt"
    if key_file.exists():
        try:
            info = np.loadtxt(str(key_file), dtype=str)
            flat = np.atleast_1d(info).flatten()
            # single row: start end key  OR multiple rows -> take first key token ending with maj/min
            if flat.size >= 3 and ("maj" in flat[2].lower() or "min" in flat[2].lower() or ":" in flat[2]):
                return parse_key_string(str(flat[2]))
            # multi-line matrix
            if flat.size > 3:
                # shape roughly Nx3
                arr = np.loadtxt(str(key_file), dtype=str)
                if arr.ndim == 1 and arr.size >= 3:
                    return parse_key_string(str(arr[2]))
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return parse_key_string(str(arr[0, 2]))
        except Exception:
            pass
    # music21 fallback (optional)
    try:
        import music21

        score = music21.converter.parse(midi_path)
        k = music21.analysis.discrete.KrumhanslSchmuckler().getSolution(score)
        name = k.name.replace(" major", "M").replace(" minor", "m").replace("-", "b")
        return parse_key_string(name)
    except Exception:
        return 0, 0, "CM"


def key_condition_vector(tonic: int, mode: int) -> np.ndarray:
    """14-D: tonic one-hot(12) + mode one-hot(2)."""
    v = np.zeros(14, dtype=np.float32)
    v[int(tonic) % 12] = 1.0
    v[12 + (1 if mode else 0)] = 1.0
    return v
