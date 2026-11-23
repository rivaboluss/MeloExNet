"""Enhanced quantized piano-roll: bar columns, in-scale, onset→note map."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi

from melody_cnn.data.chord_utils import CHORD_TYPES, build_chord_arrays, load_chord_segments
from melody_cnn.data.key_utils import key_condition_vector, load_key_for_midi, scale_pitch_classes

EMPTY, ONSET, SUSTAIN, BAR = 0, 1, 2, 3

LABEL_ACCOMP, LABEL_MELODY, LABEL_BRIDGE = 0, 1, 2
TRACK_TO_LABEL = {
    "MELODY": LABEL_MELODY,
    "BRIDGE": LABEL_BRIDGE,
    "PIANO": LABEL_ACCOMP,
    "ACCOMPANIMENT": LABEL_ACCOMP,
}

# Feature channels in feats[C,P,T]
CH_STATE, CH_INSCALE, CH_ISBAR = 0, 1, 2
CH_VEL, CH_DUR, CH_MAXV = 3, 4, 5
# harmonic / metric features (all relative to pitch, so pitch-shift invariant)
CH_CHORDTONE, CH_CHORDDIST, CH_CHORDROOT = 6, 7, 8
CH_CHORDPHASE, CH_CHORDVALID, CH_METRIC = 9, 10, 11
N_FEAT_CH = 12
VEL_NORM = 127.0
DUR_BEAT_CAP = 4.0  # beats; everything ≥4 beats → 1.0


def _metric_strength(local_step: int, steps_per_beat: int) -> float:
    """Metrical hierarchy from quantized position: downbeat > beat > 8th > 16th."""
    if local_step == 0:
        return 1.0
    if local_step % steps_per_beat == 0:
        return 0.7
    if local_step % max(steps_per_beat // 2, 1) == 0:
        return 0.4
    return 0.2


@dataclass
class MappedNote:
    note_id: int
    pitch: int
    start_beat: float
    end_beat: float
    onset_t: int  # index on expanded time axis (with bar columns)
    onset_p: int
    label: int  # 3-class track label; binary = int(label == LABEL_MELODY)
    velocity: int
    track: str


def note_label_from_track(name: str) -> int:
    key = (name or "").strip().upper()
    return TRACK_TO_LABEL.get(key, LABEL_ACCOMP)


def _tempo_bpm(pm: pretty_midi.PrettyMIDI, default: float = 120.0) -> float:
    times, tempos = pm.get_tempo_changes()
    if tempos is None or len(tempos) == 0:
        return default
    return float(tempos[0])


def seconds_to_beats(t_sec: float, bpm: float) -> float:
    return t_sec * bpm / 60.0


def beats_to_seconds(t_beat: float, bpm: float) -> float:
    return t_beat * 60.0 / bpm


def _collect_raw_notes(pm, pitch_min, pitch_max, prefer_tracks, bpm, steps_per_beat):
    instruments = list(pm.instruments)
    if prefer_tracks:
        named = {(ins.name or "").upper(): ins for ins in instruments}
        chosen = []
        for name in prefer_tracks:
            if name in named and named[name].notes:
                chosen.append(named[name])
        if chosen:
            instruments = chosen
    instruments = [ins for ins in instruments if not ins.is_drum]

    raw_notes = []
    for ins in instruments:
        track = (ins.name or "PIANO").upper()
        label = note_label_from_track(track)
        for n in ins.notes:
            if n.pitch < pitch_min or n.pitch > pitch_max:
                continue
            sb = seconds_to_beats(n.start, bpm)
            eb = seconds_to_beats(n.end, bpm)
            if eb <= sb:
                eb = sb + 1.0 / steps_per_beat
            raw_notes.append(
                {
                    "pitch": int(n.pitch),
                    "start_beat": float(sb),
                    "end_beat": float(eb),
                    "velocity": int(n.velocity),
                    "track": track,
                    "label": int(label),
                }
            )
    return raw_notes


def _paint_dense_roll(raw_notes, n_pitch, n_steps, pitch_min, steps_per_beat):
    roll = np.zeros((n_pitch, n_steps), dtype=np.uint8)
    vel_roll = np.zeros((n_pitch, n_steps), dtype=np.float32)  # velocity /127
    dur_roll = np.zeros((n_pitch, n_steps), dtype=np.float32)  # duration in beats, capped
    onsets_dense = []
    onset_owner = {}
    raw_notes = sorted(raw_notes, key=lambda x: (x["start_beat"], -x["pitch"]))
    for note_id, n in enumerate(raw_notes):
        p = n["pitch"] - pitch_min
        t0 = int(np.floor(n["start_beat"] * steps_per_beat + 1e-6))
        t1 = int(np.ceil(n["end_beat"] * steps_per_beat - 1e-6))
        t0 = min(max(t0, 0), n_steps - 1)
        t1 = min(max(t1, t0 + 1), n_steps)
        v_norm = float(n["velocity"]) / VEL_NORM
        d_beats = n["end_beat"] - n["start_beat"]
        d_norm = min(d_beats / DUR_BEAT_CAP, 1.0)
        for t in range(t0, t1):
            if roll[p, t] == EMPTY:
                roll[p, t] = SUSTAIN
                vel_roll[p, t] = v_norm
                dur_roll[p, t] = d_norm
        roll[p, t0] = ONSET
        vel_roll[p, t0] = v_norm
        dur_roll[p, t0] = d_norm
        mapped = {
            "note_id": note_id,
            "pitch": n["pitch"],
            "start_beat": n["start_beat"],
            "end_beat": n["end_beat"],
            "onset_t_dense": t0,
            "onset_p": p,
            "label": n["label"],
            "velocity": n["velocity"],
            "track": n["track"],
            "vel_norm": v_norm,
            "dur_norm": d_norm,
        }
        key = (p, t0)
        if key in onset_owner:
            prev = onsets_dense[onset_owner[key]]
            prev_score = (prev["label"] == LABEL_MELODY, prev["end_beat"] - prev["start_beat"])
            cur_score = (mapped["label"] == LABEL_MELODY, mapped["end_beat"] - mapped["start_beat"])
            if cur_score >= prev_score:
                onsets_dense[onset_owner[key]] = mapped
        else:
            onset_owner[key] = len(onsets_dense)
            onsets_dense.append(mapped)
    return roll, vel_roll, dur_roll, onsets_dense


def recompute_harmony_channels(feats: np.ndarray, pitch_min: int, scale_pcs: set) -> np.ndarray:
    """Recompute in_scale after pitch-shift aug.

    Chord channels are relative to the sounding pitch, so transposition leaves
    them unchanged; velocity / duration / metric do not depend on pitch at all.
    """
    out = feats.copy()
    _, n_pitch, n_t = out.shape
    out[CH_INSCALE] = 0.0
    for t in range(n_t):
        if out[CH_ISBAR, 0, t] > 0.5:
            continue
        state_col = out[CH_STATE, :, t]
        for p in range(n_pitch):
            st = int(state_col[p])
            if st in (ONSET, SUSTAIN):
                pc = (p + pitch_min) % 12
                out[CH_INSCALE, p, t] = 1.0 if pc in scale_pcs else 0.0
    return out


def _expand_with_bar_columns(
    dense_roll: np.ndarray,
    vel_roll: np.ndarray,
    dur_roll: np.ndarray,
    onsets_dense: list,
    steps_per_bar: int,
    steps_per_beat: int,
    scale_pcs: set,
    pitch_min: int,
    chord_root: np.ndarray,
    chord_type: np.ndarray,
    chord_phase: np.ndarray,
):
    n_pitch, n_dense = dense_roll.shape
    if n_dense < 1:
        n_dense = 1
        dense_roll = np.zeros((n_pitch, 1), dtype=np.uint8)
        vel_roll = np.zeros((n_pitch, 1), dtype=np.float32)
        dur_roll = np.zeros((n_pitch, 1), dtype=np.float32)
        chord_root = np.full(1, -1, dtype=np.int8)
        chord_type = np.full(1, -1, dtype=np.int8)
        chord_phase = np.zeros(1, dtype=np.float32)

    n_bars = int(np.ceil(n_dense / steps_per_bar))
    n_bars = max(n_bars, 1)
    t_exp = n_dense + max(n_bars - 1, 0)
    feats = np.zeros((N_FEAT_CH, n_pitch, t_exp), dtype=np.float32)
    bar_idx = np.zeros(t_exp, dtype=np.int32)
    pos_in_bar = np.zeros(t_exp, dtype=np.float32)

    dense_to_exp = np.full(n_dense, -1, dtype=np.int32)
    t = 0
    for b in range(n_bars):
        s0 = b * steps_per_bar
        s1 = min((b + 1) * steps_per_bar, n_dense)
        for d in range(s0, s1):
            local = d - s0
            st_col = dense_roll[:, d]
            feats[CH_STATE, :, t] = st_col
            feats[CH_METRIC, :, t] = _metric_strength(local, steps_per_beat)

            root_pc = int(chord_root[d])
            type_idx = int(chord_type[d])
            tones = CHORD_TYPES[type_idx][1] if type_idx >= 0 else ()
            phase_v = float(chord_phase[d])

            for p in range(n_pitch):
                st = int(st_col[p])
                if st not in (ONSET, SUSTAIN):
                    continue
                pc = (p + pitch_min) % 12
                feats[CH_INSCALE, p, t] = 1.0 if pc in scale_pcs else 0.0
                feats[CH_VEL, p, t] = vel_roll[p, d]
                feats[CH_DUR, p, t] = dur_roll[p, d]
                if tones:
                    rel = (pc - root_pc) % 12
                    feats[CH_CHORDROOT, p, t] = rel / 12.0
                    feats[CH_CHORDPHASE, p, t] = phase_v
                    feats[CH_CHORDVALID, p, t] = 1.0
                    dist = min(
                        min((pc - (root_pc + iv)) % 12, ((root_pc + iv) - pc) % 12)
                        for iv in tones
                    )
                    if dist == 0:
                        feats[CH_CHORDTONE, p, t] = 1.0
                    feats[CH_CHORDDIST, p, t] = min(dist / 6.0, 1.0)

            # max velocity among simultaneous active notes (per-column global)
            active_mask = (st_col == ONSET) | (st_col == SUSTAIN)
            if active_mask.any():
                feats[CH_MAXV, active_mask, t] = vel_roll[active_mask, d].max()
            bar_idx[t] = b
            pos_in_bar[t] = local / float(steps_per_bar)
            dense_to_exp[d] = t
            t += 1
        if b < n_bars - 1:
            feats[CH_STATE, :, t] = float(BAR)
            feats[CH_ISBAR, :, t] = 1.0
            bar_idx[t] = b
            pos_in_bar[t] = 1.0
            t += 1

    assert t == t_exp, (t, t_exp)

    onsets: List[MappedNote] = []
    for o in onsets_dense:
        et = int(dense_to_exp[o["onset_t_dense"]])
        if et < 0:
            continue
        onsets.append(
            MappedNote(
                note_id=o["note_id"],
                pitch=o["pitch"],
                start_beat=o["start_beat"],
                end_beat=o["end_beat"],
                onset_t=et,
                onset_p=o["onset_p"],
                label=o["label"],
                velocity=o["velocity"],
                track=o["track"],
            )
        )
    return feats, bar_idx, pos_in_bar, onsets, dense_to_exp


def build_roll_from_midi(
    midi_path: str,
    *,
    steps_per_beat: int = 4,
    beats_per_bar: int = 4,
    pitch_min: int = 21,
    pitch_max: int = 108,
    prefer_tracks: Optional[Tuple[str, ...]] = ("MELODY", "BRIDGE", "PIANO"),
    key_tonic: Optional[int] = None,
    key_mode: Optional[int] = None,
) -> Dict:
    """
    feats: [12,P,T] float
      ch0 state: 0 empty, 1 onset, 2 sustain, 3 bar-separator
      ch1 in_scale: binary on note cells
      ch2 is_bar: 1 on separator columns
      ch3 velocity / 127
      ch4 duration / 4 beats (capped)
      ch5 max velocity in the same column
      ch6 chord tone (1 if pitch class belongs to the current chord)
      ch7 distance to nearest chord tone / 6 (0 if chord tone)
      ch8 (pitch - root) % 12 / 12
      ch9 phase inside current chord segment (0..1)
      ch10 chord annotation valid (0 on 'N' / missing chord file)
      ch11 metrical strength (downbeat 1.0 → 16th 0.2)
    bar_idx / pos_in_bar: length T (pos in [0,1], 1.0 on barline)
    key_vec: [14]
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    bpm = _tempo_bpm(pm)
    n_pitch = pitch_max - pitch_min + 1
    steps_per_bar = beats_per_bar * steps_per_beat

    if key_tonic is None or key_mode is None:
        tonic, mode, key_str = load_key_for_midi(midi_path)
    else:
        tonic, mode = int(key_tonic), int(key_mode)
        key_str = f"pc{tonic}_{'m' if mode else 'M'}"
    scale_pcs = scale_pitch_classes(tonic, mode)
    key_vec = key_condition_vector(tonic, mode)

    raw_notes = _collect_raw_notes(pm, pitch_min, pitch_max, prefer_tracks, bpm, steps_per_beat)
    chord_segments = load_chord_segments(midi_path)
    if not raw_notes:
        feats = np.zeros((N_FEAT_CH, n_pitch, 1), dtype=np.float32)
        return {
            "feats": feats,
            "bar_idx": np.zeros(1, dtype=np.int32),
            "pos_in_bar": np.zeros(1, dtype=np.float32),
            "onsets": [],
            "bpm": bpm,
            "steps_per_beat": steps_per_beat,
            "beats_per_bar": beats_per_bar,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "key_tonic": tonic,
            "key_mode": mode,
            "key_str": key_str,
            "key_vec": key_vec,
            "chord_root": np.full(1, -1, dtype=np.int8),
            "chord_type": np.full(1, -1, dtype=np.int8),
            "chord_phase": np.zeros(1, dtype=np.float32),
        }

    max_beat = max(n["end_beat"] for n in raw_notes)
    n_steps = max(1, int(np.ceil(max_beat * steps_per_beat)) + 1)
    dense, vel_roll, dur_roll, onsets_dense = _paint_dense_roll(raw_notes, n_pitch, n_steps, pitch_min, steps_per_beat)
    chord_root, chord_type, chord_phase = build_chord_arrays(
        chord_segments, bpm, n_steps, steps_per_beat
    )
    feats, bar_idx, pos_in_bar, onsets, _ = _expand_with_bar_columns(
        dense,
        vel_roll,
        dur_roll,
        onsets_dense,
        steps_per_bar,
        steps_per_beat,
        scale_pcs,
        pitch_min,
        chord_root,
        chord_type,
        chord_phase,
    )

    return {
        "feats": feats,
        "bar_idx": bar_idx,
        "pos_in_bar": pos_in_bar,
        "onsets": onsets,
        "bpm": bpm,
        "steps_per_beat": steps_per_beat,
        "beats_per_bar": beats_per_bar,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "key_tonic": tonic,
        "key_mode": mode,
        "key_str": key_str,
        "key_vec": key_vec,
        "chord_root": chord_root,
        "chord_type": chord_type,
        "chord_phase": chord_phase,
    }


def slice_segment(
    feats: np.ndarray,
    bar_idx: np.ndarray,
    pos_in_bar: np.ndarray,
    onsets: List[MappedNote],
    t0: int,
    t1: int,
    binary_label: bool = True,
):
    """Cut time [t0:t1]. label_map uses binary melody if binary_label else 3-class."""
    seg = feats[:, :, t0:t1].copy()
    seg_bar = bar_idx[t0:t1].copy()
    seg_pos = pos_in_bar[t0:t1].copy()
    _, p_dim, length = seg.shape
    onset_mask = np.zeros((p_dim, length), dtype=np.bool_)
    label_map = np.full((p_dim, length), -1, dtype=np.int16)
    seg_onsets: List[MappedNote] = []
    for n in onsets:
        if t0 <= n.onset_t < t1:
            rel_t = n.onset_t - t0
            # skip if this column is a bar separator
            if seg[CH_ISBAR, 0, rel_t] > 0.5:
                continue
            onset_mask[n.onset_p, rel_t] = True
            if binary_label:
                lab = 1 if n.label == LABEL_MELODY else 0
            else:
                lab = n.label
            label_map[n.onset_p, rel_t] = lab
            seg_onsets.append(
                MappedNote(
                    note_id=n.note_id,
                    pitch=n.pitch,
                    start_beat=n.start_beat,
                    end_beat=n.end_beat,
                    onset_t=rel_t,
                    onset_p=n.onset_p,
                    label=n.label,
                    velocity=n.velocity,
                    track=n.track,
                )
            )
    return seg, seg_bar, seg_pos, seg_onsets, onset_mask, label_map


def make_coord_channels(
    bar_idx: np.ndarray,
    pos_in_bar: np.ndarray,
    n_pitch: int,
    max_bar_ref: Optional[int] = None,
) -> np.ndarray:
    """
    Return [3, P, T]: pitch_y, bar_norm, pos_in_bar
    (broadcast pitch over T, bar/pos over P)
    """
    t = len(bar_idx)
    max_b = max(int(bar_idx.max()) if len(bar_idx) else 0, 1)
    if max_bar_ref is not None:
        max_b = max(max_b, int(max_bar_ref), 1)
    pitch_y = np.linspace(-1.0, 1.0, n_pitch, dtype=np.float32).reshape(n_pitch, 1)
    pitch_y = np.repeat(pitch_y, t, axis=1)
    bar_norm = (bar_idx.astype(np.float32) / float(max_b)) * 2.0 - 1.0  # [-1,1]
    bar_map = np.repeat(bar_norm.reshape(1, t), n_pitch, axis=0)
    pos_map = np.repeat(pos_in_bar.astype(np.float32).reshape(1, t), n_pitch, axis=0)
    return np.stack([pitch_y, bar_map, pos_map], axis=0)


def predictions_to_pretty_midi(
    onsets: List[MappedNote],
    pred_labels: List[int],
    bpm: float,
    *,
    melody_name: str = "MELODY",
    other_name: str = "BRIDGE",
) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    mel = pretty_midi.Instrument(program=0, name=melody_name)
    other = pretty_midi.Instrument(program=0, name=other_name)
    for n, lab in zip(onsets, pred_labels):
        note = pretty_midi.Note(
            velocity=max(1, n.velocity),
            pitch=n.pitch,
            start=beats_to_seconds(n.start_beat, bpm),
            end=beats_to_seconds(n.end_beat, bpm),
        )
        if int(lab) == 1:  # binary melody
            mel.notes.append(note)
        else:
            other.notes.append(note)
    pm.instruments.extend([mel, other])
    return pm
