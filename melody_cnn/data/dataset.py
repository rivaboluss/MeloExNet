"""Segmented POP909 enhanced piano-roll dataset."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from melody_cnn.data.key_utils import key_condition_vector, scale_pitch_classes
from melody_cnn.data.pianoroll import (
    CH_CHORDDIST,
    CH_CHORDROOT,
    CH_CHORDPHASE,
    CH_CHORDTONE,
    CH_CHORDVALID,
    CH_DUR,
    CH_ISBAR,
    CH_MAXV,
    CH_METRIC,
    CH_STATE,
    CH_VEL,
    LABEL_MELODY,
    MappedNote,
    make_coord_channels,
    recompute_harmony_channels,
    slice_segment,
)

CHORD_CHANNELS = (CH_CHORDTONE, CH_CHORDDIST, CH_CHORDROOT, CH_CHORDPHASE, CH_CHORDVALID)


@dataclass
class SegmentSample:
    song_id: int
    t0: int
    path: Path


class Pop909RollDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        song_ids: Sequence[int],
        *,
        steps_per_beat: int = 4,
        beats_per_bar: int = 4,
        segment_bars: int = 4,
        segment_hop_bars: int = 2,
        min_onsets: int = 8,
        binary_label: bool = True,
        augment: bool = False,
        aug_pitch_prob: float = 0.5,
        aug_pitch_range: int = 3,
        aug_mo_prob: float = 0.25,
        use_chord: bool = True,
        use_metric: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.steps_per_beat = steps_per_beat
        self.beats_per_bar = beats_per_bar
        self.steps_per_bar = beats_per_bar * steps_per_beat
        # window covers segment_bars of music + (segment_bars-1) barline columns
        self.seg_len = segment_bars * self.steps_per_bar + max(segment_bars - 1, 0)
        self.hop = segment_hop_bars * self.steps_per_bar + max(segment_hop_bars - 1, 0)
        self.min_onsets = min_onsets
        self.binary_label = binary_label
        self.augment = augment
        self.aug_pitch_prob = float(aug_pitch_prob)
        self.aug_pitch_range = int(aug_pitch_range)
        self.aug_mo_prob = float(aug_mo_prob)
        self.use_chord = bool(use_chord)
        self.use_metric = bool(use_metric)

        index_path = self.cache_dir / "index.json"
        with index_path.open(encoding="utf-8") as f:
            full_index = {int(x["id"]): x for x in json.load(f)}

        self.segments: List[SegmentSample] = []
        self._cache: Dict[int, dict] = {}
        for sid in song_ids:
            meta = full_index.get(int(sid))
            if meta is None:
                continue
            path = self.cache_dir / meta["file"]
            data = np.load(path, allow_pickle=False)
            n_steps = int(data["feats"].shape[-1])
            t = 0
            while t + self.seg_len <= n_steps:
                onsets = data["onsets"]
                n_on = (
                    int(np.sum((onsets[:, 4] >= t) & (onsets[:, 4] < t + self.seg_len)))
                    if onsets.size
                    else 0
                )
                if n_on >= self.min_onsets:
                    self.segments.append(SegmentSample(sid, t, path))
                t += max(self.hop, 1)
            if n_steps >= self.seg_len:
                t_last = n_steps - self.seg_len
                if not self.segments or self.segments[-1].song_id != sid or self.segments[-1].t0 != t_last:
                    onsets = data["onsets"]
                    n_on = (
                        int(np.sum((onsets[:, 4] >= t_last) & (onsets[:, 4] < t_last + self.seg_len)))
                        if onsets.size
                        else 0
                    )
                    if n_on >= self.min_onsets:
                        self.segments.append(SegmentSample(sid, t_last, path))

    def __len__(self) -> int:
        return len(self.segments)

    def _load_song(self, sid: int, path: Path) -> dict:
        if sid not in self._cache:
            data = np.load(path, allow_pickle=False)
            onsets = []
            for row in data["onsets"]:
                onsets.append(
                    MappedNote(
                        note_id=int(row[0]),
                        pitch=int(row[1]),
                        start_beat=float(row[2]),
                        end_beat=float(row[3]),
                        onset_t=int(row[4]),
                        onset_p=int(row[5]),
                        label=int(row[6]),
                        velocity=int(row[7]),
                        track="",
                    )
                )
            self._cache[sid] = {
                "feats": data["feats"],
                "bar_idx": data["bar_idx"],
                "pos_in_bar": data["pos_in_bar"],
                "onsets": onsets,
                "key_tonic": int(data["key_tonic"][0]),
                "key_mode": int(data["key_mode"][0]),
                "key_vec": data["key_vec"].astype(np.float32),
                "pitch_min": int(data["pitch_min"][0]),
            }
            if len(self._cache) > 48:
                oldest = next(iter(self._cache))
                if oldest != sid:
                    self._cache.pop(oldest, None)
        return self._cache[sid]

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        song = self._load_song(seg.song_id, seg.path)
        t0, t1 = seg.t0, seg.t0 + self.seg_len
        feats, bar_idx, pos_in_bar, _, onset_mask, label_map = slice_segment(
            song["feats"],
            song["bar_idx"],
            song["pos_in_bar"],
            song["onsets"],
            t0,
            t1,
            binary_label=self.binary_label,
        )
        tonic = song["key_tonic"]
        mode = song["key_mode"]
        key_vec = song["key_vec"].copy()
        pitch_min = song["pitch_min"]

        span_mask = _melody_span_mask(
            song, t0, t1, feats, self.steps_per_beat, self.steps_per_bar, feats.shape[1]
        )

        if self.augment and random.random() < self.aug_pitch_prob:
            shift = random.randint(-self.aug_pitch_range, self.aug_pitch_range)
            if shift != 0:
                feats = _pitch_shift_feats(feats, shift)
                onset_mask = _pitch_shift_bool(onset_mask, shift)
                label_map = _pitch_shift_label(label_map, shift)
                span_mask = _pitch_shift_bool(span_mask, shift)
                tonic = (tonic + shift) % 12
                key_vec = key_condition_vector(tonic, mode)
                feats = recompute_harmony_channels(
                    feats, pitch_min, scale_pitch_classes(tonic, mode)
                )

        # MOaugm (LStoM): move the whole melody note (onset + sustain) down one octave
        if self.augment and random.random() < self.aug_mo_prob:
            feats, onset_mask, label_map = _mo_augment(
                feats, onset_mask, label_map, span_mask
            )

        coords = make_coord_channels(bar_idx, pos_in_bar, feats.shape[1])
        # normalize state channel 0..3 -> 0..1
        x = feats.copy()
        if not self.use_chord:
            x[list(CHORD_CHANNELS)] = 0.0
        if not self.use_metric:
            x[CH_METRIC] = 0.0
        x[CH_STATE] = x[CH_STATE] / 3.0
        x = np.concatenate([x, coords], axis=0)  # [12+3, P, T]

        return {
            "roll": torch.from_numpy(x.astype(np.float32)),
            "onset_mask": torch.from_numpy(onset_mask),
            "labels": torch.from_numpy(label_map.astype(np.int64)),
            "key_vec": torch.from_numpy(key_vec),
            "song_id": seg.song_id,
            "t0": seg.t0,
        }


def _pitch_shift_feats(feats: np.ndarray, shift: int) -> np.ndarray:
    """Shift all channels along pitch axis (bar columns stay fixed)."""
    if shift == 0:
        return feats.copy()
    out = np.zeros_like(feats)
    is_bar = feats[CH_ISBAR, 0] > 0.5
    out[:, :, is_bar] = feats[:, :, is_bar]
    shifted = np.roll(feats[:, :, ~is_bar], shift, axis=1)
    if shift > 0:
        shifted[:, :shift, :] = 0
    elif shift < 0:
        shifted[:, shift:, :] = 0
    out[:, :, ~is_bar] = shifted
    return out


def _pitch_shift_bool(mask: np.ndarray, shift: int) -> np.ndarray:
    p, t = mask.shape
    out = np.zeros_like(mask)
    if shift > 0:
        out[shift:, :] = mask[: p - shift, :]
    else:
        out[: p + shift, :] = mask[-shift:, :]
    return out


def _pitch_shift_label(lab: np.ndarray, shift: int) -> np.ndarray:
    p, t = lab.shape
    out = np.full_like(lab, -1)
    if shift > 0:
        out[shift:, :] = lab[: p - shift, :]
    else:
        out[: p + shift, :] = lab[-shift:, :]
    return out


def _melody_span_mask(
    song: dict,
    t0: int,
    t1: int,
    feats: np.ndarray,
    steps_per_beat: int,
    steps_per_bar: int,
    p_dim: int,
) -> np.ndarray:
    """Boolean [P, T] over the segment covering every melody onset + its sustain."""
    mask = np.zeros((p_dim, t1 - t0), dtype=bool)
    for n in song["onsets"]:
        if n.label != LABEL_MELODY:
            continue
        d0 = int(np.floor(n.start_beat * steps_per_beat + 1e-6))
        d1 = int(np.ceil(n.end_beat * steps_per_beat - 1e-6)) - 1
        if d1 < d0:
            d1 = d0
        e0 = d0 + d0 // steps_per_bar
        e1 = d1 + d1 // steps_per_bar
        r0 = max(e0, t0)
        r1 = min(e1 + 1, t1)
        if r1 <= r0:
            continue
        mask[n.onset_p, r0 - t0 : r1 - t0] = True
    # never move bar-separator columns
    bar_cols = feats[CH_ISBAR, 0, :] > 0.5
    if bar_cols.any():
        mask[:, bar_cols] = False
    return mask


def _mo_augment(
    feats: np.ndarray,
    onset_mask: np.ndarray,
    label_map: np.ndarray,
    span_mask: np.ndarray,
):
    """Move complete melody notes (onset + sustain) down one octave."""
    _, p_dim, _ = feats.shape
    rows = np.where(span_mask.any(axis=1))[0]
    if rows.size == 0:
        return feats, onset_mask, label_map
    shift = -12
    if (rows + shift < 0).any() or (rows + shift >= p_dim).any():
        return feats, onset_mask, label_map

    feats_out = feats.copy()
    onset_out = onset_mask.copy()
    label_out = label_map.copy()
    for r in rows:
        cols = np.where(span_mask[r])[0]
        nr = r + shift
        feats_out[:, nr, cols] = feats[:, r, cols]
        feats_out[:, r, cols] = 0.0
        on_cols = np.where(onset_mask[r] & (label_map[r] == 1))[0]
        onset_out[nr, on_cols] = True
        onset_out[r, on_cols] = False
        label_out[nr, on_cols] = 1
        label_out[r, on_cols] = -1
    return feats_out, onset_out, label_out


def split_song_ids(
    cache_dir: Path, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    with (Path(cache_dir) / "index.json").open(encoding="utf-8") as f:
        ids = [int(x["id"]) for x in json.load(f)]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return train_ids, val_ids, test_ids
