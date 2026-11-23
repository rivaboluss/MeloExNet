"""Precompute enhanced rolls for POP909 001-909."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from melody_cnn.data.pianoroll import build_roll_from_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps-per-beat", type=int, default=4)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--pitch-min", type=int, default=21)
    ap.add_argument("--pitch-max", type=int, default=108)
    ap.add_argument("--max-id", type=int, default=909)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for sid in tqdm(range(1, args.max_id + 1), desc="prepare"):
        folder = f"{sid:03d}"
        mid = args.pop909 / folder / f"{folder}.mid"
        if not mid.exists():
            continue
        packed = build_roll_from_midi(
            str(mid),
            steps_per_beat=args.steps_per_beat,
            beats_per_bar=args.beats_per_bar,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
        )
        onsets = packed["onsets"]
        onset_arr = np.array(
            [
                [
                    o.note_id,
                    o.pitch,
                    o.start_beat,
                    o.end_beat,
                    o.onset_t,
                    o.onset_p,
                    o.label,
                    o.velocity,
                ]
                for o in onsets
            ],
            dtype=np.float64,
        ) if onsets else np.zeros((0, 8), dtype=np.float64)

        out_path = args.out / f"{folder}.npz"
        np.savez_compressed(
            out_path,
            feats=packed["feats"],
            bar_idx=packed["bar_idx"],
            pos_in_bar=packed["pos_in_bar"],
            onsets=onset_arr,
            bpm=np.array([packed["bpm"]], dtype=np.float32),
            key_tonic=np.array([packed["key_tonic"]], dtype=np.int32),
            key_mode=np.array([packed["key_mode"]], dtype=np.int32),
            key_vec=packed["key_vec"],
            chord_root=packed["chord_root"],
            chord_type=packed["chord_type"],
            chord_phase=packed["chord_phase"],
            steps_per_beat=np.array([args.steps_per_beat], dtype=np.int32),
            beats_per_bar=np.array([args.beats_per_bar], dtype=np.int32),
            pitch_min=np.array([args.pitch_min], dtype=np.int32),
            pitch_max=np.array([args.pitch_max], dtype=np.int32),
        )
        index.append(
            {
                "id": sid,
                "file": out_path.name,
                "n_steps": int(packed["feats"].shape[-1]),
                "n_onsets": int(len(onsets)),
                "n_melody": int(sum(1 for o in onsets if o.label == 1)),
                "key": packed["key_str"],
            }
        )

    with (args.out / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    meta = {
        "version": 4,
        "channels": [
            "state", "in_scale", "is_bar", "velocity", "duration", "max_velocity",
            "chord_tone", "chord_dist", "chord_root", "chord_phase", "chord_valid",
            "metric_strength",
        ],
        "state_values": {"empty": 0, "onset": 1, "sustain": 2, "bar": 3},
        "steps_per_beat": args.steps_per_beat,
        "beats_per_bar": args.beats_per_bar,
    }
    with (args.out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(index)} songs -> {args.out}")


if __name__ == "__main__":
    main()
