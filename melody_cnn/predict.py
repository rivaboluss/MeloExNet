"""Infer melody with enhanced roll + key conditioning; restore via onset map."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from melody_cnn.data.pianoroll import (
    build_roll_from_midi,
    make_coord_channels,
    predictions_to_pretty_midi,
    slice_segment,
)
from melody_cnn.models import AnisoCoordMelodyNet


@torch.no_grad()
def predict_midi(midi_path: Path, ckpt_path: Path, out_dir: Path, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = AnisoCoordMelodyNet(
        in_channels=cfg.get("in_channels", 9),
        base_channels=cfg["base_channels"],
        num_classes=cfg["num_classes"],
        k_pitch=cfg["kernel_pitch"],
        k_time=cfg["kernel_time"],
        num_blocks=cfg["num_blocks"],
        dropout=cfg["dropout"],
        cond_dim=cfg.get("cond_dim", 14),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    packed = build_roll_from_midi(
        str(midi_path),
        steps_per_beat=cfg["steps_per_beat"],
        beats_per_bar=cfg.get("beats_per_bar", 4),
        pitch_min=cfg["pitch_min"],
        pitch_max=cfg["pitch_max"],
    )
    feats = packed["feats"]
    bar_idx = packed["bar_idx"]
    pos_in_bar = packed["pos_in_bar"]
    onsets = packed["onsets"]
    bpm = packed["bpm"]
    key_vec = torch.from_numpy(packed["key_vec"]).unsqueeze(0).to(device)

    steps_per_bar = cfg.get("beats_per_bar", 4) * cfg["steps_per_beat"]
    segment_bars = cfg.get("segment_bars", 4)
    hop_bars = cfg.get("segment_hop_bars", 2)
    seg_len = segment_bars * steps_per_bar + max(segment_bars - 1, 0)
    hop = hop_bars * steps_per_bar + max(hop_bars - 1, 0)
    n_steps = feats.shape[-1]

    logit_sum = defaultdict(lambda: np.zeros(cfg["num_classes"], dtype=np.float64))
    logit_cnt = defaultdict(int)

    starts = list(range(0, max(1, n_steps - seg_len + 1), max(hop, 1)))
    if n_steps >= seg_len and (not starts or starts[-1] != n_steps - seg_len):
        starts.append(max(0, n_steps - seg_len))

    # pad if short
    if n_steps < seg_len:
        pad_t = seg_len - n_steps
        feats = np.pad(feats, ((0, 0), (0, 0), (0, pad_t)))
        bar_idx = np.pad(bar_idx, (0, pad_t), constant_values=int(bar_idx[-1]) if len(bar_idx) else 0)
        pos_in_bar = np.pad(pos_in_bar, (0, pad_t), constant_values=0.0)
        starts = [0]
        n_steps = seg_len

    for t0 in starts:
        t1 = t0 + seg_len
        seg, seg_bar, seg_pos, seg_onsets, _, _ = slice_segment(
            feats, bar_idx, pos_in_bar, onsets, t0, t1, binary_label=True
        )
        if seg.shape[-1] < seg_len:
            pad = seg_len - seg.shape[-1]
            seg = np.pad(seg, ((0, 0), (0, 0), (0, pad)))
            seg_bar = np.pad(seg_bar, (0, pad), constant_values=int(seg_bar[-1]) if len(seg_bar) else 0)
            seg_pos = np.pad(seg_pos, (0, pad))
        coords = make_coord_channels(seg_bar, seg_pos, seg.shape[1])
        x = seg.copy()
        x[0] = x[0] / 3.0
        x = np.concatenate([x, coords], axis=0)
        xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        logits = model(xt, key_vec)[0].float().cpu().numpy()
        for o in seg_onsets:
            key = o.note_id
            logit_sum[key] += logits[:, o.onset_p, o.onset_t]
            logit_cnt[key] += 1

    pred_labels = []
    for o in onsets:
        if o.note_id in logit_sum:
            lab = int(np.argmax(logit_sum[o.note_id] / max(logit_cnt[o.note_id], 1)))
        else:
            lab = 0
        pred_labels.append(lab)

    out_dir.mkdir(parents=True, exist_ok=True)
    pm = predictions_to_pretty_midi(onsets, pred_labels, bpm)
    out_mid = out_dir / f"{midi_path.stem}_pred_MELODY+BRIDGE.mid"
    pm.write(str(out_mid))
    n_mel = sum(pred_labels)
    print(
        f"Wrote {out_mid} onsets={len(onsets)} pred_melody={n_mel} "
        f"key={packed['key_str']}"
    )
    return out_mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--midi", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    predict_midi(args.midi, args.ckpt, args.out, args.device)


if __name__ == "__main__":
    main()
