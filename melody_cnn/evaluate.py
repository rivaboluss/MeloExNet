"""Note-level evaluation matching the inference pipeline.

Overlapping segments are stitched per note (average melody probability over all
segments that cover it), then a single global threshold is applied. Reports
metrics at 0.5, at the threshold tuned on validation, PR average precision, and
per-song F1 spread.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from melody_cnn.data.dataset import Pop909RollDataset
from melody_cnn.models import AnisoCoordMelodyNet
from melody_cnn.utils import resolve_path


def build_model(cfg: dict, device: torch.device) -> AnisoCoordMelodyNet:
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
    return model


def make_dataset(cfg: dict, cache_dir: Path, ids) -> Pop909RollDataset:
    return Pop909RollDataset(
        cache_dir,
        ids,
        steps_per_beat=cfg["steps_per_beat"],
        beats_per_bar=cfg["beats_per_bar"],
        segment_bars=cfg["segment_bars"],
        segment_hop_bars=cfg["segment_hop_bars"],
        min_onsets=cfg["min_onsets"],
        binary_label=bool(cfg.get("binary_label", True)),
        use_chord=bool(cfg.get("use_chord", True)),
        use_metric=bool(cfg.get("use_metric", True)),
        augment=False,
    )


@torch.no_grad()
def stitch_scores(model, loader, device):
    """Average melody probability per unique note across all covering segments."""
    prob_sum: dict = defaultdict(float)
    count: dict = defaultdict(int)
    label_of: dict = {}
    for batch in tqdm(loader, desc="  infer", leave=False):
        roll = batch["roll"].to(device, non_blocking=True)
        key = batch["key_vec"].to(device, non_blocking=True)
        probs = torch.softmax(model(roll, key).float(), dim=1)[:, 1].cpu().numpy()
        mask = batch["onset_mask"].numpy()
        labels = batch["labels"].numpy()
        songs = batch["song_id"].numpy()
        offsets = batch["t0"].numpy()
        for bi in range(len(songs)):
            pi, ti = np.nonzero(mask[bi])
            if pi.size == 0:
                continue
            sid = int(songs[bi])
            off = int(offsets[bi])
            vals = probs[bi][pi, ti]
            for pp, tt, val in zip(pi.tolist(), ti.tolist(), vals.tolist()):
                k = (sid, pp, off + tt)
                prob_sum[k] += val
                count[k] += 1
                label_of.setdefault(k, int(labels[bi][pp, tt]))
    keys = list(prob_sum.keys())
    y = np.fromiter((label_of[k] for k in keys), dtype=np.int64, count=len(keys))
    s = np.fromiter((prob_sum[k] / count[k] for k in keys), dtype=np.float64, count=len(keys))
    sid = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
    return y, s, sid


def binary_stats(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "thr": round(float(thr), 4),
        "acc": float(np.mean(pred == pos)),
        "mel_p": prec,
        "mel_r": rec,
        "mel_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_notes": int(len(y)),
    }


def sweep_best_f1(y: np.ndarray, s: np.ndarray, lo: float = 0.02, hi: float = 0.98, n: int = 193) -> dict:
    best = None
    for thr in np.linspace(lo, hi, n):
        cur = binary_stats(y, s, thr)
        if best is None or cur["mel_f1"] > best["mel_f1"]:
            best = cur
    return best


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    keep = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    tp = tp[keep]
    fp = fp[keep]
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(np.sum(ys == 1)), 1)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def per_song_f1(y: np.ndarray, s: np.ndarray, sid: np.ndarray, thr: float) -> dict:
    scores = [
        binary_stats(y[sid == k], s[sid == k], thr)["mel_f1"] for k in np.unique(sid)
    ]
    arr = np.asarray(scores, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n_songs": int(arr.size)}


def evaluate_split(model, cfg, cache_dir, ids, device) -> dict:
    ds = make_dataset(cfg, cache_dir, ids)
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    y, s, sid = stitch_scores(model, loader, device)
    return {
        "at_0.5": binary_stats(y, s, 0.5),
        "best_f1": sweep_best_f1(y, s),
        "ap": average_precision(y, s),
        "per_song_at_0.5": per_song_f1(y, s, sid, 0.5),
        "_y": y,
        "_s": s,
        "_sid": sid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-scores", action="store_true", help="save per-note y/score/song arrays")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_path = args.ckpt if args.ckpt.is_absolute() else resolve_path(root, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    cache_dir = resolve_path(root, cfg["cache_dir"])
    splits = json.loads((ckpt_path.parent / "split.json").read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val = evaluate_split(model, cfg, cache_dir, splits["val"], device)
    test = evaluate_split(model, cfg, cache_dir, splits["test"], device)
    thr = val["best_f1"]["thr"]
    test_at_val_thr = binary_stats(test["_y"], test["_s"], thr)
    test_per_song = per_song_f1(test["_y"], test["_s"], test["_sid"], thr)

    report = {
        "ckpt": str(ckpt_path),
        "cache_dir": str(cache_dir),
        "epoch": ckpt.get("epoch"),
        "in_channels": cfg.get("in_channels"),
        "best_thr_from_val": thr,
        "val": {
            "at_0.5": val["at_0.5"],
            "best_f1": val["best_f1"],
            "ap": val["ap"],
            "per_song_at_0.5": val["per_song_at_0.5"],
        },
        "test": {
            "at_0.5": test["at_0.5"],
            "at_val_thr": test_at_val_thr,
            "best_f1": test["best_f1"],
            "ap": test["ap"],
            "per_song_at_val_thr": test_per_song,
        },
    }

    def line(tag, m):
        return (
            f"{tag:<18} thr={m['thr']:.3f} acc={m['acc']:.4f} "
            f"P={m['mel_p']:.4f} R={m['mel_r']:.4f} F1={m['mel_f1']:.4f}"
        )

    print(f"ckpt={ckpt_path.name} ({ckpt_path.parent.name}) epoch={ckpt.get('epoch')} "
          f"in_ch={cfg.get('in_channels')} notes_test={test['at_0.5']['n_notes']}")
    print(line("val @0.5", val["at_0.5"]))
    print(line("val @best", val["best_f1"]))
    print(f"val  AP={val['ap']:.4f}  per_song_F1@0.5={val['per_song_at_0.5']['mean']:.4f}"
          f"±{val['per_song_at_0.5']['std']:.4f}")
    print(line("test @0.5", test["at_0.5"]))
    print(line("test @val_thr", test_at_val_thr))
    print(line("test @best", test["best_f1"]))
    print(f"test AP={test['ap']:.4f}  per_song_F1@val_thr={test_per_song['mean']:.4f}"
          f"±{test_per_song['std']:.4f}")

    out_path = args.out or (ckpt_path.parent / "eval_stitched.json")
    out_path = out_path if out_path.is_absolute() else resolve_path(root, out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if args.dump_scores:
        np.savez_compressed(
            out_path.with_suffix(".scores.npz"),
            y=test["_y"],
            score=test["_s"],
            song=test["_sid"],
            thr=np.float32(thr),
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
