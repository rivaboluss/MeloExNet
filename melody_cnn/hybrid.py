"""Hybrid melody extractor: local piano-roll CNN + note-level BiGRU.

Each note onset is described by
  * a local patch of the piano-roll around (pitch, time), encoded by a small CNN
  * handcrafted note features (pitch, duration, velocity, chord, metre, intervals)

The per-note vectors are then ordered by onset within a whole song and passed
through a bidirectional GRU, so melody continuity is modelled across the whole
piece instead of per 4-bar window.

Use --no-patch to drop the CNN branch and keep only handcrafted + BiGRU, which
gives the ablation "sequence model alone".
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from melody_cnn.data.chord_utils import CHORD_TYPES
from melody_cnn.data.key_utils import scale_pitch_classes

PATCH_P = 8   # +-8 semitones
PATCH_T = 6   # +-6 grid steps (1.5 beats at 4 steps/beat)
PATCH_H = 2 * PATCH_P + 1
PATCH_W = 2 * PATCH_T + 1
ROLL_CH = 12
N_FEAT = 20

# feature groups used by the --drop ablation (indices into x / roll channels)
FEAT_GROUPS = {
    "chord": [6, 7, 8, 9, 10],
    "metric": [11],
    "velocity": [2, 3],
    "duration": [1],
    "context": [12, 13, 14, 15, 16, 17, 18, 19],
}
ROLL_GROUPS = {
    "chord": [6, 7, 8, 9, 10],
    "metric": [11],
    "velocity": [3, 5],
    "duration": [4],
}
FEAT_GROUPS["notes"] = list(range(N_FEAT))


# ---------------------------------------------------------------- features
def _scalar(v):
    return np.atleast_1d(v).ravel()[0]


def _metric_strength(local_step: int, steps_per_beat: int) -> float:
    if local_step == 0:
        return 1.0
    if local_step % steps_per_beat == 0:
        return 0.7
    if local_step % max(steps_per_beat // 2, 1) == 0:
        return 0.4
    return 0.2


def song_record(path: Path) -> dict | None:
    """Note-level features extracted from one cached song npz."""
    z = np.load(path, allow_pickle=True)
    o = z["onsets"]
    if o.ndim != 2 or len(o) < 2:
        return None
    n = len(o)
    pitch = o[:, 1].astype(np.float32)
    start = o[:, 2].astype(np.float32)
    end = o[:, 3].astype(np.float32)
    t_idx = o[:, 4].astype(np.int64)
    p_idx = o[:, 5].astype(np.int64)
    y = (o[:, 6] == 1).astype(np.int64)
    vel = o[:, 7].astype(np.float32) if o.shape[1] > 7 else np.full(n, 64.0, np.float32)

    steps_per_beat = int(_scalar(z["steps_per_beat"]))
    beats_per_bar = int(_scalar(z["beats_per_bar"]))
    steps_per_bar = steps_per_beat * beats_per_bar
    tonic = int(_scalar(z["key_tonic"]))
    mode = int(_scalar(z["key_mode"]))

    # simultaneous-note relations
    overlap = (start[None, :] < end[:, None] + 1e-6) & (end[None, :] > start[:, None])
    lower = np.where(overlap & (pitch[None, :] < pitch[:, None]), pitch[None, :], -1.0).max(axis=1)
    upper = np.where(overlap & (pitch[None, :] > pitch[:, None]), pitch[None, :], 1e9).min(axis=1)
    has_lower = lower >= 0
    has_upper = upper < 1e8
    dist_below = np.where(has_lower, pitch - lower, 0.0) / 12.0
    dist_above = np.where(has_upper, upper - pitch, 0.0) / 12.0
    is_highest = (~has_upper).astype(np.float32)
    sim_count = np.maximum(overlap.sum(axis=1), 1)
    sim_mean = (overlap * pitch[None, :]).sum(axis=1) / sim_count
    pitch_rel = (pitch - sim_mean) / 12.0

    # previous / next note in time order
    order = np.lexsort((pitch, start))
    prev_int = np.zeros(n, np.float32)
    next_int = np.zeros(n, np.float32)
    prev_ioi = np.zeros(n, np.float32)
    next_ioi = np.zeros(n, np.float32)
    if n > 1:
        a, b = order[:-1], order[1:]
        delta_pitch = (pitch[b] - pitch[a]) / 12.0
        delta_time = np.maximum(start[b] - start[a], 0.0) / 4.0
        next_int[a] = delta_pitch
        prev_int[b] = delta_pitch
        next_ioi[a] = np.minimum(delta_time, 1.0)
        prev_ioi[b] = np.minimum(delta_time, 1.0)

    # chord features from the cached per-step arrays
    root_arr = z["chord_root"].astype(np.int64)
    type_arr = z["chord_type"].astype(np.int64)
    phase_arr = z["chord_phase"].astype(np.float32)
    dense = t_idx - t_idx // (steps_per_bar + 1)
    dense = np.clip(dense, 0, len(root_arr) - 1)
    root = root_arr[dense]
    ctype = type_arr[dense]
    phase = phase_arr[dense]
    pc = pitch.astype(np.int64) % 12
    valid = (root >= 0) & (ctype >= 0)
    rel = np.zeros(n, np.float32)
    cdist = np.zeros(n, np.float32)
    ctone = np.zeros(n, np.float32)
    for i in np.where(valid)[0]:
        tones = CHORD_TYPES[int(ctype[i])][1]
        r = int(root[i])
        rel[i] = ((int(pc[i]) - r) % 12) / 12.0
        d = min(min((int(pc[i]) - (r + iv)) % 12, ((r + iv) - int(pc[i])) % 12) for iv in tones)
        cdist[i] = min(d / 6.0, 1.0)
        ctone[i] = 1.0 if d == 0 else 0.0

    scale_pcs = scale_pitch_classes(tonic, mode)
    in_scale = np.array([1.0 if int(p) % 12 in scale_pcs else 0.0 for p in pitch], np.float32)
    local_step = np.round(start * steps_per_beat).astype(np.int64) % steps_per_bar
    metric = np.array([_metric_strength(int(s), steps_per_beat) for s in local_step], np.float32)

    # per-column max velocity
    max_vel = np.zeros(int(t_idx.max()) + 1, np.float32)
    np.maximum.at(max_vel, t_idx, vel)
    vel_rel = (vel - max_vel[t_idx]) / 127.0

    feats = np.stack(
        [
            (pitch - 60.0) / 24.0,
            np.minimum(end - start, 4.0) / 4.0,
            vel / 127.0,
            vel_rel,
            np.mod(start, beats_per_bar) / beats_per_bar,
            in_scale,
            ctone,
            cdist,
            rel,
            np.clip(phase, 0.0, 1.0),
            valid.astype(np.float32),
            metric,
            dist_below,
            dist_above,
            prev_int,
            next_int,
            prev_ioi,
            next_ioi,
            is_highest,
            pitch_rel,
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "path": str(path),
        "x": feats,
        "y": y,
        "p_idx": p_idx,
        "t_idx": t_idx,
        "tonic": tonic,
        "mode": mode,
        "steps_per_beat": steps_per_beat,
        "steps_per_bar": steps_per_bar,
    }


# ---------------------------------------------------------------- patch store
class PatchStore:
    """LRU cache of padded piano-roll tensors (state channel already scaled)."""

    def __init__(self, capacity: int = 24, zero_ch=None):
        self.capacity = capacity
        self.zero_ch = set(zero_ch or ())
        self._cache: dict[str, np.ndarray] = {}
        self._order: list[str] = []

    def get(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        z = np.load(path, allow_pickle=True)
        roll = z["feats"].astype(np.float32)
        roll[0] = roll[0] / 3.0
        for ch in self.zero_ch:
            roll[ch] = 0.0
        pad = np.pad(roll, ((0, 0), (PATCH_P, PATCH_P), (PATCH_T, PATCH_T)))
        self._cache[path] = pad
        self._order.append(path)
        if len(self._order) > self.capacity:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return pad


# ---------------------------------------------------------------- batching
def make_batches(songs, batch_size, shuffle, seed=0):
    order = list(range(len(songs)))
    if shuffle:
        order = list(np.random.default_rng(seed).permutation(order))
    order.sort(key=lambda i: len(songs[i]["y"]))
    out = []
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        if shuffle:
            np.random.default_rng(seed + i).shuffle(idx)
        out.append([songs[j] for j in idx])
    return out


def collate(batch, store: PatchStore, mean, std, use_patch: bool, augment: bool, rng):
    b = len(batch)
    lens = [len(s["y"]) for s in batch]
    n_max = max(lens)
    x = np.zeros((b, n_max, N_FEAT), np.float32)
    y = np.full((b, n_max), -1, np.int64)
    mask = np.zeros((b, n_max), bool)
    patches = np.zeros((b, n_max, ROLL_CH, PATCH_H, PATCH_W), np.float32) if use_patch else None

    for i, s in enumerate(batch):
        n = lens[i]
        nf = (s["x"] - mean) / std
        y[i, :n] = s["y"]
        mask[i, :n] = True
        if not use_patch:
            x[i, :n] = nf
            continue

        shift = 0
        if augment and rng.random() < 0.5:
            shift = int(rng.integers(-3, 4))
        if shift:
            nf = nf.copy()
            nf[:, 0] += shift / 24.0
            scale = scale_pitch_classes((s["tonic"] + shift) % 12, s["mode"])
            pcs = (s["x"][:, 0] * 24.0 + 60.0 + shift).round().astype(np.int64) % 12
            nf[:, 5] = np.array([1.0 if int(p) in scale else 0.0 for p in pcs], np.float32)
        x[i, :n] = nf

        pad = store.get(s["path"])
        rows = s["p_idx"]
        cols = s["t_idx"]
        for k in range(n):
            patch = pad[:, rows[k] : rows[k] + PATCH_H, cols[k] : cols[k] + PATCH_W]
            if shift:
                patch = np.roll(patch, shift, axis=1)
                if shift > 0:
                    patch[:, :shift] = 0.0
                else:
                    patch[:, shift:] = 0.0
            patches[i, k] = patch

    out = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "mask": torch.from_numpy(mask),
    }
    if use_patch:
        out["patch"] = torch.from_numpy(patches)
    return out


# ---------------------------------------------------------------- model
class PatchEncoder(nn.Module):
    def __init__(self, in_ch: int = ROLL_CH, out_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(64 * 4, out_dim)

    def forward(self, p):
        b, n, c, h, w = p.shape
        z = self.body(p.reshape(b * n, c, h, w)).flatten(1)
        return self.fc(z).reshape(b, n, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridNet(nn.Module):
    def __init__(self, use_patch=True, hidden=128, layers=2, dropout=0.2, seq="gru"):
        super().__init__()
        self.use_patch = use_patch
        self.seq = seq
        feat_dim = 0
        if use_patch:
            self.patch_enc = PatchEncoder()
            feat_dim += 128
        self.note_mlp = nn.Sequential(
            nn.Linear(N_FEAT, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(),
        )
        feat_dim += 64
        if seq == "transformer":
            self.pos = PositionalEncoding(feat_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=4, dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True, norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            self.encoder = nn.GRU(
                feat_dim, hidden, num_layers=layers, batch_first=True,
                bidirectional=True, dropout=dropout if layers > 1 else 0.0,
            )
            feat_dim = hidden * 2
        self.head = nn.Linear(feat_dim, 2)

    def forward(self, patch, x, mask=None):
        z = self.note_mlp(x)
        if self.use_patch:
            z = torch.cat([self.patch_enc(patch), z], dim=-1)
        if self.seq == "transformer":
            pad = ~mask if mask is not None else None
            h = self.encoder(self.pos(z), src_key_padding_mask=pad)
        else:
            h, _ = self.encoder(z)
        return self.head(h)


def masked_focal(logits, target, mask, alpha=0.75, gamma=2.0):
    # padded positions carry target=-1; clamp so CE is defined, they are masked out below
    ce = F.cross_entropy(logits, target.clamp(min=0), reduction="none")
    p_t = torch.exp(-ce)
    alpha_t = torch.where(target == 1, alpha, 1.0 - alpha)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, songs, store, mean, std, use_patch, device, batch_size=8):
    model.eval()
    ys, ps, sids = [], [], []
    for bi, batch in enumerate(make_batches(songs, batch_size, False)):
        out = collate(batch, store, mean, std, use_patch, False, None)
        patch = out["patch"].to(device) if use_patch else None
        logits = model(patch, out["x"].to(device), out["mask"].to(device))
        prob = torch.softmax(logits, dim=-1)[..., 1]
        m = out["mask"].numpy()
        ys.append(out["y"].numpy()[m])
        ps.append(prob.cpu().numpy()[m])
        for j, s in enumerate(batch):
            sids.extend([s["path"]] * int(m[j].sum()))
    return np.concatenate(ys), np.concatenate(ps), np.array(sids)


def metrics(y, p, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred != 1) & (y == 1)).sum())
    tn = int(((pred != 1) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "thr": float(thr),
        "acc": (tp + tn) / max(len(y), 1),
        "mel_p": precision,
        "mel_r": recall,
        "mel_f1": f1,
        "vfa": fp / max(tn + fp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_notes": int(len(y)),
    }


def best_threshold(y, p):
    grid = np.arange(0.30, 0.751, 0.005)
    best = (-1.0, 0.5)
    for t in grid:
        f1 = metrics(y, p, float(t))["mel_f1"]
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def per_song_scores(y, p, sids, thr):
    vals = []
    for sid in np.unique(sids):
        m = sids == sid
        vals.append(metrics(y[m], p[m], thr)["mel_f1"])
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache/pop909_rolls_chord"))
    ap.add_argument("--split", type=Path, default=Path("runs/aniso_bar_key_v2/split.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/hybrid_cnn_gru"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--no-patch", action="store_true")
    ap.add_argument("--seq", choices=["gru", "transformer"], default="gru")
    ap.add_argument("--drop", default="", help="comma list: chord,metric,velocity,duration,context,notes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0, help="linear warmup epochs, then cosine decay")
    ap.add_argument("--dump-scores", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((root / args.split).read_text(encoding="utf-8"))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(ids):
        out = []
        for sid in ids:
            r = song_record(cache_dir / f"{int(sid):03d}.npz")
            if r is not None:
                out.append(r)
        return out

    t0 = time.time()
    print("loading songs ...", flush=True)
    train = load(split["train"])
    val = load(split["val"])
    test = load(split["test"])
    print(f"songs train={len(train)} val={len(val)} test={len(test)} ({time.time()-t0:.1f}s)", flush=True)

    drop = [g.strip() for g in args.drop.split(",") if g.strip()]
    feat_idx = sorted({i for g in drop for i in FEAT_GROUPS.get(g, [])})
    roll_ch = sorted({c for g in drop for c in ROLL_GROUPS.get(g, [])})
    if feat_idx:
        print(f"dropping note features {feat_idx}", flush=True)
        for songs in (train, val, test):
            for s in songs:
                s["x"][:, feat_idx] = 0.0
    if roll_ch:
        print(f"dropping roll channels {roll_ch}", flush=True)

    allx = np.concatenate([s["x"] for s in train], axis=0)
    mean = allx.mean(axis=0)
    std = allx.std(axis=0)
    std[std < 1e-6] = 1.0

    use_patch = not args.no_patch
    model = HybridNet(
        use_patch=use_patch, hidden=args.hidden, layers=args.layers, seq=args.seq
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = None
    if args.warmup > 0:
        warm = args.warmup

        def lr_lambda(ep):
            if ep < warm:
                return (ep + 1) / warm
            prog = (ep - warm) / max(1, args.epochs - warm)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    store = PatchStore(zero_ch=roll_ch)

    history = []
    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        batches = make_batches(train, args.batch_size, True, seed=epoch)
        total, seen = 0.0, 0
        t_start = time.time()
        for batch in batches:
            out = collate(batch, store, mean, std, use_patch, True, rng)
            patch = out["patch"].to(device, non_blocking=True) if use_patch else None
            logits = model(patch, out["x"].to(device, non_blocking=True),
                           out["mask"].to(device, non_blocking=True))
            loss = masked_focal(
                logits.reshape(-1, 2),
                out["y"].reshape(-1).to(device),
                out["mask"].reshape(-1).float().to(device),
                alpha=args.alpha,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * int(out["mask"].sum())
            seen += int(out["mask"].sum())

        yv, pv, _ = evaluate(model, val, store, mean, std, use_patch, device)
        vm = metrics(yv, pv, 0.5)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val": vm})
        print(
            f"e{epoch:02d} loss={total/max(seen,1):.4f} val_acc={vm['acc']:.4f} "
            f"F1={vm['mel_f1']:.4f} P={vm['mel_p']:.4f} R={vm['mel_r']:.4f} "
            f"({time.time()-t_start:.0f}s)",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if sched is not None:
            sched.step()
        if vm["mel_f1"] > best_f1:
            best_f1 = vm["mel_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": vars(args),
                    "mean": mean,
                    "std": std,
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    yv, pv, _ = evaluate(model, val, store, mean, std, use_patch, device)
    thr, _ = best_threshold(yv, pv)
    yt, pt, sidt = evaluate(model, test, store, mean, std, use_patch, device)

    report = {
        "cache_dir": str(cache_dir),
        "epoch": int(ckpt["epoch"]),
        "use_patch": use_patch,
        "best_thr_from_val": thr,
        "val": {"at_0.5": metrics(yv, pv, 0.5), "best_f1": metrics(yv, pv, thr),
                "ap": float(average_precision_score(yv, pv))},
        "test": {"at_0.5": metrics(yt, pt, 0.5), "at_val_thr": metrics(yt, pt, thr),
                 "ap": float(average_precision_score(yt, pt))},
    }
    m, s, n = per_song_scores(yt, pt, sidt, thr)
    report["test"]["per_song_at_val_thr"] = {"mean": m, "std": s, "n_songs": n}
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    tm = report["test"]["at_val_thr"]
    print(
        f"TEST acc={tm['acc']:.4f} P={tm['mel_p']:.4f} R={tm['mel_r']:.4f} "
        f"F1={tm['mel_f1']:.4f} AP={report['test']['ap']:.4f} VFA={tm['vfa']:.4f}",
        flush=True,
    )
    if args.dump_scores:
        np.savez_compressed(
            out_dir / "test_scores.npz", y=yt, score=pt, song=sidt, thr=np.float32(thr)
        )
    print(f"wrote {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
