"""LStoM baseline reproduction (Kosta et al., ISMIR 2022) on our own split.

Notes are classified with the paper's six handcrafted features and a 6-layer
biLSTM (hidden 140) -> FC 20 -> FC 2, trained with focal loss (alpha=0.25,
gamma=2) and the paper's PSaugm / MOaugm augmentation. Running this on the same
train/val/test split as our CNN makes the two numbers directly comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from melody_cnn.data.key_utils import scale_pitch_classes
from melody_cnn.utils import resolve_path

N_FEAT = 6


class LStoM(nn.Module):
    def __init__(self, hidden=140, layers=6, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            N_FEAT, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden * 2, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc2(torch.relu(self.fc1(h)))


def _scalar(v):
    return np.atleast_1d(v).ravel()[0]


def _song_features(z, augment=False, rng=None):
    """Return (feats[N,6], labels[N]) for one song."""
    o = z["onsets"]
    if len(o) == 0:
        return None, None
    pitch = o[:, 1].astype(np.float32)
    start = o[:, 2].astype(np.float32)
    end = o[:, 3].astype(np.float32)
    label = (o[:, 6] == 1).astype(np.int64)
    beats_per_bar = float(_scalar(z["beats_per_bar"]))

    tonic = int(_scalar(z["key_tonic"]))
    mode = int(_scalar(z["key_mode"]))

    shift = 0
    if augment and rng is not None:
        if rng.random() < 0.5:
            shift = int(rng.integers(-6, 6))  # PSaugm: tau in [-6, 5]
            pitch = pitch + shift

    # distance to nearest simultaneously sounding lower / higher note
    n = len(o)
    dist_below = np.zeros(n, dtype=np.float32)
    dist_above = np.zeros(n, dtype=np.float32)
    overlaps = (start[None, :] < end[:, None] + 1e-6) & (end[None, :] > start[:, None])
    for i in range(n):
        m = overlaps[i]
        p = pitch[m]
        below = p[p < pitch[i] - 1e-6]
        above = p[p > pitch[i] + 1e-6]
        dist_below[i] = 0.0 if below.size == 0 else pitch[i] - below.max()
        dist_above[i] = 0.0 if above.size == 0 else above.min() - pitch[i]

    pos_in_bar = np.mod(start, beats_per_bar).astype(np.float32)
    dur = (end - start).astype(np.float32)
    scale_pcs = scale_pitch_classes((tonic + shift) % 12, mode)
    in_scale = np.array(
        [1.0 if int(round(p_)) % 12 in scale_pcs else 0.0 for p_ in pitch],
        dtype=np.float32,
    )

    feats = np.stack([pitch, dur, dist_below, dist_above, pos_in_bar, in_scale], axis=1)
    return feats, label


def load_split(cache_dir: Path, ids, augment=False, seed=0):
    rng = np.random.default_rng(seed)
    songs = []
    for sid in ids:
        z = np.load(cache_dir / f"{int(sid):03d}.npz", allow_pickle=True)
        f, y = _song_features(z, augment=augment, rng=rng if augment else None)
        if f is None or len(f) < 2:
            continue
        songs.append((f, y))
    return songs


def normalize(songs, mean=None, std=None):
    if mean is None:
        allf = np.concatenate([f for f, _ in songs], axis=0)
        mean = allf.mean(axis=0)
        std = allf.std(axis=0)
        std[std < 1e-6] = 1.0
    out = [((f - mean) / std).astype(np.float32) for f, _ in songs]
    return list(zip(out, [y for _, y in songs])), mean, std


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(logits, target, reduction="none")
    p_t = torch.exp(-ce)
    alpha_t = torch.where(target == 1, alpha, 1.0 - alpha)
    return alpha_t * (1.0 - p_t).pow(gamma) * ce


def make_batches(songs, batch_size, shuffle=False, seed=0):
    order = list(range(len(songs)))
    if shuffle:
        order = list(np.random.default_rng(seed).permutation(order))
    # bucket by length to minimise padding
    order.sort(key=lambda i: len(songs[i][0]))
    out = []
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        if shuffle:
            np.random.default_rng(seed + i).shuffle(idx)
        out.append([songs[j] for j in idx])
    return out


def collate(batch, device):
    lens = [len(f) for f, _ in batch]
    t_max = max(lens)
    x = np.zeros((len(batch), t_max, N_FEAT), dtype=np.float32)
    y = np.zeros((len(batch), t_max), dtype=np.int64)
    m = np.zeros((len(batch), t_max), dtype=bool)
    for i, (f, labels) in enumerate(batch):
        x[i, :len(f)] = f
        y[i, :len(labels)] = labels
        m[i, :len(labels)] = True
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(m).to(device))


@torch.no_grad()
def evaluate(model, songs, device, batch_size=16):
    model.eval()
    tp = fp = fn = 0
    correct = total = 0
    n_neg = 0
    for batch in make_batches(songs, batch_size):
        x, y, m = collate(batch, device)
        logits = model(x)
        pred = logits.argmax(dim=2)
        m_np = m.cpu().numpy()
        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()
        correct += int(((pred_np == y_np) & m_np).sum())
        total += int(m_np.sum())
        pos = (y_np == 1) & m_np
        neg = (y_np != 1) & m_np
        tp += int(((pred_np == 1) & pos).sum())
        fp += int(((pred_np == 1) & neg).sum())
        fn += int(((pred_np != 1) & pos).sum())
        n_neg += int(neg.sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return {
        "acc": correct / max(total, 1),
        "mel_p": p,
        "mel_r": r,
        "mel_f1": f1,
        "vfa": fp / max(n_neg, 1),
        "n_notes": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache/pop909_rolls_veldur"))
    ap.add_argument("--split", type=Path, default=Path("runs/aniso_bar_key_v2/split.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/lstom_repro"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=140)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.75, help="focal alpha for the melody class")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else resolve_path(root, args.cache_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(resolve_path(root, args.split).read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print("loading features...")
    train = load_split(cache_dir, splits["train"], augment=False)
    val = load_split(cache_dir, splits["val"], augment=False)
    test = load_split(cache_dir, splits["test"], augment=False)
    train_n, mean, std = normalize(train)
    val_n, _, _ = normalize(val, mean, std)
    test_n, _, _ = normalize(test, mean, std)
    print(f"songs: train={len(train_n)} val={len(val_n)} test={len(test_n)}")

    model = LStoM(hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {"mel_f1": -1.0}
    history = []
    for epoch in range(args.epochs):
        model.train()
        batches = make_batches(train_n, args.batch_size, shuffle=True, seed=epoch)
        total_loss = 0.0
        n_notes = 0
        for batch in batches:
            x, y, m = collate(batch, device)
            logits = model(x)
            loss_all = focal_loss(logits.reshape(-1, 2), y.reshape(-1), alpha=args.alpha)
            loss = (loss_all * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach()) * int(m.sum())
            n_notes += int(m.sum())
        vm = evaluate(model, val_n, device, args.batch_size)
        history.append({"epoch": epoch, "train_loss": total_loss / max(n_notes, 1), **{f"val_{k}": v for k, v in vm.items()}})
        print(f"e{epoch:02d} loss={total_loss/max(n_notes,1):.4f} val_acc={vm['acc']:.4f} "
              f"F1={vm['mel_f1']:.4f} P={vm['mel_p']:.4f} R={vm['mel_r']:.4f} VFA={vm['vfa']:.4f}", flush=True)
        if vm["mel_f1"] > best["mel_f1"]:
            best = {**vm, "epoch": epoch}
            torch.save({"model": model.state_dict(), "cfg": vars(args), "mean": mean, "std": std}, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    tm = evaluate(model, test_n, device, args.batch_size)
    print(f"TEST acc={tm['acc']:.4f} P={tm['mel_p']:.4f} R={tm['mel_r']:.4f} F1={tm['mel_f1']:.4f} VFA={tm['vfa']:.4f}", flush=True)
    (out_dir / "test_metrics.json").write_text(json.dumps({"best_val": best, "test": tm}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
