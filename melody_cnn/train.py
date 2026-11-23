"""Train anisotropic melody CNN with LStoM pitch features and focal loss."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from melody_cnn.data.dataset import Pop909RollDataset, split_song_ids
from melody_cnn.models import AnisoCoordMelodyNet
from melody_cnn.utils import load_config, resolve_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_focal(logits, labels, onset_mask, alpha=0.25, gamma=2.0):
    b, c, p, t = logits.shape
    logits_f = logits.permute(0, 2, 3, 1).reshape(-1, c)
    labels_f = labels.reshape(-1)
    mask_f = onset_mask.reshape(-1)
    if mask_f.sum() == 0:
        return logits_f.sum() * 0.0
    logits_m = logits_f[mask_f]
    labels_m = labels_f[mask_f]
    ce = F.cross_entropy(logits_m, labels_m, reduction="none")
    p_t = torch.exp(-ce)
    alpha_t = torch.where(
        labels_m == 1,
        torch.full_like(ce, alpha),
        torch.full_like(ce, 1.0 - alpha),
    )
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def masked_ce(logits, labels, onset_mask, class_weights=None):
    b, c, p, t = logits.shape
    logits_f = logits.permute(0, 2, 3, 1).reshape(-1, c)
    labels_f = labels.reshape(-1)
    mask_f = onset_mask.reshape(-1)
    if mask_f.sum() == 0:
        return logits_f.sum() * 0.0
    logits_m = logits_f[mask_f]
    labels_m = labels_f[mask_f]
    w = torch.tensor(class_weights, dtype=logits_m.dtype, device=logits_m.device) if class_weights else None
    return F.cross_entropy(logits_m, labels_m, weight=w)


@torch.no_grad()
def eval_epoch(model, loader, device, focal_alpha, focal_gamma, loss_kind="focal", class_weights=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = total = 0
    tp = fp = fn = 0
    for batch in loader:
        roll = batch["roll"].to(device)
        mask = batch["onset_mask"].to(device)
        labels = batch["labels"].to(device)
        key = batch["key_vec"].to(device)
        logits = model(roll, key)
        if loss_kind == "ce":
            loss = masked_ce(logits, labels, mask, class_weights)
        else:
            loss = masked_focal(logits, labels, mask, focal_alpha, focal_gamma)
        total_loss += float(loss.item())
        n_batches += 1
        pred = logits.argmax(dim=1)
        m = mask
        correct += int((pred[m] == labels[m]).sum().item())
        total += int(m.sum().item())
        p_m = pred[m] == 1
        y_m = labels[m] == 1
        tp += int((p_m & y_m).sum().item())
        fp += int((p_m & ~y_m).sum().item())
        fn += int((~p_m & y_m).sum().item())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": correct / max(total, 1),
        "mel_p": prec,
        "mel_r": rec,
        "mel_f1": f1,
        "n_onsets": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out-dir", type=str, default=None, help="override cfg['out_dir']")
    ap.add_argument("--cache-dir", type=str, default=None, help="override cfg['cache_dir']")
    ap.add_argument("--loss", type=str, default=None, choices=["ce", "focal"])
    ap.add_argument("--class-weights", type=float, nargs=2, default=None, metavar=("NEG","POS"))
    ap.add_argument("--focal-alpha", type=float, default=None)
    ap.add_argument("--focal-gamma", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--segment-bars", type=int, default=None)
    ap.add_argument("--segment-hop-bars", type=int, default=None)
    ap.add_argument("--no-chord", action="store_true", help="ablate chord channels")
    ap.add_argument("--no-metric", action="store_true", help="ablate metric-strength channel")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = args.config if args.config.is_absolute() else resolve_path(root, args.config)
    cfg = load_config(cfg_path)
    for key, val in (
        ("out_dir", args.out_dir),
        ("cache_dir", args.cache_dir),
        ("loss", args.loss),
        ("focal_alpha", args.focal_alpha),
        ("focal_gamma", args.focal_gamma),
        ("epochs", args.epochs),
        ("num_workers", args.num_workers),
        ("segment_bars", args.segment_bars),
        ("segment_hop_bars", args.segment_hop_bars),
    ):
        if val is not None:
            cfg[key] = val
    if args.class_weights is not None:
        cfg["class_weights"] = list(args.class_weights)
    if args.no_chord:
        cfg["use_chord"] = False
    if args.no_metric:
        cfg["use_metric"] = False
    set_seed(int(cfg["seed"]))

    cache_dir = resolve_path(root, cfg["cache_dir"])
    pop909 = resolve_path(root, cfg["pop909_dir"])
    out_dir = resolve_path(root, cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (cache_dir / "index.json").exists():
        import sys
        from melody_cnn.data.prepare_pop909 import main as prep_main

        sys.argv = [
            "prepare_pop909",
            "--pop909",
            str(pop909),
            "--out",
            str(cache_dir),
            "--steps-per-beat",
            str(cfg["steps_per_beat"]),
            "--beats-per-bar",
            str(cfg["beats_per_bar"]),
            "--pitch-min",
            str(cfg["pitch_min"]),
            "--pitch-max",
            str(cfg["pitch_max"]),
        ]
        prep_main()

    train_ids, val_ids, test_ids = split_song_ids(
        cache_dir, cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
    )
    with (out_dir / "split.json").open("w", encoding="utf-8") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f)

    ds_kwargs = dict(
        steps_per_beat=cfg["steps_per_beat"],
        beats_per_bar=cfg["beats_per_bar"],
        segment_bars=cfg["segment_bars"],
        segment_hop_bars=cfg["segment_hop_bars"],
        min_onsets=cfg["min_onsets"],
        binary_label=bool(cfg.get("binary_label", True)),
        use_chord=bool(cfg.get("use_chord", True)),
        use_metric=bool(cfg.get("use_metric", True)),
    )
    train_ds = Pop909RollDataset(
        cache_dir,
        train_ids,
        augment=True,
        aug_pitch_prob=float(cfg.get("aug_pitch_prob", 0.5)),
        aug_pitch_range=int(cfg.get("aug_pitch_range", 3)),
        aug_mo_prob=float(cfg.get("aug_mo_prob", 0.25)),
        **ds_kwargs,
    )
    val_ds = Pop909RollDataset(cache_dir, val_ids, augment=False, **ds_kwargs)
    test_ds = Pop909RollDataset(cache_dir, test_ids, augment=False, **ds_kwargs)

    loader_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = AnisoCoordMelodyNet(
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        num_classes=cfg["num_classes"],
        k_pitch=cfg["kernel_pitch"],
        k_time=cfg["kernel_time"],
        num_blocks=cfg["num_blocks"],
        dropout=cfg["dropout"],
        cond_dim=cfg["cond_dim"],
    ).to(device)

    focal_alpha = float(cfg.get("focal_alpha", 0.25))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    loss_kind = cfg.get("loss", "focal")
    class_weights = cfg.get("class_weights", None)

    def compute_loss(logits, labels, mask):
        if loss_kind == "ce":
            return masked_ce(logits, labels, mask, class_weights)
        return masked_focal(logits, labels, mask, focal_alpha, focal_gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp")) and device.type == "cuda")

    best_f1 = -1.0
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            roll = batch["roll"].to(device)
            mask = batch["onset_mask"].to(device)
            labels = batch["labels"].to(device)
            key = batch["key_vec"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(roll, key)
                loss = compute_loss(logits, labels, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            n += 1
            pbar.set_postfix(loss=running / max(n, 1))

        val_m = eval_epoch(model, val_loader, device, focal_alpha, focal_gamma, loss_kind, class_weights)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"Epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_acc={val_m['acc']:.4f} mel_f1={val_m['mel_f1']:.4f} "
            f"P={val_m['mel_p']:.3f} R={val_m['mel_r']:.3f}"
        )
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if val_m["mel_f1"] > best_f1:
            best_f1 = val_m["mel_f1"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": val_m}, out_dir / "best.pt")
            print(f"  saved best.pt (mel_f1={best_f1:.4f})")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_m = eval_epoch(model, test_loader, device, focal_alpha, focal_gamma, loss_kind, class_weights)
    print("TEST:", test_m)
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_m, f, indent=2)


if __name__ == "__main__":
    main()
