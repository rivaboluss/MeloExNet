"""Multi-seed aggregation, ensembling and paired significance for the hybrid model.

For every run dir (each trained with a different seed) the checkpoint is reloaded,
val/test note scores are recomputed, and the per-seed metrics are reported as
mean +- std. The seeds are then averaged into one ensemble whose threshold is
tuned on the *averaged* validation probabilities, and the ensemble is compared to
a baseline score dump (e.g. the LStoM-style CNN) with a paired per-song test.

Usage:
  python -m melody_cnn.ensemble --runs runs/hybrid_cnn_gru runs/hybrid_seed1 runs/hybrid_seed2 \
      --baseline runs/lstom_repro_a75/eval_stitched.scores.npz --out runs/hybrid_ensemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score

from melody_cnn.baseline_lstom import LStoM, _song_features
from melody_cnn.hybrid import (
    HybridNet,
    PatchStore,
    best_threshold,
    evaluate,
    metrics,
    song_record,
)


def load_run(run_dir: Path, root: Path, split_ids, cache_dir: Path, device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = HybridNet(
        use_patch=not cfg.get("no_patch", False),
        hidden=cfg.get("hidden", 128),
        layers=cfg.get("layers", 2),
        seq=cfg.get("seq", "gru"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def load(ids):
        out = []
        for sid in ids:
            rec = song_record(cache_dir / f"{int(sid):03d}.npz")
            if rec is not None:
                out.append(rec)
        return out

    store = PatchStore()
    mean, std = ckpt["mean"], ckpt["std"]
    use_patch = not cfg.get("no_patch", False)
    vy, vp, _ = evaluate(model, load(split_ids["val"]), store, mean, std, use_patch, device)
    ty, tp, tsid = evaluate(model, load(split_ids["test"]), store, mean, std, use_patch, device)
    thr, _ = best_threshold(vy, vp)
    songs = np.array([int(Path(str(p)).stem) for p in tsid], dtype=np.int64)
    return {
        "run": run_dir.name,
        "seed": int(cfg.get("seed", -1)),
        "epoch": int(ckpt["epoch"]),
        "thr": thr,
        "val_y": vy, "val_p": vp,
        "test_y": ty, "test_p": tp, "test_song": songs,
    }


def per_song_f1(y, p, songs, thr) -> dict[int, float]:
    return {
        int(sid): metrics(y[songs == sid], p[songs == sid], thr)["mel_f1"]
        for sid in np.unique(songs)
    }


def paired_stats(a: dict[int, float], b: dict[int, float], n_boot: int = 10000) -> dict:
    common = sorted(set(a) & set(b))
    xa = np.array([a[k] for k in common])
    xb = np.array([b[k] for k in common])
    d = xa - xb
    t_stat, p_t = stats.ttest_rel(xa, xb)
    try:
        _, p_w = stats.wilcoxon(xa, xb)
    except ValueError:
        p_w = float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    return {
        "n_songs": len(common),
        "mean_a": float(xa.mean()),
        "mean_b": float(xb.mean()),
        "mean_delta": float(d.mean()),
        "std_delta": float(d.std(ddof=1)),
        "cohen_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
        "paired_t": float(t_stat),
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w),
        "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
    }


def lstom_scores(ckpt_path: Path, root: Path, split_ids, device):
    """Per-note test scores of the LStoM-style sequence baseline (argmax == thr 0.5)."""
    path = ckpt_path if ckpt_path.is_absolute() else root / ckpt_path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    cache_dir = Path(cfg["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    model = LStoM(hidden=cfg.get("hidden", 140), layers=cfg.get("layers", 6)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = ckpt["mean"], ckpt["std"]

    ys, ps, sids = [], [], []
    with torch.no_grad():
        for sid in split_ids["test"]:
            z = np.load(cache_dir / f"{int(sid):03d}.npz", allow_pickle=True)
            feats, labels = _song_features(z)
            if feats is None or len(feats) < 2:
                continue
            x = torch.from_numpy(((feats - mean) / std).astype(np.float32)).to(device)[None]
            prob = torch.softmax(model(x).float(), dim=-1)[0, :, 1].cpu().numpy()
            ys.append(labels)
            ps.append(prob)
            sids.append(np.full(len(labels), int(sid), dtype=np.int64))
    y, p, song = np.concatenate(ys), np.concatenate(ps), np.concatenate(sids)
    m = metrics(y, p, 0.5)
    m["ap"] = float(average_precision_score(y, p))
    ps_map = per_song_f1(y, p, song, 0.5)
    m["per_song"] = {"mean": float(np.mean(list(ps_map.values()))),
                     "std": float(np.std(list(ps_map.values()))), "n_songs": len(ps_map)}
    return m, ps_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", type=Path, default=Path("runs/aniso_bar_key_v2/split.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache/pop909_rolls_chord"))
    ap.add_argument("--lstom-ckpt", type=Path, default=None,
                    help="LStoM-style baseline checkpoint (runs/lstom_repro_a75/best.pt)")
    ap.add_argument("--out", type=Path, default=Path("runs/hybrid_ensemble"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir
    out_dir = args.out if args.out.is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    split_ids = json.loads((root / args.split).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = []
    for r in args.runs:
        rd = Path(r) if Path(r).is_absolute() else root / r
        print(f"scoring {rd.name} ...", flush=True)
        runs.append(load_run(rd, root, split_ids, cache_dir, device))

    ref_y = runs[0]["test_y"]
    ref_song = runs[0]["test_song"]
    for r in runs[1:]:
        if not np.array_equal(r["test_y"], ref_y) or not np.array_equal(r["test_song"], ref_song):
            raise SystemExit(f"test notes of {r['run']} do not align with {runs[0]['run']}")

    seeds = []
    for r in runs:
        m = metrics(r["test_y"], r["test_p"], r["thr"])
        m["ap"] = float(average_precision_score(r["test_y"], r["test_p"]))
        ps = per_song_f1(r["test_y"], r["test_p"], r["test_song"], r["thr"])
        m["per_song"] = {"mean": float(np.mean(list(ps.values()))),
                         "std": float(np.std(list(ps.values()))), "n_songs": len(ps)}
        seeds.append({"run": r["run"], "seed": r["seed"], "epoch": r["epoch"], **m, "_per_song": ps})

    f1s = np.array([s["mel_f1"] for s in seeds])
    aps = np.array([s["ap"] for s in seeds])
    psf1 = np.array([s["per_song"]["mean"] for s in seeds])

    val_p = np.mean([r["val_p"] for r in runs], axis=0)
    ens_thr, _ = best_threshold(runs[0]["val_y"], val_p)
    ens_p = np.mean([r["test_p"] for r in runs], axis=0)
    ens = metrics(ref_y, ens_p, ens_thr)
    ens["ap"] = float(average_precision_score(ref_y, ens_p))
    ens_ps = per_song_f1(ref_y, ens_p, ref_song, ens_thr)
    ens["per_song"] = {"mean": float(np.mean(list(ens_ps.values()))),
                       "std": float(np.std(list(ens_ps.values()))), "n_songs": len(ens_ps)}

    report = {
        "runs": [s["run"] for s in seeds],
        "n_seeds": len(seeds),
        "seeds": [{k: v for k, v in s.items() if not k.startswith("_")} for s in seeds],
        "seed_summary": {
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "ap_mean": float(aps.mean()), "ap_std": float(aps.std(ddof=1)) if len(aps) > 1 else 0.0,
            "per_song_f1_mean": float(psf1.mean()),
            "per_song_f1_std": float(psf1.std(ddof=1)) if len(psf1) > 1 else 0.0,
        },
        "ensemble": {"thr": ens_thr, **ens},
        "comparisons": {},
    }

    if len(seeds) > 1:
        report["comparisons"]["ensemble_vs_first_seed"] = paired_stats(ens_ps, seeds[0]["_per_song"])
        report["comparisons"]["ensemble_vs_mean_seed"] = paired_stats(
            ens_ps, {k: np.mean([s["_per_song"][k] for s in seeds]) for k in ens_ps}
        )

    if args.lstom_ckpt is not None:
        b_metrics, b_ps = lstom_scores(args.lstom_ckpt, root, split_ids, device)
        report["baseline"] = {"name": "lstom_repro(a75)", "thr": 0.5, **b_metrics}
        report["comparisons"]["ensemble_vs_baseline"] = paired_stats(ens_ps, b_ps)
        report["comparisons"]["first_seed_vs_baseline"] = paired_stats(seeds[0]["_per_song"], b_ps)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nper-seed")
    for s in seeds:
        print(f"  seed {s['seed']:>2} ({s['run']:<18}) thr={s['thr']:.3f} acc={s['acc']:.4f} "
              f"P={s['mel_p']:.4f} R={s['mel_r']:.4f} F1={s['mel_f1']:.4f} AP={s['ap']:.4f} "
              f"psF1={s['per_song']['mean']:.4f}")
    ss = report["seed_summary"]
    print(f"  mean±std  F1={ss['f1_mean']:.4f}±{ss['f1_std']:.4f}  AP={ss['ap_mean']:.4f}±{ss['ap_std']:.4f}  "
          f"psF1={ss['per_song_f1_mean']:.4f}±{ss['per_song_f1_std']:.4f}")
    print(f"ensemble  thr={ens_thr:.3f} acc={ens['acc']:.4f} P={ens['mel_p']:.4f} R={ens['mel_r']:.4f} "
          f"F1={ens['mel_f1']:.4f} AP={ens['ap']:.4f} psF1={ens['per_song']['mean']:.4f}")
    if "baseline" in report:
        b = report["baseline"]
        print(f"baseline  thr={b['thr']:.3f} acc={b['acc']:.4f} P={b['mel_p']:.4f} R={b['mel_r']:.4f} "
              f"F1={b['mel_f1']:.4f} AP={b['ap']:.4f} psF1={b['per_song']['mean']:.4f}")
    for name, cmp in report["comparisons"].items():
        print(f"{name}: delta={cmp['mean_delta']:+.4f} CI95=[{cmp['bootstrap_ci95'][0]:+.4f},"
              f"{cmp['bootstrap_ci95'][1]:+.4f}] t={cmp['paired_t']:.2f} p_t={cmp['p_ttest']:.3g} "
              f"p_wilcoxon={cmp['p_wilcoxon']:.3g} W/L={cmp['wins']}/{cmp['losses']} dz={cmp['cohen_dz']:.2f}")
    print(f"wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
