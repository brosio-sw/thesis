#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from scipy.stats import chi2

BASE_DIR = Path("data/steering_masked_activations")
ACTS_DIR = BASE_DIR / "activations"
SCORES_FILE = BASE_DIR / "scores" / "tail_all_sentiment.json"
OUT_ROOT = Path("data/activations_modelling/gaussian")

REPRESENTATIVE_LAYERS = [0, 8, 16, 24, 31]
DEFAULT_ALPHAS = [8.0, 18.0]
POS_THRESHOLD = 0.45
NEG_THRESHOLD = 0.55
PLOT_MAX_POINTS_PER_CLASS = 4000
PCA_FIT_MAX_POINTS = 12000


def discover_parts() -> list[dict]:
    parts = []
    for acts_file in sorted(ACTS_DIR.glob("tail_all_masked_part*.pt")):
        meta_file = ACTS_DIR / f"{acts_file.stem}_meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        parts.append(
            {
                "acts_file": acts_file,
                "meta_file": meta_file,
                "chunk_idx": int(meta.get("chunk_idx", -1)),
            }
        )
    parts.sort(key=lambda x: x["chunk_idx"])
    return parts


def iter_parts(parts: list[dict]) -> Iterable[tuple[dict, dict]]:
    for p in parts:
        acts = torch.load(p["acts_file"], map_location="cpu")
        meta = json.loads(p["meta_file"].read_text())
        yield acts, meta


def extract_negative_scores(meta: dict, per_text_scores: np.ndarray) -> np.ndarray:
    if "target_negative" in meta:
        return np.asarray(meta["target_negative"], dtype=np.float64)
    idx = np.asarray(meta["owner_text_idx"], dtype=np.int64)
    return per_text_scores[idx]


def split_pos_neg(y_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = y_neg <= POS_THRESHOLD
    neg = y_neg >= NEG_THRESHOLD
    return pos, neg


def class_half_split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    n_train = n // 2
    return x[:n_train], x[n_train:]


def fit_pca_basis(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n > PCA_FIT_MAX_POINTS:
        sel = rng.choice(n, size=PCA_FIT_MAX_POINTS, replace=False)
        x_fit = x[sel]
    else:
        x_fit = x

    xt = torch.tensor(x_fit, dtype=torch.float32)
    center = xt.mean(dim=0, keepdim=True)
    xc = xt - center
    _, _, v = torch.pca_lowrank(xc, q=2)
    w = v.T.cpu().numpy()  # [2, d]
    c = center.cpu().numpy().squeeze(0)
    return c, w


def project(x: np.ndarray, center: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x - center) @ w.T


def plot_ellipse(ax, mean2d: np.ndarray, cov2d: np.ndarray, color: str, label: str) -> None:
    vals, vecs = np.linalg.eigh(cov2d)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = math.sqrt(chi2.ppf(0.95, df=2))
    width = 2.0 * scale * math.sqrt(vals[0])
    height = 2.0 * scale * math.sqrt(vals[1])
    ell = Ellipse(xy=mean2d, width=width, height=height, angle=angle, edgecolor=color, facecolor="none", lw=1.8, label=label)
    ax.add_patch(ell)


def mean_distance_after_steer(x: np.ndarray, mu: np.ndarray, direction: np.ndarray, alpha: float, chunk: int = 2048) -> float:
    total = 0.0
    count = 0
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        diff = xb - mu + alpha * direction
        total += float(np.linalg.norm(diff, axis=1).sum())
        count += xb.shape[0]
    return total / max(1, count)


def sample_rows(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= k:
        return x
    idx = rng.choice(x.shape[0], size=k, replace=False)
    return x[idx]


def run_layer(layer: int, alphas: list[float], parts: list[dict], per_text_scores: np.ndarray, rng: np.random.Generator) -> dict:
    x_pos_parts: list[np.ndarray] = []
    x_neg_parts: list[np.ndarray] = []

    for acts, meta in iter_parts(parts):
        y = extract_negative_scores(meta, per_text_scores)
        pos_mask, neg_mask = split_pos_neg(y)
        x = acts[layer].detach().cpu().numpy().astype(np.float32, copy=False)
        if np.any(pos_mask):
            x_pos_parts.append(x[pos_mask].copy())
        if np.any(neg_mask):
            x_neg_parts.append(x[neg_mask].copy())
        del acts

    if not x_pos_parts or not x_neg_parts:
        raise RuntimeError(f"Layer {layer}: missing positive or negative activations after thresholding")

    x_pos = np.concatenate(x_pos_parts, axis=0)
    x_neg = np.concatenate(x_neg_parts, axis=0)

    x_pos_train, x_pos_test = class_half_split(x_pos)
    x_neg_train, x_neg_test = class_half_split(x_neg)

    mu_pos = x_pos_train.mean(axis=0, dtype=np.float64)
    mu_neg = x_neg_train.mean(axis=0, dtype=np.float64)
    mu_all = np.concatenate([x_pos_train, x_neg_train], axis=0).mean(axis=0, dtype=np.float64)

    direction = mu_pos - mu_neg
    direction_norm = float(np.linalg.norm(direction))

    x_test_all = np.concatenate([x_pos_test, x_neg_test], axis=0)

    center, w = fit_pca_basis(x_test_all, rng)
    z_pos = project(x_pos_test, center, w)
    z_neg = project(x_neg_test, center, w)

    mu_pos_2d = project(mu_pos[None, :], center, w)[0]
    mu_neg_2d = project(mu_neg[None, :], center, w)[0]
    mu_all_2d = project(mu_all[None, :], center, w)[0]
    dir_2d = direction @ w.T

    cov_pos_2d = np.cov(z_pos.T)
    cov_neg_2d = np.cov(z_neg.T)

    metrics = {
        "layer": layer,
        "direction_definition": "direction = mu_pos_train - mu_neg_train",
        "thresholds": {
            "positive_if_negative_score_le": POS_THRESHOLD,
            "negative_if_negative_score_ge": NEG_THRESHOLD,
            "neutral_ignored": True,
        },
        "counts": {
            "positive_total": int(x_pos.shape[0]),
            "negative_total": int(x_neg.shape[0]),
            "positive_train": int(x_pos_train.shape[0]),
            "positive_test": int(x_pos_test.shape[0]),
            "negative_train": int(x_neg_train.shape[0]),
            "negative_test": int(x_neg_test.shape[0]),
        },
        "direction_norm": direction_norm,
        "alphas": {},
    }

    for alpha in alphas:
        mean_disp = float(alpha * direction_norm)
        d_pos = mean_distance_after_steer(x_test_all, mu_pos, direction, alpha)
        d_neg = mean_distance_after_steer(x_test_all, mu_neg, direction, alpha)
        d_all = mean_distance_after_steer(x_test_all, mu_all, direction, alpha)
        metrics["alphas"][str(alpha)] = {
            "mean_displacement_norm": mean_disp,
            "mean_distance_to_mu_pos": d_pos,
            "mean_distance_to_mu_neg": d_neg,
            "mean_distance_to_mu_all": d_all,
        }

    layer_dir = OUT_ROOT / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    (layer_dir / f"diffmean_steering_metrics_layer_{layer:02d}.json").write_text(json.dumps(metrics, indent=2))

    plot_pos = sample_rows(z_pos, PLOT_MAX_POINTS_PER_CLASS, rng)
    plot_neg = sample_rows(z_neg, PLOT_MAX_POINTS_PER_CLASS, rng)

    alpha_colors = {
        alphas[0]: "#1f77b4",
        alphas[1]: "#ff7f0e" if len(alphas) > 1 else "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(plot_pos[:, 0], plot_pos[:, 1], s=8, alpha=0.28, c="#2ca02c", label="original positive test")
    ax.scatter(plot_neg[:, 0], plot_neg[:, 1], s=8, alpha=0.28, c="#d62728", label="original negative test")

    for alpha in alphas:
        color = alpha_colors[alpha]
        steer_shift = alpha * dir_2d
        plot_pos_s = plot_pos + steer_shift
        plot_neg_s = plot_neg + steer_shift
        ax.scatter(plot_pos_s[:, 0], plot_pos_s[:, 1], s=8, alpha=0.24, c=color, marker="x", label=f"steered +alpha={alpha} (from pos)")
        ax.scatter(plot_neg_s[:, 0], plot_neg_s[:, 1], s=8, alpha=0.24, c=color, marker="+", label=f"steered +alpha={alpha} (from neg)")

    plot_ellipse(ax, mu_pos_2d, cov_pos_2d, "#2ca02c", "pos test ellipse 95%")
    plot_ellipse(ax, mu_neg_2d, cov_neg_2d, "#d62728", "neg test ellipse 95%")

    ax.scatter([mu_all_2d[0]], [mu_all_2d[1]], c="black", s=70, marker="*", label="mu_all(train)")
    ax.annotate(
        "steer dir (mu_pos - mu_neg)",
        xy=(mu_all_2d[0] + dir_2d[0], mu_all_2d[1] + dir_2d[1]),
        xytext=(mu_all_2d[0], mu_all_2d[1]),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=9,
        color="black",
    )

    ax.set_title(f"Layer {layer}: DiffMean steering of test activations (alpha in {alphas})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    alpha_str = "_".join(str(int(a)) if float(a).is_integer() else str(a) for a in alphas)
    combined_path = layer_dir / f"diffmean_steering_scatter_layer_{layer:02d}_alpha_{alpha_str}.png"
    fig.savefig(combined_path, dpi=160)
    plt.close(fig)

    for alpha in alphas:
        fig, ax = plt.subplots(figsize=(10, 8))
        color = alpha_colors[alpha]
        steer_shift = alpha * dir_2d
        plot_pos_s = plot_pos + steer_shift
        plot_neg_s = plot_neg + steer_shift

        ax.scatter(plot_pos[:, 0], plot_pos[:, 1], s=8, alpha=0.28, c="#2ca02c", label="original positive test")
        ax.scatter(plot_neg[:, 0], plot_neg[:, 1], s=8, alpha=0.28, c="#d62728", label="original negative test")
        ax.scatter(plot_pos_s[:, 0], plot_pos_s[:, 1], s=8, alpha=0.25, c=color, marker="x", label=f"steered +alpha={alpha} (from pos)")
        ax.scatter(plot_neg_s[:, 0], plot_neg_s[:, 1], s=8, alpha=0.25, c=color, marker="+", label=f"steered +alpha={alpha} (from neg)")

        plot_ellipse(ax, mu_pos_2d, cov_pos_2d, "#2ca02c", "pos test ellipse 95%")
        plot_ellipse(ax, mu_neg_2d, cov_neg_2d, "#d62728", "neg test ellipse 95%")

        ax.scatter([mu_all_2d[0]], [mu_all_2d[1]], c="black", s=70, marker="*", label="mu_all(train)")
        ax.annotate(
            "steer dir",
            xy=(mu_all_2d[0] + dir_2d[0], mu_all_2d[1] + dir_2d[1]),
            xytext=(mu_all_2d[0], mu_all_2d[1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=9,
            color="black",
        )

        ax.set_title(f"Layer {layer}: DiffMean steering with alpha={alpha}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        per_alpha_path = layer_dir / f"diffmean_steering_scatter_layer_{layer:02d}_alpha_{alpha}.png"
        fig.savefig(per_alpha_path, dpi=160)
        plt.close(fig)

    return {
        "layer": layer,
        "metrics_file": str(layer_dir / f"diffmean_steering_metrics_layer_{layer:02d}.json"),
        "plot_file": str(combined_path),
    }


def parse_layers(arg: str) -> list[int]:
    if arg.strip().lower() == "representative":
        return REPRESENTATIVE_LAYERS
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffMean steering visualization on activation test sets")
    parser.add_argument("--layers", type=str, default="0", help="Comma-separated layer ids, or representative")
    parser.add_argument("--alphas", type=str, default="8,18", help="Comma-separated steering strengths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if len(alphas) < 1:
        raise ValueError("At least one alpha is required")

    scores = json.loads(SCORES_FILE.read_text())
    per_text_scores = np.asarray(scores["per_text"], dtype=np.float64)
    parts = discover_parts()
    rng = np.random.default_rng(args.seed)

    summary = {
        "direction_definition": "direction = mu_pos_train - mu_neg_train",
        "layers": layers,
        "alphas": alphas,
        "pos_threshold": POS_THRESHOLD,
        "neg_threshold": NEG_THRESHOLD,
        "outputs": [],
    }

    for layer in layers:
        print(f"[run] layer={layer} alphas={alphas}")
        out = run_layer(layer, alphas, parts, per_text_scores, rng)
        summary["outputs"].append(out)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_file = OUT_ROOT / "diffmean_steering_run_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"[done] summary -> {out_file}")


if __name__ == "__main__":
    main()
