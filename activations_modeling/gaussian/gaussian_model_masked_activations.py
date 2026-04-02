#!/usr/bin/env python3
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
TEXTS_FILE = BASE_DIR / "texts" / "tail_all.json"
OUT_DIR = Path("data/activations_modelling/gaussian")

POS_THRESHOLD = 0.45
NEG_THRESHOLD = 0.55
VAL_MODULO = 5
VAL_REMAINDER = 0
SHRINKAGE = 0.05
VAR_FLOOR = 1e-6
REPRESENTATIVE_LAYERS = [0, 8, 16, 24, 31]
MAX_PLOT_POINTS = 3000


@dataclass
class RunningDiagStats:
    n: int
    sum_x: np.ndarray
    sum_x2: np.ndarray

    @classmethod
    def init(cls, dim: int) -> "RunningDiagStats":
        return cls(n=0, sum_x=np.zeros(dim, dtype=np.float64), sum_x2=np.zeros(dim, dtype=np.float64))

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.n += x.shape[0]
        self.sum_x += x.sum(axis=0, dtype=np.float64)
        self.sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)


@dataclass
class DiagGaussian:
    mean: np.ndarray
    var: np.ndarray
    n_train: int
    cov_type: str


def discover_parts() -> List[dict]:
    parts = []
    for acts_file in sorted(ACTS_DIR.glob("tail_all_masked_part*.pt")):
        stem = acts_file.stem
        meta_file = ACTS_DIR / f"{stem}_meta.json"
        if not meta_file.exists():
            continue
        m = json.loads(meta_file.read_text())
        parts.append(
            {
                "acts_file": acts_file.name,
                "meta_file": meta_file.name,
                "chunk_idx": int(m.get("chunk_idx", -1)),
                "n_rows": int(m.get("n_rows", 0)),
            }
        )
    parts.sort(key=lambda x: x["chunk_idx"])
    return parts


def iter_parts(parts: List[dict]) -> Iterable[Tuple[dict, dict]]:
    for entry in parts:
        acts = torch.load(ACTS_DIR / entry["acts_file"], map_location="cpu")
        meta = json.loads((ACTS_DIR / entry["meta_file"]).read_text())
        yield acts, meta


def split_masks(neg_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = neg_scores <= POS_THRESHOLD
    neg = neg_scores >= NEG_THRESHOLD
    neu = (~pos) & (~neg)
    return pos, neg, neu


def extract_negative_scores(meta: dict, per_text_scores: np.ndarray) -> np.ndarray:
    if "target_negative" in meta:
        return np.asarray(meta["target_negative"], dtype=np.float64)
    if "owner_text_idx" in meta:
        idx = np.asarray(meta["owner_text_idx"], dtype=np.int64)
        return per_text_scores[idx]
    raise KeyError("No sentiment score available in meta (target_negative/owner_text_idx missing)")


def train_val_mask(n: int, offset: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64) + offset
    return (idx % VAL_MODULO) != VAL_REMAINDER


def finalize_diag_model(stats: RunningDiagStats) -> DiagGaussian:
    mean = stats.sum_x / max(stats.n, 1)
    var = stats.sum_x2 / max(stats.n, 1) - np.square(mean)
    var = np.maximum(var, VAR_FLOOR)
    avg_var = float(var.mean())
    var = (1.0 - SHRINKAGE) * var + SHRINKAGE * avg_var
    var = np.maximum(var, VAR_FLOOR)
    return DiagGaussian(mean=mean, var=var, n_train=stats.n, cov_type="diagonal_shrinkage")


def nll_diag(x: np.ndarray, model: DiagGaussian) -> np.ndarray:
    d = x.shape[1]
    diff = x - model.mean
    md2 = np.sum((diff * diff) / model.var, axis=1)
    logdet = np.sum(np.log(model.var))
    return 0.5 * (d * np.log(2.0 * np.pi) + logdet + md2)


def mahalanobis2_diag(x: np.ndarray, model: DiagGaussian) -> np.ndarray:
    diff = x - model.mean
    return np.sum((diff * diff) / model.var, axis=1)


def reservoir_update(rng: np.random.Generator, reservoir: List[np.ndarray], labels: List[int], batch: np.ndarray, batch_labels: np.ndarray, cap: int, seen: int) -> int:
    for i in range(batch.shape[0]):
        seen += 1
        if len(reservoir) < cap:
            reservoir.append(batch[i].copy())
            labels.append(int(batch_labels[i]))
        else:
            j = rng.integers(0, seen)
            if j < cap:
                reservoir[j] = batch[i].copy()
                labels[j] = int(batch_labels[i])
    return seen


def plot_cov_ellipse(ax, mean2d: np.ndarray, cov2d: np.ndarray, color: str, label: str) -> None:
    vals, vecs = np.linalg.eigh(cov2d)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = math.sqrt(chi2.ppf(0.95, df=2))
    width = 2.0 * scale * math.sqrt(vals[0])
    height = 2.0 * scale * math.sqrt(vals[1])
    ell = Ellipse(xy=mean2d, width=width, height=height, angle=angle, edgecolor=color, facecolor="none", lw=2, label=label)
    ax.add_patch(ell)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUT_DIR)
    manifest = json.loads((ACTS_DIR / "manifest.json").read_text())
    parts = discover_parts()
    scores = json.loads(SCORES_FILE.read_text())
    texts = json.loads(TEXTS_FILE.read_text())
    per_text_scores = np.asarray(scores["per_text"], dtype=np.float64)

    first_obj = torch.load(ACTS_DIR / parts[0]["acts_file"], map_location="cpu")
    n_layers = len(first_obj)
    hidden_dim = int(first_obj[0].shape[1])
    del first_obj
    total_rows = int(sum(p["n_rows"] for p in parts))

    phase1 = {
        "verified_path": str(ACTS_DIR),
        "files_total": len(list(ACTS_DIR.iterdir())),
        "pt_files": len(list(ACTS_DIR.glob("*.pt"))),
        "json_files": len(list(ACTS_DIR.glob("*.json"))),
        "n_parts": len(parts),
        "manifest_n_parts": len(manifest.get("files", [])),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "total_rows": total_rows,
        "layer_association": {
            "exists": True,
            "storage": "inside each .pt part as dict keys 0..31",
            "directory_by_layer": False,
        },
        "meta_fields": [
            "chunk_idx",
            "text_start_idx",
            "text_end_idx_exclusive",
            "n_input_texts",
            "n_rows",
            "owner_text_idx",
            "mask_ratio",
            "target_negative",
            "reconstruction",
            "source_text",
        ],
        "score_fields": list(scores.keys()),
        "text_fields": list(texts.keys()),
        "memory_estimate": {
            "activation_bytes": int(total_rows * n_layers * hidden_dim * 4),
            "activation_gib": float(total_rows * n_layers * hidden_dim * 4 / (1024 ** 3)),
            "fits_memory_recommendation": "process_incrementally",
        },
    }
    (OUT_DIR / "phase1_data_layout_summary.json").write_text(json.dumps(phase1, indent=2))

    layer_reports = []
    model_store: Dict[int, Dict[str, DiagGaussian]] = {}
    rng = np.random.default_rng(0)

    pooled_stats = {layer: RunningDiagStats.init(hidden_dim) for layer in range(n_layers)}
    pos_stats = {layer: RunningDiagStats.init(hidden_dim) for layer in range(n_layers)}
    neg_stats = {layer: RunningDiagStats.init(hidden_dim) for layer in range(n_layers)}
    counts = {
        layer: {"total": 0, "pos": 0, "neg": 0, "neu": 0}
        for layer in range(n_layers)
    }
    sample_points = {layer: [] for layer in REPRESENTATIVE_LAYERS}
    sample_labels = {layer: [] for layer in REPRESENTATIVE_LAYERS}
    sample_seen = {layer: 0 for layer in REPRESENTATIVE_LAYERS}

    offset = 0
    for acts, meta in iter_parts(parts):
        y = extract_negative_scores(meta, per_text_scores)
        n = len(y)
        pos_mask, neg_mask, neu_mask = split_masks(y)
        tv = train_val_mask(n, offset)
        lab = np.where(pos_mask, 1, np.where(neg_mask, -1, 0))

        for layer in range(n_layers):
            x = acts[layer].detach().cpu().numpy().astype(np.float64, copy=False)
            assert x.shape[0] == n, f"row mismatch on layer {layer}"
            pooled_stats[layer].update(x[tv])
            pos_stats[layer].update(x[tv & pos_mask])
            neg_stats[layer].update(x[tv & neg_mask])

            counts[layer]["total"] += n
            counts[layer]["pos"] += int(pos_mask.sum())
            counts[layer]["neg"] += int(neg_mask.sum())
            counts[layer]["neu"] += int(neu_mask.sum())

            if layer in REPRESENTATIVE_LAYERS:
                sample_seen[layer] = reservoir_update(
                    rng,
                    sample_points[layer],
                    sample_labels[layer],
                    x,
                    lab,
                    MAX_PLOT_POINTS,
                    sample_seen[layer],
                )

        offset += n
        del acts

    for layer in range(n_layers):
        model_store[layer] = {
            "pooled": finalize_diag_model(pooled_stats[layer]),
            "positive": finalize_diag_model(pos_stats[layer]),
            "negative": finalize_diag_model(neg_stats[layer]),
        }

    nll_sum = {
        layer: {"pooled": 0.0, "positive": 0.0, "negative": 0.0}
        for layer in range(n_layers)
    }
    nll_n = {
        layer: {"pooled": 0, "positive": 0, "negative": 0}
        for layer in range(n_layers)
    }

    offset = 0
    for acts, meta in iter_parts(parts):
        y = extract_negative_scores(meta, per_text_scores)
        n = len(y)
        pos_mask, neg_mask, _ = split_masks(y)
        val = ~train_val_mask(n, offset)

        y_pos = pos_mask[val]
        y_neg = neg_mask[val]

        for layer in range(n_layers):
            x = acts[layer].detach().cpu().numpy().astype(np.float64, copy=False)
            xv = x[val]
            pooled_model = model_store[layer]["pooled"]
            pos_model = model_store[layer]["positive"]
            neg_model = model_store[layer]["negative"]

            if xv.shape[0] > 0:
                nll = nll_diag(xv, pooled_model)
                nll_sum[layer]["pooled"] += float(nll.sum())
                nll_n[layer]["pooled"] += int(xv.shape[0])

            if np.any(y_pos):
                nll = nll_diag(xv[y_pos], pos_model)
                nll_sum[layer]["positive"] += float(nll.sum())
                nll_n[layer]["positive"] += int(y_pos.sum())

            if np.any(y_neg):
                nll = nll_diag(xv[y_neg], neg_model)
                nll_sum[layer]["negative"] += float(nll.sum())
                nll_n[layer]["negative"] += int(y_neg.sum())

        offset += n
        del acts

    for layer in range(n_layers):
        pooled_model = model_store[layer]["pooled"]
        pos_model = model_store[layer]["positive"]
        neg_model = model_store[layer]["negative"]
        avg_nll = {
            k: (nll_sum[layer][k] / nll_n[layer][k] if nll_n[layer][k] > 0 else float("nan"))
            for k in nll_sum[layer]
        }
        mix_nll_split = (
            (nll_sum[layer]["positive"] + nll_sum[layer]["negative"])
            / (nll_n[layer]["positive"] + nll_n[layer]["negative"])
            if (nll_n[layer]["positive"] + nll_n[layer]["negative"]) > 0
            else float("nan")
        )

        rep = {
            "layer": layer,
            "sample_count": counts[layer]["total"],
            "activation_dim": hidden_dim,
            "full_covariance_feasible": False,
            "covariance_choice": "diagonal_shrinkage",
            "pooled": {
                "train_count": pooled_model.n_train,
                "val_count": nll_n[layer]["pooled"],
                "avg_val_nll": avg_nll["pooled"],
                "condition_diag": float(pooled_model.var.max() / pooled_model.var.min()),
            },
            "split": {
                "pos_threshold_on_negative_score": POS_THRESHOLD,
                "neg_threshold_on_negative_score": NEG_THRESHOLD,
                "positive_count": counts[layer]["pos"],
                "negative_count": counts[layer]["neg"],
                "neutral_count": counts[layer]["neu"],
                "positive_train_count": pos_model.n_train,
                "negative_train_count": neg_model.n_train,
                "positive_val_count": nll_n[layer]["positive"],
                "negative_val_count": nll_n[layer]["negative"],
                "avg_val_nll_positive": avg_nll["positive"],
                "avg_val_nll_negative": avg_nll["negative"],
                "avg_val_nll_mixture": mix_nll_split,
            },
            "delta_nll_split_minus_pooled": float(mix_nll_split - avg_nll["pooled"]),
        }
        layer_reports.append(rep)

        if layer in REPRESENTATIVE_LAYERS and len(sample_points[layer]) >= 20:
            layer_dir = OUT_DIR / f"layer_{layer:02d}"
            ensure_dir(layer_dir)

            Xs = np.stack(sample_points[layer], axis=0)
            Ls = np.asarray(sample_labels[layer], dtype=np.int64)
            X = torch.tensor(Xs, dtype=torch.float32)
            center = X.mean(dim=0, keepdim=True)
            Xc = X - center
            _, _, V = torch.pca_lowrank(Xc, q=2)
            W = V.T.cpu().numpy()

            Z = (Xs - center.numpy()) @ W.T

            pooled = pooled_model
            posm = pos_model
            negm = neg_model

            def proj_model(m: DiagGaussian):
                mu2 = (m.mean - center.numpy().squeeze(0)) @ W.T
                cov2 = (W * m.var[np.newaxis, :]) @ W.T
                return mu2, cov2

            mu2_p, cov2_p = proj_model(pooled)
            mu2_pos, cov2_pos = proj_model(posm)
            mu2_neg, cov2_neg = proj_model(negm)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(Z[Ls == 1, 0], Z[Ls == 1, 1], s=8, alpha=0.35, label="positive", color="#2ca02c")
            ax.scatter(Z[Ls == -1, 0], Z[Ls == -1, 1], s=8, alpha=0.35, label="negative", color="#d62728")
            if np.any(Ls == 0):
                ax.scatter(Z[Ls == 0, 0], Z[Ls == 0, 1], s=8, alpha=0.35, label="neutral", color="#7f7f7f")
            plot_cov_ellipse(ax, mu2_p, cov2_p, "#1f77b4", "pooled 95%")
            plot_cov_ellipse(ax, mu2_pos, cov2_pos, "#2ca02c", "positive 95%")
            plot_cov_ellipse(ax, mu2_neg, cov2_neg, "#d62728", "negative 95%")
            ax.set_title(f"Layer {layer}: PCA scatter with Gaussian ellipses")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(layer_dir / "pca_scatter_ellipses.png", dpi=150)
            plt.close(fig)

            md2_pool = mahalanobis2_diag(Xs, pooled)
            md2_pos = mahalanobis2_diag(Xs[Ls == 1], posm) if np.any(Ls == 1) else np.array([])
            md2_neg = mahalanobis2_diag(Xs[Ls == -1], negm) if np.any(Ls == -1) else np.array([])

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(md2_pool, bins=60, alpha=0.5, label="pooled/on all", color="#1f77b4", density=True)
            if md2_pos.size > 0:
                ax.hist(md2_pos, bins=60, alpha=0.4, label="positive/on positive", color="#2ca02c", density=True)
            if md2_neg.size > 0:
                ax.hist(md2_neg, bins=60, alpha=0.4, label="negative/on negative", color="#d62728", density=True)
            ax.set_title(f"Layer {layer}: Mahalanobis distance^2 histograms")
            ax.set_xlabel("Squared Mahalanobis distance")
            ax.set_ylabel("Density")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(layer_dir / "mahalanobis_hist.png", dpi=150)
            plt.close(fig)

            q = np.linspace(0.01, 0.99, 99)
            emp = np.quantile(md2_pool, q)
            theo = chi2.ppf(q, df=hidden_dim)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(theo, emp, marker="o", linestyle="", ms=3, alpha=0.8)
            lo = min(theo.min(), emp.min())
            hi = max(theo.max(), emp.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_title(f"Layer {layer}: QQ plot of squared Mahalanobis (pooled)")
            ax.set_xlabel("Chi-square quantiles")
            ax.set_ylabel("Empirical quantiles")
            fig.tight_layout()
            fig.savefig(layer_dir / "qq_md2_vs_chi2.png", dpi=150)
            plt.close(fig)

            ev = np.sort(pooled.var)[::-1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(np.arange(1, ev.size + 1), ev, lw=1.2)
            ax.set_yscale("log")
            ax.set_title(f"Layer {layer}: covariance eigenvalue spectrum (diag approx)")
            ax.set_xlabel("Eigenvalue rank")
            ax.set_ylabel("Eigenvalue (log scale)")
            fig.tight_layout()
            fig.savefig(layer_dir / "cov_eigen_spectrum.png", dpi=150)
            plt.close(fig)

    overall = {
        "rule": {
            "score_used": "meta.target_negative if present else scores.per_text[owner_text_idx]",
            "positive_if": f"target_negative <= {POS_THRESHOLD}",
            "negative_if": f"target_negative >= {NEG_THRESHOLD}",
            "neutral_if": f"{POS_THRESHOLD} < target_negative < {NEG_THRESHOLD}",
            "val_split": f"deterministic row index mod {VAL_MODULO} == {VAL_REMAINDER}",
        },
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "total_rows": total_rows,
        "representative_layers_for_plots": REPRESENTATIVE_LAYERS,
        "layers": layer_reports,
    }
    (OUT_DIR / "gaussian_layer_report.json").write_text(json.dumps(overall, indent=2))

    pooled_nll = np.array([r["pooled"]["avg_val_nll"] for r in layer_reports], dtype=np.float64)
    split_nll = np.array([r["split"]["avg_val_nll_mixture"] for r in layer_reports], dtype=np.float64)
    delta = split_nll - pooled_nll
    summary = {
        "mean_pooled_val_nll": float(np.nanmean(pooled_nll)),
        "mean_split_val_nll": float(np.nanmean(split_nll)),
        "mean_delta_split_minus_pooled": float(np.nanmean(delta)),
        "n_layers_split_better": int(np.sum(delta < 0)),
        "n_layers_pooled_better": int(np.sum(delta > 0)),
        "n_layers_tie": int(np.sum(np.isclose(delta, 0.0))),
    }
    (OUT_DIR / "gaussian_comparison_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({"phase1_file": str(OUT_DIR / "phase1_data_layout_summary.json"), "report_file": str(OUT_DIR / "gaussian_layer_report.json"), "summary_file": str(OUT_DIR / "gaussian_comparison_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
