from __future__ import annotations

"""
Analyze online gradient-vs-causal validation step records.

Inputs:
- A run directory containing debug files from
  gradient_filling/grad_causal_validation_instruct.py, typically:
  data/gradient_filling/compare_mean_steering_online_grad_causal_validation_instruct/
  <split>/a1.0__sched_fixed1__online_grad_causal_validation/

Outputs (under <run_dir>/corr):
- metrics_summary.json
- per_step_metrics.csv
- mean_spearman_by_score.png
- top1_match_rate_by_score.png
- mean_top3_overlap_by_score.png
- spearman_hist_grad_x_input_weighted_sum.png (optional)
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BASE_DIR = Path(
    "data/gradient_filling/compare_mean_steering_online_grad_causal_validation_instruct"
)
DEFAULT_RUN_NAME = "a1.0__sched_fixed1__online_grad_causal_validation"

SCORE_FIELDS = {
    "baseline_low_conf_score": "Baseline",
    "grad_norm_weighted_mean": "GradNorm",
    "grad_x_input_weighted_sum": "GradXInput",
}
CAUSAL_FIELD = "causal_delta"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _rankdata_average_ties(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank_value = ((i + j) / 2.0) + 1.0
        ranks[order[i : j + 1]] = rank_value
        i = j + 1

    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None

    xr = _rankdata_average_ties(x)
    yr = _rankdata_average_ties(y)

    if float(np.std(xr)) < 1e-12 or float(np.std(yr)) < 1e-12:
        return None

    corr = float(np.corrcoef(xr, yr)[0, 1])
    if math.isnan(corr) or math.isinf(corr):
        return None
    return corr


def _top_k_indices(values: list[float], k: int) -> list[int]:
    if not values:
        return []
    k_eff = min(k, len(values))
    return list(np.argsort(np.asarray(values, dtype=np.float64))[::-1][:k_eff])


def _top1_match(pred: list[float], target: list[float]) -> float | None:
    if not pred or not target or len(pred) != len(target):
        return None
    return 1.0 if int(np.argmax(pred)) == int(np.argmax(target)) else 0.0


def _topk_overlap(pred: list[float], target: list[float], k: int = 3) -> float | None:
    if not pred or not target or len(pred) != len(target):
        return None
    k_eff = min(k, len(pred))
    pred_set = set(_top_k_indices(pred, k_eff))
    target_set = set(_top_k_indices(target, k_eff))
    return float(len(pred_set.intersection(target_set)) / max(1, k_eff))


def _find_default_run_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "full_run" / DEFAULT_RUN_NAME,
        base_dir / "smoke_test" / DEFAULT_RUN_NAME,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return candidates[0]


def _load_step_records(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    debug_paths = sorted(run_dir.glob("debug_example_*.json"))
    steps: list[dict[str, Any]] = []
    source_files: list[str] = []

    for dp in debug_paths:
        with dp.open("r") as f:
            payload = json.load(f)
        run_steps = payload.get("steps", [])
        if not isinstance(run_steps, list):
            continue
        for step in run_steps:
            if isinstance(step, dict):
                steps.append(step)
                source_files.append(dp.name)

    return steps, source_files


def _analyze_steps(
    steps: list[dict[str, Any]], source_files: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_step_rows: list[dict[str, Any]] = []

    spearman_vals: dict[str, list[float]] = {k: [] for k in SCORE_FIELDS}
    top1_vals: dict[str, list[float]] = {k: [] for k in SCORE_FIELDS}
    top3_vals: dict[str, list[float]] = {k: [] for k in SCORE_FIELDS}

    analyzed_steps = 0

    for idx, step in enumerate(steps):
        candidates = step.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue

        vectors: dict[str, list[float]] = {k: [] for k in SCORE_FIELDS}
        causal: list[float] = []

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            c = _safe_float(cand.get(CAUSAL_FIELD))
            if c is None:
                continue

            current: dict[str, float] = {}
            valid = True
            for key in SCORE_FIELDS:
                v = _safe_float(cand.get(key))
                if v is None:
                    valid = False
                    break
                current[key] = v

            if not valid:
                continue

            causal.append(c)
            for key in SCORE_FIELDS:
                vectors[key].append(current[key])

        n = len(causal)
        if n == 0:
            continue

        analyzed_steps += 1

        row: dict[str, Any] = {
            "step_index": int(idx),
            "source_file": source_files[idx] if idx < len(source_files) else "",
            "global_step": step.get("global_step"),
            "block_idx": step.get("block_idx"),
            "batch_idx": step.get("batch_idx"),
            "chosen_commit_position": step.get("chosen_commit_position"),
            "n_candidates_used": n,
        }

        for key in SCORE_FIELDS:
            s = _spearman(vectors[key], causal)
            t1 = _top1_match(vectors[key], causal)
            t3 = _topk_overlap(vectors[key], causal, k=3)

            row[f"spearman__{key}"] = s
            row[f"top1_match__{key}"] = t1
            row[f"top3_overlap__{key}"] = t3

            if s is not None:
                spearman_vals[key].append(float(s))
            if t1 is not None:
                top1_vals[key].append(float(t1))
            if t3 is not None:
                top3_vals[key].append(float(t3))

        per_step_rows.append(row)

    def _mean_or_none(vals: list[float]) -> float | None:
        return None if not vals else float(np.mean(np.asarray(vals, dtype=np.float64)))

    def _median_or_none(vals: list[float]) -> float | None:
        return None if not vals else float(median(vals))

    summary: dict[str, Any] = {
        "n_steps_total_loaded": int(len(steps)),
        "n_steps_analyzed": int(analyzed_steps),
        "scores": {},
    }

    for key, label in SCORE_FIELDS.items():
        summary["scores"][key] = {
            "label": label,
            "mean_spearman": _mean_or_none(spearman_vals[key]),
            "median_spearman": _median_or_none(spearman_vals[key]),
            "top1_match_rate": _mean_or_none(top1_vals[key]),
            "mean_top3_overlap": _mean_or_none(top3_vals[key]),
            "n_spearman_steps": int(len(spearman_vals[key])),
            "n_top1_steps": int(len(top1_vals[key])),
            "n_top3_steps": int(len(top3_vals[key])),
        }

    return per_step_rows, summary


def _write_per_step_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "step_index",
        "source_file",
        "global_step",
        "block_idx",
        "batch_idx",
        "chosen_commit_position",
        "n_candidates_used",
        "spearman__baseline_low_conf_score",
        "spearman__grad_norm_weighted_mean",
        "spearman__grad_x_input_weighted_sum",
        "top1_match__baseline_low_conf_score",
        "top1_match__grad_norm_weighted_mean",
        "top1_match__grad_x_input_weighted_sum",
        "top3_overlap__baseline_low_conf_score",
        "top3_overlap__grad_norm_weighted_mean",
        "top3_overlap__grad_x_input_weighted_sum",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _plot_metric_bar(
    *,
    summary: dict[str, Any],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None,
) -> None:
    keys = list(SCORE_FIELDS.keys())
    labels = [SCORE_FIELDS[k] for k in keys]
    vals = [summary["scores"].get(k, {}).get(metric_key) for k in keys]

    y = [float(v) if v is not None else np.nan for v in vals]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_gradx_spearman_hist(rows: list[dict[str, Any]], out_path: Path) -> None:
    vals: list[float] = []
    for r in rows:
        v = r.get("spearman__grad_x_input_weighted_sum")
        if isinstance(v, (float, int)):
            vals.append(float(v))

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(vals, bins=20)
    ax.set_title("Per-step Spearman: GradXInput vs Causal Delta")
    ax.set_xlabel("Spearman")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_gradnorm_top3_by_global_step(rows: list[dict[str, Any]], out_path: Path) -> None:
    by_step: dict[int, list[float]] = {}

    for r in rows:
        step_raw = r.get("global_step")
        overlap_raw = r.get("top3_overlap__grad_norm_weighted_mean")
        if not isinstance(step_raw, (int, float)):
            continue
        if not isinstance(overlap_raw, (int, float)):
            continue

        step = int(step_raw)
        overlap = float(overlap_raw)
        by_step.setdefault(step, []).append(overlap)

    if not by_step:
        return

    steps_sorted = sorted(by_step.keys())
    means = [float(np.mean(np.asarray(by_step[s], dtype=np.float64))) for s in steps_sorted]

    # Keep readable width while preserving one bar per denoising step.
    fig_w = max(10.0, 0.25 * len(steps_sorted) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(steps_sorted, means, width=0.8)
    ax.set_title("GradNorm Top-3 Overlap by Denoising Step")
    ax.set_xlabel("Denoising Step (global_step)")
    ax.set_ylabel("Mean Top-3 Overlap")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for online validation outputs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run directory. If omitted, auto-uses full_run then smoke_test default run name.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="corr",
        help="Subfolder name inside run directory for analysis outputs.",
    )
    parser.add_argument(
        "--skip-hist",
        action="store_true",
        help="Skip histogram plot for per-step Spearman(grad_x_input_weighted_sum, causal_delta).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir if args.run_dir is not None else _find_default_run_dir(args.base_dir)
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}. Pass --run-dir explicitly or verify base outputs."
        )

    steps, source_files = _load_step_records(run_dir)
    if not steps:
        raise RuntimeError(
            f"No step records found in {run_dir}. Expected debug_example_*.json with a steps list."
        )

    per_step_rows, summary = _analyze_steps(steps, source_files)
    summary["run_dir"] = str(run_dir)

    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "per_step_metrics.csv"
    _write_per_step_csv(per_step_rows, csv_path)

    _plot_metric_bar(
        summary=summary,
        metric_key="mean_spearman",
        title="Mean Spearman vs Causal Delta",
        ylabel="Mean Spearman",
        out_path=out_dir / "mean_spearman_by_score.png",
        ylim=(-1.0, 1.0),
    )
    _plot_metric_bar(
        summary=summary,
        metric_key="top1_match_rate",
        title="Top-1 Match Rate vs Causal Delta",
        ylabel="Match Rate",
        out_path=out_dir / "top1_match_rate_by_score.png",
        ylim=(0.0, 1.0),
    )
    _plot_metric_bar(
        summary=summary,
        metric_key="mean_top3_overlap",
        title="Mean Top-3 Overlap vs Causal Delta",
        ylabel="Mean Overlap",
        out_path=out_dir / "mean_top3_overlap_by_score.png",
        ylim=(0.0, 1.0),
    )
    _plot_gradnorm_top3_by_global_step(
        rows=per_step_rows,
        out_path=out_dir / "gradnorm_mean_top3_overlap_by_denoising_step.png",
    )

    if not args.skip_hist:
        _plot_gradx_spearman_hist(
            rows=per_step_rows,
            out_path=out_dir / "spearman_hist_grad_x_input_weighted_sum.png",
        )

    print(f"[ok] run_dir: {run_dir}")
    print(f"[ok] analyzed steps: {summary['n_steps_analyzed']} / {summary['n_steps_total_loaded']}")
    print(f"[ok] wrote: {summary_path}")
    print(f"[ok] wrote: {csv_path}")
    print(f"[ok] wrote plots under: {out_dir}")


if __name__ == "__main__":
    main()
