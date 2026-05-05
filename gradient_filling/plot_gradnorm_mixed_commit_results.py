from __future__ import annotations

import argparse
import json
import re
import sys
from os import getenv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.fluency_metrics import compute_perplexity
from eval.sentiment_metrics import compute_sentiment_metrics


# Hardcoded for now; change here later if needed.
TARGET_ALPHA = 1.0
SENTIMENT_DEVICE = getenv("SENTIMENT_DEVICE", "cpu")
PERPLEXITY_DEVICE = "cpu"


def _extract_lambda_from_name(name: str) -> float | None:
    m = re.search(r"(?:mixlam|causalmix)_([0-9]+(?:\.[0-9]+)?)$", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_alpha_from_name(name: str) -> float | None:
    m = re.search(r"^a([0-9]+(?:\.[0-9]+)?)__", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _load_valid_answers(run_dir: Path) -> list[str]:
    out: list[str] = []
    generations_path = run_dir / "generations.jsonl"
    if not generations_path.exists():
        return out
    with generations_path.open("r") as f:
        for line in f:
            obj = json.loads(line)
            if bool(obj.get("is_valid", False)):
                out.append(str(obj.get("answer", "")))
    return out


def _percentile_err(values: list[float], mean: float) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    p15 = float(np.percentile(arr, 15))
    p85 = float(np.percentile(arr, 85))
    return max(0.0, mean - p15), max(0.0, p85 - mean)


def _load_cache(cache_path: Path) -> dict[str, dict[str, list[float]]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def _save_cache(cache_path: Path, cache: dict[str, dict[str, list[float]]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(cache, f, indent=2)


def _collect_rows(
    full_run_dir: Path,
    target_alpha: float,
    *,
    cache_path: Path,
    recompute_metrics: bool,
) -> list[dict]:
    if not full_run_dir.exists():
        raise FileNotFoundError(f"Could not find full_run dir: {full_run_dir}")

    cache = _load_cache(cache_path)
    cache_changed = False
    rows: list[dict] = []
    for child in sorted(full_run_dir.iterdir()):
        if not child.is_dir():
            continue

        run_name = child.name
        alpha = _extract_alpha_from_name(run_name)
        lam = _extract_lambda_from_name(run_name)
        if alpha is None or lam is None:
            continue
        if abs(float(alpha) - float(target_alpha)) > 1e-9:
            continue

        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue

        with metrics_path.open("r") as f:
            metrics = json.load(f)

        invalid_fraction = metrics.get("invalid_fraction", None)
        ppl_valid = metrics.get("ppl_answer_mean", None)
        sent_valid = metrics.get("sent_answer_mean", None)
        if invalid_fraction is None or ppl_valid is None or sent_valid is None:
            continue

        ppl_mean = float(ppl_valid)
        sent_mean = float(sent_valid)

        ppl_err_low = 0.0
        ppl_err_high = 0.0
        sent_err_low = 0.0
        sent_err_high = 0.0

        cache_key = run_name
        cached = cache.get(cache_key, {})
        sent_scores: list[float] = []
        ppl_scores: list[float] = []

        if not recompute_metrics:
            sent_cached = cached.get("sent_scores", []) if isinstance(cached, dict) else []
            ppl_cached = cached.get("ppl_scores", []) if isinstance(cached, dict) else []
            if isinstance(sent_cached, list) and isinstance(ppl_cached, list):
                sent_scores = [float(x) for x in sent_cached]
                ppl_scores = [float(x) for x in ppl_cached]

        if not sent_scores or not ppl_scores:
            valid_answers = _load_valid_answers(child)
            if valid_answers:
                sent = compute_sentiment_metrics(valid_answers, device=SENTIMENT_DEVICE)
                ppl = compute_perplexity(valid_answers, device=PERPLEXITY_DEVICE)
                sent_scores = [float(x) for x in sent["scores"].tolist()]
                ppl_scores = [float(x) for x in ppl["per_text_ppl"]]
                cache[cache_key] = {
                    "sent_scores": sent_scores,
                    "ppl_scores": ppl_scores,
                }
                cache_changed = True

        if sent_scores and ppl_scores:
            sent_err_low, sent_err_high = _percentile_err(sent_scores, sent_mean)
            ppl_err_low, ppl_err_high = _percentile_err(ppl_scores, ppl_mean)

        rows.append(
            {
                "run_name": run_name,
                "lambda_mix": float(lam),
                "invalid_percent": float(invalid_fraction) * 100.0,
                "ppl_answer_mean": ppl_mean,
                "sent_answer_mean": sent_mean,
                "ppl_err_low": ppl_err_low,
                "ppl_err_high": ppl_err_high,
                "sent_err_low": sent_err_low,
                "sent_err_high": sent_err_high,
            }
        )

    if not rows:
        raise ValueError(
            f"No valid run folders found under {full_run_dir} for alpha={target_alpha}"
        )

    if cache_changed:
        _save_cache(cache_path, cache)

    rows.sort(key=lambda x: x["lambda_mix"])
    return rows


def plot_results(
    full_run_dir: Path,
    output_path: Path,
    title_prefix: str,
    alpha: float,
    *,
    cache_path: Path,
    recompute_metrics: bool,
) -> None:
    rows = _collect_rows(
        full_run_dir=full_run_dir,
        target_alpha=alpha,
        cache_path=cache_path,
        recompute_metrics=recompute_metrics,
    )

    x_labels = [f"{r['lambda_mix']:.2f}" for r in rows]
    invalid_vals = [r["invalid_percent"] for r in rows]
    ppl_vals = [r["ppl_answer_mean"] for r in rows]
    sent_vals = [r["sent_answer_mean"] for r in rows]
    ppl_err_low = [r["ppl_err_low"] for r in rows]
    ppl_err_high = [r["ppl_err_high"] for r in rows]
    sent_err_low = [r["sent_err_low"] for r in rows]
    sent_err_high = [r["sent_err_high"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(x_labels, invalid_vals)
    axes[0].set_title("Invalid Percent vs Lambda Mix", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Lambda Mix", fontsize=11)
    axes[0].set_ylabel("Invalid %", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    axes[1].bar(x_labels, ppl_vals, yerr=[ppl_err_low, ppl_err_high], capsize=3)
    axes[1].set_title("PPL (Valid) vs Lambda Mix", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lambda Mix", fontsize=11)
    axes[1].set_ylabel("Perplexity (valid)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    axes[2].bar(x_labels, sent_vals, yerr=[sent_err_low, sent_err_high], capsize=3)
    axes[2].set_title("Sentiment (Valid) vs Lambda Mix", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Lambda Mix", fontsize=11)
    axes[2].set_ylabel("Mean P(NEGATIVE) valid", fontsize=11)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle(
        f"{title_prefix} Mixed Commit Results (alpha={alpha:g})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[ok] plot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot invalid%, valid PPL, and valid sentiment vs lambda mix from full_run folders."
    )
    parser.add_argument(
        "--commit-type",
        type=str,
        choices=["gradnorm", "causal"],
        default="gradnorm",
        help="Which mixed-commit experiment to plot.",
    )
    parser.add_argument(
        "--full-run-dir",
        type=Path,
        default=None,
        help="Optional full_run directory path. If omitted, defaults depend on --commit-type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, defaults depend on --commit-type.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=TARGET_ALPHA,
        help="Alpha to plot. Defaults to the hardcoded TARGET_ALPHA.",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Force recomputing per-run sentiment/PPL distributions instead of using cache.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Optional cache path for per-run metric distributions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.commit_type == "gradnorm":
        default_full_run_dir = Path(
            "data/gradient_filling/compare_mean_steering_online_gradnorm_mixed_commit_instruct/full_run"
        )
        default_output = Path(
            "data/gradient_filling/compare_mean_steering_online_gradnorm_mixed_commit_instruct/full_run/lambda_mix_results_3panels.png"
        )
        title_prefix = "GradNorm"
    else:
        default_full_run_dir = Path(
            "data/gradient_filling/compare_mean_steering_online_causal_mixed_commit_instruct/full_run"
        )
        default_output = Path(
            "data/gradient_filling/compare_mean_steering_online_causal_mixed_commit_instruct/full_run/lambda_mix_results_3panels.png"
        )
        title_prefix = "Causal"

    full_run_dir = args.full_run_dir if args.full_run_dir is not None else default_full_run_dir
    output = args.output if args.output is not None else default_output
    cache_path = (
        args.cache_path
        if args.cache_path is not None
        else full_run_dir / "plot_metric_distributions_cache.json"
    )
    plot_results(
        full_run_dir,
        output,
        title_prefix,
        alpha=float(args.alpha),
        cache_path=cache_path,
        recompute_metrics=bool(args.recompute_metrics),
    )
