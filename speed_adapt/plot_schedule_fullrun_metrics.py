from __future__ import annotations

"""
Plot per-alpha schedule metrics from compare_mean_steering_adaptive_schedule_instruct full_run output.

Creates one PNG per alpha with 4 bar plots (mean + percentile error bars):
- avg denoising steps
- mean sentiment (P(NEGATIVE), answer-only)
- valid fraction
- mean perplexity (answer-only)

Error bars are asymmetric, using the 15th and 85th percentiles.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from eval.fluency_metrics import compute_perplexity
from eval.sentiment_metrics import compute_sentiment_metrics


DEFAULT_RECOMPUTE_CACHE_PATH = Path(
	"data/speed_adapt/compare_mean_steering_adaptive_schedule_instruct/full_run/clean_plot_metric_distributions.json"
)


def _to_float(value: str | float | int | None) -> float | None:
	if value is None:
		return None
	text = str(value).strip()
	if text == "" or text.lower() == "none":
		return None
	return float(text)


def _to_alpha_key(value: float) -> str:
	return f"{value:.1f}"


def _fmt_alpha_for_filename(alpha_key: str) -> str:
	return alpha_key.replace(".", "p")


def _parse_float_list(value: str) -> list[float]:
	text = str(value).strip()
	if not text:
		return []
	if not (text.startswith("[") and text.endswith("]")):
		return []
	try:
		arr = json.loads(text)
	except Exception:
		return []
	if not isinstance(arr, list):
		return []
	out: list[float] = []
	for x in arr:
		try:
			out.append(float(x))
		except Exception:
			continue
	return out


def _build_run_label(row: dict[str, str]) -> str:
	forced_label = str(row.get("plot_label_override", "")).strip()
	if forced_label:
		return forced_label

	schedule_mode = row.get("schedule_mode", "unknown")
	q_mode = row.get("q_conf_mode", "")
	score_src = row.get("schedule_score_source", "")
	mix_lambda = _to_float(row.get("mix_lambda", ""))

	label = schedule_mode
	if schedule_mode == "mix_adaptive" and q_mode and q_mode != "na":
		q_mid = _to_float(row.get("q_mid", ""))
		q_high = _to_float(row.get("q_high", ""))
		if q_mid is not None and q_high is not None:
			label = f"{schedule_mode}\n{q_mode}\nmid={q_mid:g} hi={q_high:g}"
		else:
			label = f"{schedule_mode}\n{q_mode}"
	elif schedule_mode == "calibrated_mix" and q_mode and q_mode != "na":
		if mix_lambda is None:
			label = f"{schedule_mode}\n{q_mode}"
		else:
			label = f"{schedule_mode}\n{q_mode}\nlambda={mix_lambda:g}"
	elif schedule_mode == "adaptive_123" and score_src:
		label = f"{schedule_mode}\n{score_src}"
	return label


def _is_unfinished_alpha4_mean_all_current_threshold(row: dict[str, str]) -> bool:
	alpha = _to_float(row.get("alpha", ""))
	q_mid = _to_float(row.get("q_mid", ""))
	q_high = _to_float(row.get("q_high", ""))
	if alpha is None or q_mid is None or q_high is None:
		return False
	return (
		abs(alpha - 4.0) < 1e-9
		and row.get("schedule_mode", "") == "mix_adaptive"
		and row.get("q_conf_mode", "") == "mean_all_block"
		and abs(q_mid - 0.22) < 1e-9
		and abs(q_high - 0.41) < 1e-9
	)


def _load_rows(summary_csv: Path) -> list[dict[str, str]]:
	if not summary_csv.exists():
		raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

	with summary_csv.open("r", newline="") as f:
		reader = csv.DictReader(f)
		return [dict(r) for r in reader]


def _normalize_row_types(row: dict[str, object]) -> dict[str, str]:
	out: dict[str, str] = {}
	for k, v in row.items():
		if v is None:
			out[k] = ""
		else:
			out[k] = str(v)
	return out


def _inject_baseline_row_for_clean_plot(
	*,
	alpha_key: str,
	alpha_rows: list[dict[str, str]],
	all_rows: list[dict[str, str]],
	clean_plot: bool,
) -> list[dict[str, str]]:
	"""
	For clean plots at alpha=1 or alpha=2, include one extra bar for the
	alpha=0, fixed_1 baseline run.
	"""
	if not clean_plot:
		return alpha_rows
	if alpha_key not in {"1.0", "2.0"}:
		return alpha_rows

	for row in alpha_rows:
		if str(row.get("plot_label_override", "")).strip() == "baseline":
			return alpha_rows

	baseline_candidates = [
		r
		for r in all_rows
		if abs(_to_float(r.get("alpha", "nan")) - 0.0) < 1e-9
		and str(r.get("schedule_mode", "")).strip() == "fixed_1"
	]
	if not baseline_candidates:
		return alpha_rows

	baseline_row = dict(baseline_candidates[0])
	baseline_row["plot_label_override"] = "baseline"
	return [baseline_row] + list(alpha_rows)


def _load_rows_from_run_dirs(full_run_dir: Path) -> list[dict[str, str]]:
	allowed_schedule_modes = {
		"fixed_1",
		"fixed_2",
		"fixed_3",
		"adaptive_123",
		"mix_adaptive",
		"calibrated_mix",
	}
	allowed_mix_q_modes = {"mean_top3", "min_top3"}
	rows: list[dict[str, str]] = []
	for metrics_path in sorted(full_run_dir.rglob("metrics.json")):
		child = metrics_path.parent

		with metrics_path.open("r") as f:
			metrics = json.load(f)

		run_info_path = child / "run_info.json"
		run_info: dict[str, object] = {}
		if run_info_path.exists():
			with run_info_path.open("r") as f:
				run_info = json.load(f)

		merged = dict(metrics)
		for key in ["schedule_score_source", "q_conf_mode", "q_mid", "q_high", "method", "alpha", "schedule_mode"]:
			if key not in merged and key in run_info:
				merged[key] = run_info[key]

		if "schedule_score_source" not in merged:
			merged["schedule_score_source"] = ""
		if "q_conf_mode" not in merged:
			merged["q_conf_mode"] = "na"

		schedule_mode = str(merged.get("schedule_mode", "")).strip()
		if schedule_mode not in allowed_schedule_modes:
			continue

		q_mode = str(merged.get("q_conf_mode", "")).strip()
		if schedule_mode in {"mix_adaptive", "calibrated_mix"} and q_mode not in allowed_mix_q_modes:
			continue

		merged["run_rel"] = str(child.relative_to(full_run_dir))

		rows.append(_normalize_row_types(merged))

	return rows


def _keep_only_clean_plot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
	fixed_modes = {"fixed_1", "fixed_2", "fixed_3"}
	out: list[dict[str, str]] = []
	for row in rows:
		mode = str(row.get("schedule_mode", "")).strip()
		if mode in fixed_modes:
			out.append(row)
			continue
		if mode == "calibrated_mix":
			lam = _to_float(row.get("mix_lambda", ""))
			if lam is not None:
				out.append(row)
	return out


def _load_valid_answers(run_dir: Path, max_valid: int | None) -> list[str]:
	answers: list[str] = []
	generations_path = run_dir / "generations.jsonl"
	if not generations_path.exists():
		return answers

	with generations_path.open("r") as f:
		for line in f:
			obj = json.loads(line)
			if not bool(obj.get("is_valid", False)):
				continue
			answers.append(str(obj.get("answer", "")))
			if max_valid is not None and len(answers) >= max_valid:
				break
	return answers


def _recompute_clean_plot_distributions(
	*,
	rows: list[dict[str, str]],
	full_run_dir: Path,
	cache_path: Path,
	sentiment_device: str,
	perplexity_device: str,
	max_valid_per_run: int | None,
) -> dict[str, dict[str, Any]]:
	by_run: dict[str, dict[str, Any]] = _load_recomputed_cache(cache_path)

	unique_rows: dict[str, dict[str, str]] = {}
	for row in rows:
		run_rel = str(row.get("run_rel", "")).strip()
		if run_rel:
			unique_rows[run_rel] = row

	n_skipped_cached = 0
	n_computed = 0

	for run_rel, row in sorted(unique_rows.items()):
		cached = by_run.get(run_rel)
		if isinstance(cached, dict):
			sent_vals = cached.get("sent_scores", [])
			ppl_vals = cached.get("ppl_scores", [])
			if isinstance(sent_vals, list) and isinstance(ppl_vals, list) and sent_vals and ppl_vals:
				n_skipped_cached += 1
				continue

		run_dir = full_run_dir / run_rel
		answers = _load_valid_answers(run_dir, max_valid=max_valid_per_run)
		if not answers:
			by_run[run_rel] = {
				"run_rel": run_rel,
				"n_valid_used": 0,
				"sent_scores": [],
				"ppl_scores": [],
				"sent_mean_recomputed": None,
				"ppl_mean_recomputed": None,
				"sent_mean_original": _to_float(row.get("sent_answer_mean", "")),
				"ppl_mean_original": _to_float(row.get("ppl_answer_mean", "")),
				"sent_abs_diff": None,
				"ppl_abs_diff": None,
			}
			print(f"[warn] no valid answers for {run_rel}")
			n_computed += 1
			continue

		sent = compute_sentiment_metrics(answers, device=sentiment_device)
		ppl = compute_perplexity(answers, device=perplexity_device)

		sent_scores = [float(x) for x in sent["scores"].tolist()]
		ppl_scores = [float(x) for x in ppl["per_text_ppl"]]
		sent_mean = float(sent["mean_negative"])
		ppl_mean = float(ppl["mean_ppl"])

		sent_mean_original = _to_float(row.get("sent_answer_mean", ""))
		ppl_mean_original = _to_float(row.get("ppl_answer_mean", ""))
		sent_abs_diff = None if sent_mean_original is None else abs(sent_mean - sent_mean_original)
		ppl_abs_diff = None if ppl_mean_original is None else abs(ppl_mean - ppl_mean_original)

		by_run[run_rel] = {
			"run_rel": run_rel,
			"n_valid_used": int(len(answers)),
			"sent_scores": sent_scores,
			"ppl_scores": ppl_scores,
			"sent_mean_recomputed": sent_mean,
			"ppl_mean_recomputed": ppl_mean,
			"sent_mean_original": sent_mean_original,
			"ppl_mean_original": ppl_mean_original,
			"sent_abs_diff": sent_abs_diff,
			"ppl_abs_diff": ppl_abs_diff,
		}
		n_computed += 1
		print(
			f"[recompute] {run_rel} n_valid={len(answers)} "
			f"sent_mean={sent_mean:.6f} ppl_mean={ppl_mean:.6f}"
		)

	sent_diffs = [v["sent_abs_diff"] for v in by_run.values() if v["sent_abs_diff"] is not None]
	ppl_diffs = [v["ppl_abs_diff"] for v in by_run.values() if v["ppl_abs_diff"] is not None]

	summary = {
		"n_runs": int(len(by_run)),
		"max_sent_abs_diff": (None if not sent_diffs else float(max(sent_diffs))),
		"mean_sent_abs_diff": (None if not sent_diffs else float(sum(sent_diffs) / len(sent_diffs))),
		"max_ppl_abs_diff": (None if not ppl_diffs else float(max(ppl_diffs))),
		"mean_ppl_abs_diff": (None if not ppl_diffs else float(sum(ppl_diffs) / len(ppl_diffs))),
	}

	payload = {
		"cache_path": str(cache_path),
		"sentiment_device": sentiment_device,
		"perplexity_device": perplexity_device,
		"max_valid_per_run": max_valid_per_run,
		"summary": summary,
		"runs": by_run,
	}
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	with cache_path.open("w") as f:
		json.dump(payload, f, indent=2)

	print(
		"[recompute-summary] "
		f"runs={summary['n_runs']} "
		f"computed_now={n_computed} "
		f"reused_cached={n_skipped_cached} "
		f"max_sent_abs_diff={summary['max_sent_abs_diff']} "
		f"max_ppl_abs_diff={summary['max_ppl_abs_diff']}"
	)
	print(f"[ok] wrote recomputed metric cache to {cache_path}")
	return by_run


def _load_recomputed_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
	if not cache_path.exists():
		return {}
	with cache_path.open("r") as f:
		payload = json.load(f)
	runs = payload.get("runs", {})
	if isinstance(runs, dict):
		return runs
	return {}


def _plot_one_alpha(
	alpha_key: str,
	rows: list[dict[str, str]],
	out_dir: Path,
	*,
	clean_plot: bool,
	recomputed_by_run: dict[str, dict[str, Any]] | None,
	ppl_valid_threshold: float | None,
) -> Path:
	rows_sorted = sorted(
		rows,
		key=lambda r: (
			r.get("schedule_mode", ""),
			r.get("q_conf_mode", ""),
			r.get("schedule_score_source", ""),
		),
	)

	labels = [_build_run_label(r) for r in rows_sorted]
	avg_steps_mean = [_to_float(r.get("avg_denoising_steps_per_generation", "")) for r in rows_sorted]
	sent_mean = [_to_float(r.get("sent_answer_mean", "")) for r in rows_sorted]
	ppl_mean = [_to_float(r.get("ppl_answer_mean", "")) for r in rows_sorted]

	valid_fraction_mean: list[float | None] = []
	for r in rows_sorted:
		n_total = _to_float(r.get("n_total", ""))
		n_valid = _to_float(r.get("n_valid", ""))
		if n_total is None or n_total <= 0 or n_valid is None:
			valid_fraction_mean.append(None)
		else:
			valid_fraction_mean.append(float(n_valid / n_total))

	avg_steps_dist: list[list[float]] = []
	for r, mean_val in zip(rows_sorted, avg_steps_mean):
		dist = _parse_float_list(r.get("denoising_steps_per_generation", ""))
		if dist:
			avg_steps_dist.append(dist)
		elif mean_val is not None:
			avg_steps_dist.append([mean_val])
		else:
			avg_steps_dist.append([])

	sent_dist = [[v] if v is not None else [] for v in sent_mean]
	valid_dist = [[v] if v is not None else [] for v in valid_fraction_mean]
	ppl_dist = [[v] if v is not None else [] for v in ppl_mean]

	if recomputed_by_run:
		for i, r in enumerate(rows_sorted):
			run_rel = str(r.get("run_rel", "")).strip()
			if not run_rel:
				continue
			rec = recomputed_by_run.get(run_rel)
			if not rec:
				continue
			sent_vals = rec.get("sent_scores", [])
			ppl_vals = rec.get("ppl_scores", [])
			if not isinstance(sent_vals, list) or not isinstance(ppl_vals, list) or not ppl_vals:
				continue

			if ppl_valid_threshold is None:
				if sent_vals:
					sent_dist[i] = [float(x) for x in sent_vals]
				ppl_dist[i] = [float(x) for x in ppl_vals]
				n_total = _to_float(r.get("n_total", ""))
				n_valid = float(len(ppl_vals))
				if n_total is not None and n_total > 0:
					valid_dist[i] = [n_valid / n_total]
				continue

			filtered = [
				(float(s), float(p))
				for s, p in zip(sent_vals, ppl_vals)
				if float(p) < float(ppl_valid_threshold)
			]
			if filtered:
				sent_dist[i] = [s for s, _ in filtered]
				ppl_dist[i] = [p for _, p in filtered]
			else:
				sent_dist[i] = []
				ppl_dist[i] = []

			n_total = _to_float(r.get("n_total", ""))
			if n_total is not None and n_total > 0:
				valid_dist[i] = [float(len(filtered) / n_total)]

	metrics = [
		("Avg Denoising Steps", avg_steps_dist),
		("Mean Sentiment P(NEGATIVE)", sent_dist),
		("Valid Fraction", valid_dist),
		("Mean Perplexity", ppl_dist),
	]

	fig_width = max(20.0, 1.9 * len(labels) + 10.0)
	fig, axes = plt.subplots(2, 2, figsize=(fig_width, 11))
	axes_flat = axes.flatten()

	for ax, (title, values) in zip(axes_flat, metrics):
		xs = list(range(len(labels)))
		means: list[float] = []
		err_low: list[float] = []
		err_high: list[float] = []
		for vals in values:
			if not vals:
				means.append(float("nan"))
				err_low.append(float("nan"))
				err_high.append(float("nan"))
				continue
			arr = np.asarray(vals, dtype=np.float64)
			m = float(np.mean(arr))
			p15 = float(np.percentile(arr, 15))
			p85 = float(np.percentile(arr, 75))
			means.append(m)
			err_low.append(max(0.0, m - p15))
			err_high.append(max(0.0, p85 - m))

		ax.bar(xs, means, yerr=[err_low, err_high], capsize=3)
		ax.set_title(title)
		ax.set_xticks(xs)
		ax.set_xticklabels(labels, rotation=28, ha="right")
		ax.grid(axis="y", linestyle="--", alpha=0.35)

		if title in {"Mean Sentiment P(NEGATIVE)", "Valid Fraction"}:
			ax.set_ylim(0.0, 1.0)

	title_prefix = "Schedule Metrics"
	if clean_plot:
		title_prefix = "Schedule Metrics (clean:plot)"
	if ppl_valid_threshold is not None:
		title_prefix += f" [valid: ppl<{ppl_valid_threshold:g}]"
	fig.suptitle(f"{title_prefix} - alpha={alpha_key}", fontsize=14)
	fig.tight_layout(rect=[0, 0.06, 1, 0.97])
	fig.subplots_adjust(wspace=0.22, hspace=0.42)

	suffix = "_clean_plot" if clean_plot else ""
	out_path = out_dir / f"alpha_{_fmt_alpha_for_filename(alpha_key)}_metrics_4panels{suffix}.png"
	fig.savefig(out_path, dpi=160)
	plt.close(fig)
	return out_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--summary-csv",
		type=Path,
		default=Path(
			"data/speed_adapt/compare_mean_steering_adaptive_schedule_instruct/full_run/summary.csv"
		),
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path(
			"data/speed_adapt/compare_mean_steering_adaptive_schedule_instruct/full_run"
		),
	)
	parser.add_argument(
		"--full-run-dir",
		type=Path,
		default=Path(
			"data/speed_adapt/compare_mean_steering_adaptive_schedule_instruct/full_run"
		),
		help="Directory containing run subfolders with metrics.json",
	)
	parser.add_argument(
		"--alphas",
		nargs="*",
		type=float,
		default=[2.0, 4.0],
		help="Alpha values to plot, e.g. --alphas 2 4",
	)
	parser.add_argument(
		"--include-alpha4-current-mean-all",
		action="store_true",
		help="Include alpha=4, mix_adaptive, mean_all_block at current thresholds (0.22, 0.41).",
	)
	parser.add_argument(
		"--clean-plot",
		action="store_true",
		help="Keep only fixed schedules and calibrated lambda runs (clean:plot).",
	)
	parser.add_argument(
		"--recompute-clean-metrics",
		action="store_true",
		help="Recompute per-text sentiment/perplexity distributions for clean:plot rows and store cache.",
	)
	parser.add_argument(
		"--recompute-cache-path",
		type=Path,
		default=DEFAULT_RECOMPUTE_CACHE_PATH,
	)
	parser.add_argument("--sentiment-device", type=str, default="cpu")
	parser.add_argument("--perplexity-device", type=str, default="cpu")
	parser.add_argument(
		"--recompute-max-valid-per-run",
		type=int,
		default=None,
		help="Optional cap for valid samples per run when recomputing metrics.",
	)
	parser.add_argument(
		"--ppl-valid-threshold",
		type=float,
		default=None,
		help="If set, use cached ppl/sent scores and define valid sentences as ppl<threshold for valid fraction and sentiment/perplexity panels.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	rows_from_runs = _load_rows_from_run_dirs(args.full_run_dir)
	if rows_from_runs:
		rows = rows_from_runs
		print(f"[info] loaded {len(rows)} rows from run folders in {args.full_run_dir}")
	else:
		rows = _load_rows(args.summary_csv)
		print(f"[info] loaded {len(rows)} rows from summary csv {args.summary_csv}")

	if not args.include_alpha4_current_mean_all:
		before = len(rows)
		rows = [r for r in rows if not _is_unfinished_alpha4_mean_all_current_threshold(r)]
		removed = before - len(rows)
		if removed > 0:
			print(
				"[info] excluded alpha=4 mix_adaptive mean_all_block at q_mid=0.22/q_high=0.41 "
				f"(rows removed: {removed})"
			)

	if args.clean_plot:
		before = len(rows)
		rows = _keep_only_clean_plot_rows(rows)
		removed = before - len(rows)
		print(f"[info] clean:plot filter applied (rows removed: {removed}, rows kept: {len(rows)})")

	recomputed_by_run: dict[str, dict[str, Any]] = {}
	if args.recompute_clean_metrics:
		if not args.clean_plot:
			raise ValueError("--recompute-clean-metrics currently requires --clean-plot")
		recomputed_by_run = _recompute_clean_plot_distributions(
			rows=rows,
			full_run_dir=args.full_run_dir,
			cache_path=args.recompute_cache_path,
			sentiment_device=args.sentiment_device,
			perplexity_device=args.perplexity_device,
			max_valid_per_run=args.recompute_max_valid_per_run,
		)
	elif args.clean_plot:
		recomputed_by_run = _load_recomputed_cache(args.recompute_cache_path)
		if recomputed_by_run:
			print(
				f"[info] loaded recomputed clean metric cache for {len(recomputed_by_run)} runs "
				f"from {args.recompute_cache_path}"
			)

	args.out_dir.mkdir(parents=True, exist_ok=True)

	alpha_targets = {_to_alpha_key(a) for a in args.alphas}

	created_paths: list[Path] = []
	for alpha_key in sorted(alpha_targets, key=float):
		alpha_rows = [r for r in rows if _to_alpha_key(float(r.get("alpha", "nan"))) == alpha_key]
		alpha_rows = _inject_baseline_row_for_clean_plot(
			alpha_key=alpha_key,
			alpha_rows=alpha_rows,
			all_rows=rows,
			clean_plot=bool(args.clean_plot),
		)
		if not alpha_rows:
			print(f"[warn] no rows found for alpha={alpha_key}")
			continue
		out_path = _plot_one_alpha(
			alpha_key=alpha_key,
			rows=alpha_rows,
			out_dir=args.out_dir,
			clean_plot=bool(args.clean_plot),
			recomputed_by_run=recomputed_by_run,
			ppl_valid_threshold=args.ppl_valid_threshold,
		)
		created_paths.append(out_path)
		print(f"[ok] wrote {out_path}")

	if not created_paths:
		raise RuntimeError("No plots were created. Check summary.csv and requested alphas.")


if __name__ == "__main__":
	main()

