from __future__ import annotations

"""
Compute prompt-response coherence for schedule comparison runs.

This script reads existing generation artifacts (no LLaDA generation), and computes
SimCSE cosine similarity coherence between:
- prompt text fed to the model (25-word prompt prefix)
- generated answer text

Coherence is computed on the same valid subset used by perplexity in the compare
script (rows with is_valid=True in generations.jsonl).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.fluency_metrics import compute_coherence


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _is_unfinished_alpha4_mean_all_current_threshold(row: dict[str, str]) -> bool:
    alpha = _to_float(row.get("alpha"))
    q_mid = _to_float(row.get("q_mid"))
    q_high = _to_float(row.get("q_high"))
    if alpha is None or q_mid is None or q_high is None:
        return False
    return (
        abs(alpha - 4.0) < 1e-9
        and row.get("schedule_mode", "") == "mix_adaptive"
        and row.get("q_conf_mode", "") == "mean_all_block"
        and abs(q_mid - 0.22) < 1e-9
        and abs(q_high - 0.41) < 1e-9
    )


def _discover_runs(full_run_dir: Path) -> list[dict[str, str]]:
    allowed_schedule_modes = {"fixed_1", "fixed_2", "fixed_3", "adaptive_123", "mix_adaptive"}
    runs: list[dict[str, str]] = []

    for metrics_path in sorted(full_run_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        run_info_path = run_dir / "run_info.json"
        gens_path = run_dir / "generations.jsonl"
        if not gens_path.exists():
            continue

        metrics = _load_json(metrics_path)
        run_info = _load_json(run_info_path) if run_info_path.exists() else {}

        merged: dict[str, Any] = dict(metrics)
        for key in ["schedule_score_source", "q_conf_mode", "q_mid", "q_high", "method", "alpha", "schedule_mode"]:
            if key not in merged and key in run_info:
                merged[key] = run_info[key]

        schedule_mode = str(merged.get("schedule_mode", "")).strip()
        if schedule_mode not in allowed_schedule_modes:
            continue

        row = {k: ("" if v is None else str(v)) for k, v in merged.items()}
        row["run_dir"] = str(run_dir)
        row["run_rel"] = str(run_dir.relative_to(full_run_dir))

        if _is_unfinished_alpha4_mean_all_current_threshold(row):
            continue

        runs.append(row)

    return runs


def _load_valid_prompt_response_pairs(
    generations_path: Path,
    max_pairs: int | None,
) -> tuple[list[str], list[str], int, int, int]:
    prompts: list[str] = []
    responses: list[str] = []
    n_total = 0
    n_valid_total = 0

    with generations_path.open("r") as f:
        for line in f:
            n_total += 1
            obj = json.loads(line)
            is_valid = bool(obj.get("is_valid", False))
            if not is_valid:
                continue

            n_valid_total += 1
            if max_pairs is None or len(prompts) < max_pairs:
                prompts.append(str(obj.get("prompt", "")))
                responses.append(str(obj.get("answer", "")))

    return prompts, responses, n_total, n_valid_total, len(prompts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-run-dir",
        type=Path,
        default=Path("data/speed_adapt/compare_mean_steering_adaptive_schedule_instruct/full_run"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <full-run-dir>/coherence",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--smoke-valid-per-run",
        type=int,
        default=3,
        help="Max valid prompt/response pairs per run when --smoke-test is set.",
    )
    parser.add_argument(
        "--max-valid-per-run",
        type=int,
        default=None,
        help="Optional cap for full mode as well.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_run_dir = args.full_run_dir
    out_dir = args.out_dir or (full_run_dir / "coherence")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(full_run_dir)
    if not runs:
        raise RuntimeError(f"No eligible runs found in {full_run_dir}")

    max_pairs = args.smoke_valid_per_run if args.smoke_test else args.max_valid_per_run

    rows: list[dict[str, Any]] = []
    for row in runs:
        run_dir = Path(row["run_dir"])
        gens_path = run_dir / "generations.jsonl"
        prompts, responses, n_total, n_valid_total, n_valid_used = _load_valid_prompt_response_pairs(
            gens_path,
            max_pairs=max_pairs,
        )

        if not prompts:
            result = {
                "mean_coherence": None,
                "per_text_coherence": [],
            }
        else:
            result = compute_coherence(prompts=prompts, responses=responses, device=args.device)

        per_run_payload = {
            "run_dir": row["run_rel"],
            "alpha": _to_float(row.get("alpha")),
            "schedule_mode": row.get("schedule_mode"),
            "schedule_score_source": row.get("schedule_score_source"),
            "q_conf_mode": row.get("q_conf_mode"),
            "q_mid": _to_float(row.get("q_mid")),
            "q_high": _to_float(row.get("q_high")),
            "n_total_rows_in_file": int(n_total),
            "n_valid_rows_in_file": int(n_valid_total),
            "n_valid_used_for_coherence": int(n_valid_used),
            "mean_coherence": result["mean_coherence"],
            "per_text_coherence": result["per_text_coherence"],
            "device": args.device,
            "smoke_test": bool(args.smoke_test),
        }

        with (run_dir / "coherence_metrics.json").open("w") as f:
            json.dump(per_run_payload, f, indent=2)

        rows.append(per_run_payload)
        print(
            f"[ok] {row['run_rel']} -> mean_coherence={per_run_payload['mean_coherence']} "
            f"(n={per_run_payload['n_valid_used_for_coherence']})"
        )

    summary_json = {
        "full_run_dir": str(full_run_dir),
        "device": args.device,
        "smoke_test": bool(args.smoke_test),
        "max_valid_per_run": max_pairs,
        "n_runs": len(rows),
        "rows": rows,
    }

    with (out_dir / ("smoke_summary.json" if args.smoke_test else "summary.json")).open("w") as f:
        json.dump(summary_json, f, indent=2)

    csv_path = out_dir / ("smoke_summary.csv" if args.smoke_test else "summary.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_dir",
                "alpha",
                "schedule_mode",
                "schedule_score_source",
                "q_conf_mode",
                "q_mid",
                "q_high",
                "n_total_rows_in_file",
                "n_valid_rows_in_file",
                "n_valid_used_for_coherence",
                "mean_coherence",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["run_dir"],
                    r["alpha"],
                    r["schedule_mode"],
                    r["schedule_score_source"],
                    r["q_conf_mode"],
                    r["q_mid"],
                    r["q_high"],
                    r["n_total_rows_in_file"],
                    r["n_valid_rows_in_file"],
                    r["n_valid_used_for_coherence"],
                    r["mean_coherence"],
                ]
            )

    print(f"[done] wrote coherence summaries to {out_dir}")


if __name__ == "__main__":
    main()
