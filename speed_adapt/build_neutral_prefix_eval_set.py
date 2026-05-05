from __future__ import annotations

"""
build_neutral_prefix_eval_set.py

Build a neutral-prefix evaluation subset from Amazon Polarity.

Selection rule per candidate review:
1) full source review must have > min_source_words
2) prompt is the first prompt_words words
3) score prompt with sentiment model used in eval/sentiment_metrics.py
4) keep only if neutral_min <= P(NEGATIVE) <= neutral_max

Outputs under --out-dir/(smoke_test|full_run):
- neutral_prefix_eval_set.json
- summary.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.sentiment_metrics import compute_sentiment_metrics


SEED = 42

SENTIMENT_DEVICE = os.getenv("SENTIMENT_DEVICE", "cpu")

DATASET_NAME = "fancyzhx/amazon_polarity"
DATASET_SPLIT = "train"

START_IDX = 50_000
TARGET_COUNT = 500

MIN_SOURCE_WORDS = 100
PROMPT_WORDS = 25
NEUTRAL_MIN = 0.4
NEUTRAL_MAX = 0.6

SCORE_BATCH_SIZE = 64

SMOKE_TEST = False
SMOKE_TARGET_COUNT = 2

OUT_DIR = Path("data/speed_adapt/neutral_prefix_eval_set")


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()


def _score_prefixes(prefixes: list[str], device: str) -> list[float]:
    if not prefixes:
        return []
    m = compute_sentiment_metrics(prefixes, device=device)
    return [float(x) for x in m["scores"].tolist()]


def build_eval_set(
    *,
    dataset_name: str,
    split: str,
    start_idx: int,
    target_count: int,
    min_source_words: int,
    prompt_words: int,
    neutral_min: float,
    neutral_max: float,
    score_batch_size: int,
    sentiment_device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)

    kept: list[dict[str, Any]] = []
    scanned_after_start = 0

    candidate_rows: list[dict[str, Any]] = []
    candidate_prefixes: list[str] = []

    def flush_candidates() -> None:
        nonlocal candidate_rows, candidate_prefixes, kept
        if not candidate_rows:
            return

        scores = _score_prefixes(candidate_prefixes, device=sentiment_device)
        for row, s in zip(candidate_rows, scores):
            if neutral_min <= s <= neutral_max:
                row["prefix_negative_score"] = float(s)
                kept.append(row)
                if len(kept) >= target_count:
                    break

        candidate_rows = []
        candidate_prefixes = []

    for i, ex in enumerate(ds):
        if i < start_idx:
            continue

        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        words = text.split()
        n_words = len(words)

        scanned_after_start += 1
        if n_words <= min_source_words:
            continue

        prompt_text = _truncate(text, prompt_words)
        if len(prompt_text.split()) < 5:
            continue

        candidate_rows.append(
            {
                "dataset_index": int(i),
                "label": int(ex["label"]),
                "full_text_word_count": int(n_words),
                "prompt_words": int(prompt_words),
                "prompt_text": prompt_text,
            }
        )
        candidate_prefixes.append(prompt_text)

        if len(candidate_rows) >= score_batch_size:
            flush_candidates()
            if len(kept) >= target_count:
                break

    if len(kept) < target_count:
        flush_candidates()

    kept = kept[:target_count]

    meta = {
        "dataset": dataset_name,
        "split": split,
        "start_idx": int(start_idx),
        "target_count": int(target_count),
        "neutral_min": float(neutral_min),
        "neutral_max": float(neutral_max),
        "min_source_words": int(min_source_words),
        "prompt_words": int(prompt_words),
        "sentiment_device": sentiment_device,
        "score_definition": "P(NEGATIVE) in [0,1]",
        "number_scanned_after_start": int(scanned_after_start),
        "number_kept": int(len(kept)),
    }

    return kept, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=DATASET_NAME)
    p.add_argument("--split", type=str, default=DATASET_SPLIT)
    p.add_argument("--start-idx", type=int, default=START_IDX)
    p.add_argument("--target-count", type=int, default=TARGET_COUNT)
    p.add_argument("--min-source-words", type=int, default=MIN_SOURCE_WORDS)
    p.add_argument("--prompt-words", type=int, default=PROMPT_WORDS)
    p.add_argument("--neutral-min", type=float, default=NEUTRAL_MIN)
    p.add_argument("--neutral-max", type=float, default=NEUTRAL_MAX)
    p.add_argument("--score-batch-size", type=int, default=SCORE_BATCH_SIZE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-target-count", type=int, default=SMOKE_TARGET_COUNT)
    p.set_defaults(smoke_test=SMOKE_TEST)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(SEED)

    run_dir = args.out_dir / ("smoke_test" if args.smoke_test else "full_run")
    run_dir.mkdir(parents=True, exist_ok=True)

    target_count = args.smoke_target_count if args.smoke_test else args.target_count

    rows, meta = build_eval_set(
        dataset_name=args.dataset,
        split=args.split,
        start_idx=args.start_idx,
        target_count=target_count,
        min_source_words=args.min_source_words,
        prompt_words=args.prompt_words,
        neutral_min=args.neutral_min,
        neutral_max=args.neutral_max,
        score_batch_size=args.score_batch_size,
        sentiment_device=SENTIMENT_DEVICE,
    )

    label_counts = {
        "negative_0": int(sum(1 for r in rows if r["label"] == 0)),
        "positive_1": int(sum(1 for r in rows if r["label"] == 1)),
    }

    mean_prefix_score = float(np.mean([r["prefix_negative_score"] for r in rows])) if rows else None

    payload = {
        "meta": {
            **meta,
            "smoke_test": bool(args.smoke_test),
        },
        "rows": rows,
    }

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "start_idx": int(args.start_idx),
        "target_count": int(target_count),
        "number_kept": int(len(rows)),
        "neutral_min": float(args.neutral_min),
        "neutral_max": float(args.neutral_max),
        "min_source_words": int(args.min_source_words),
        "prompt_words": int(args.prompt_words),
        "score_definition": "P(NEGATIVE) in [0,1]",
        "mean_prefix_negative_score": mean_prefix_score,
        "label_counts": label_counts,
        "number_scanned_after_start": int(meta["number_scanned_after_start"]),
        "smoke_test": bool(args.smoke_test),
    }

    _write_json(run_dir / "neutral_prefix_eval_set.json", payload)
    _write_json(run_dir / "summary.json", summary)

    print(f"[done] wrote outputs under {run_dir}")
    print(f"[done] kept={len(rows)} scanned_after_start={meta['number_scanned_after_start']}")


if __name__ == "__main__":
    main()
