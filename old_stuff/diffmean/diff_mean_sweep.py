from __future__ import annotations

"""
diffmean_lambda_sweep_holdout.py

Evaluate ONLY DiffMean steering on a held-out Amazon Polarity slice while
reusing the already-saved mean-diff vectors from the unified LLaDA pipeline.

What this does
--------------
- loads prompts from amazon_polarity train[50000:50500]
- evaluates only the first two activation variants:
    * real_full_pooled
    * masked_pooled
- sweeps DiffMean steering strength lambda from 1 to 20 (inclusive)
- computes bad-pattern / invalid-output fractions
- computes sentiment and perplexity ONLY on valid generations
- does NOT save individual answers

Example:
    python diffmean_lambda_sweep_holdout.py \
        --out-root data/alignment_variants_v4/full_run
"""

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity
from steering.precomputed_steering import PrecomputedLayerSteering


# -----------------------------------------------------------------------------
# Defaults aligned with the existing steering pipeline
# -----------------------------------------------------------------------------

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
PROMPT_WORDS = 20
STEER_LAYERS = list(range(9, 25))
VARIANTS = ["real_full_pooled", "masked_pooled"]

EVAL_START_IDX = 50_000
EVAL_END_IDX = 50_200

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",
)

BAD_PATTERNS = [
    r"Is this .*?\?",
    r"positive or negative\?",
    r"\bAnswer:",
    r"\bYesTitle:",
    r"\bNoTitle:",
    r"\bTitle:",
    r"\bReview:",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()



def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)



def has_bad_pattern(text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in BAD_PATTERNS)



def has_repetition_loop(text: str) -> bool:
    toks = text.split()
    if len(toks) < 12:
        return False
    for n in [2, 3, 4]:
        for i in range(len(toks) - 2 * n + 1):
            if toks[i:i+n] == toks[i+n:i+2*n]:
                return True
    return False



def classify_generation_quality(answer: str) -> dict[str, bool]:
    bad_pattern = has_bad_pattern(answer)
    repetition = has_repetition_loop(answer)
    empty = not answer.strip()
    is_valid = not (bad_pattern or repetition or empty)
    return {
        "is_valid": is_valid,
        "has_bad_pattern": bad_pattern,
        "has_repetition_loop": repetition,
        "is_empty": empty,
    }


# -----------------------------------------------------------------------------
# Held-out data
# -----------------------------------------------------------------------------


def load_holdout_prompts(
    start_idx: int = EVAL_START_IDX,
    end_idx: int = EVAL_END_IDX,
    prompt_words: int = PROMPT_WORDS,
) -> tuple[list[str], list[int], dict[str, Any]]:
    split = f"train[{start_idx}:{end_idx}]"
    ds = load_dataset("fancyzhx/amazon_polarity", split=split)

    prompts: list[str] = []
    labels: list[int] = []

    for ex in ds:
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        prompt = _truncate(text, prompt_words)
        if len(prompt.split()) < 5:
            continue
        prompts.append(prompt)
        labels.append(int(ex["label"]))

    meta = {
        "dataset": "fancyzhx/amazon_polarity",
        "split": split,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "prompt_words": prompt_words,
        "n_prompts": len(prompts),
        "label_counts": {
            "negative_0": int(sum(1 for x in labels if x == 0)),
            "positive_1": int(sum(1 for x in labels if x == 1)),
        },
    }
    return prompts, labels, meta


# -----------------------------------------------------------------------------
# Generation / eval
# -----------------------------------------------------------------------------


def generate_with_steering(model, tokenizer, prompts: list[str], steerer, desc: str) -> list[str]:
    answers: list[str] = []
    for prompt in tqdm(prompts, desc=desc, leave=False):
        enc = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(DEVICE)
        out = llada_generate(
            model=model,
            prompt=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            steps=GEN_PARAMS["steps"],
            gen_length=GEN_PARAMS["gen_length"],
            block_length=GEN_PARAMS["block_length"],
            temperature=GEN_PARAMS["temperature"],
            fill_strategy=GEN_PARAMS["fill_strategy"],
            remasking=None,
            remask_fraction=0.0,
            remask_fixed_count=None,
            remask_start_frac=0.0,
            steering=steerer,
            show_progress=False,
        )
        generated_ids = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers.extend(decoded)
    return answers



def evaluate_texts_filtered(prompts: list[str], answers: list[str]) -> dict[str, Any]:
    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]
    quality = [classify_generation_quality(a) for a in answers]

    valid_indices = [i for i, q in enumerate(quality) if q["is_valid"]]
    valid_answers = [answers[i] for i in valid_indices]
    valid_combined = [combined[i] for i in valid_indices]

    n_total = len(answers)
    n_valid = len(valid_indices)
    n_invalid = n_total - n_valid

    bad_pattern_count = int(sum(q["has_bad_pattern"] for q in quality))
    repetition_count = int(sum(q["has_repetition_loop"] for q in quality))
    empty_count = int(sum(q["is_empty"] for q in quality))

    invalid_fraction = float(n_invalid / max(1, n_total))
    bad_pattern_fraction = float(bad_pattern_count / max(1, n_total))
    repetition_fraction = float(repetition_count / max(1, n_total))
    empty_fraction = float(empty_count / max(1, n_total))

    if n_valid > 0:
        sent_ans = compute_sentiment_metrics(valid_answers, device=DEVICE)
        sent_comb = compute_sentiment_metrics(valid_combined, device=DEVICE)
        ppl_ans = compute_perplexity(valid_answers, device="cpu")
        ppl_comb = compute_perplexity(valid_combined, device="cpu")
        sent_answer_mean = sent_ans["mean_negative"]
        sent_answer_fraction = sent_ans["negative_fraction"]
        sent_combined_mean = sent_comb["mean_negative"]
        ppl_answer_mean = ppl_ans["mean_ppl"]
        ppl_combined_mean = ppl_comb["mean_ppl"]
        mean_answer_words_valid_only = float(sum(len(a.split()) for a in valid_answers) / max(1, len(valid_answers)))
    else:
        sent_answer_mean = None
        sent_answer_fraction = None
        sent_combined_mean = None
        ppl_answer_mean = None
        ppl_combined_mean = None
        mean_answer_words_valid_only = None

    return {
        "evaluation_subset": "valid_generations_only",
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_fraction": invalid_fraction,
        "bad_pattern_count": bad_pattern_count,
        "bad_pattern_fraction": bad_pattern_fraction,
        "repetition_count": repetition_count,
        "repetition_fraction": repetition_fraction,
        "empty_count": empty_count,
        "empty_fraction": empty_fraction,
        "sent_answer_mean": sent_answer_mean,
        "sent_answer_fraction": sent_answer_fraction,
        "sent_combined_mean": sent_combined_mean,
        "ppl_answer_mean": ppl_answer_mean,
        "ppl_combined_mean": ppl_combined_mean,
        "mean_answer_words_all": float(sum(len(a.split()) for a in answers) / max(1, len(answers))),
        "mean_answer_words_valid_only": mean_answer_words_valid_only,
    }


# -----------------------------------------------------------------------------
# DiffMean sweep
# -----------------------------------------------------------------------------


def load_mean_vectors(probe_root: Path, variant: str, layer_ids: list[int]) -> dict[int, torch.Tensor]:
    p = probe_root / variant / "mean_diff_vectors.pt"
    obj = torch.load(p, map_location="cpu", weights_only=False)
    vectors: dict[int, torch.Tensor] = {}
    for k, v in obj.items():
        li = int(k)
        if li in layer_ids:
            vectors[li] = v.float().cpu()
    if not vectors:
        raise RuntimeError(f"No steering vectors found for variant={variant} under {p}")
    return vectors



def run_lambda_sweep_for_variant(
    variant: str,
    lambdas: list[float],
    prompts: list[str],
    model,
    tokenizer,
    probe_root: Path,
    eval_root: Path,
) -> dict[str, Any]:
    variant_eval_dir = eval_root / variant
    variant_eval_dir.mkdir(parents=True, exist_ok=True)

    vectors = load_mean_vectors(probe_root, variant, STEER_LAYERS)
    results: dict[str, Any] = {}

    for lam in lambdas:
        tag = f"lambda_{int(lam):02d}" if float(lam).is_integer() else f"lambda_{lam:.2f}".replace(".", "p")
        steerer = PrecomputedLayerSteering(vectors=vectors, layer_ids=STEER_LAYERS, alpha=float(lam))
        answers = generate_with_steering(model, tokenizer, prompts, steerer, f"diffmean[{variant} λ={lam}]")
        metrics = evaluate_texts_filtered(prompts, answers)
        metrics["method"] = "mean_diff"
        metrics["variant"] = variant
        metrics["lambda"] = float(lam)
        _write_json(variant_eval_dir / f"{tag}_metrics.json", metrics)
        results[tag] = metrics

    csv_lines = [
        "variant,lambda,n_total,n_valid,n_invalid,invalid_fraction,bad_pattern_fraction,repetition_fraction,empty_fraction,sent_answer_mean,sent_answer_fraction,sent_combined_mean,ppl_answer_mean,ppl_combined_mean,mean_answer_words_all,mean_answer_words_valid_only"
    ]
    for tag, m in results.items():
        csv_lines.append(
            ",".join([
                variant,
                str(m["lambda"]),
                str(m["n_total"]),
                str(m["n_valid"]),
                str(m["n_invalid"]),
                str(m["invalid_fraction"]),
                str(m["bad_pattern_fraction"]),
                str(m["repetition_fraction"]),
                str(m["empty_fraction"]),
                str(m["sent_answer_mean"]),
                str(m["sent_answer_fraction"]),
                str(m["sent_combined_mean"]),
                str(m["ppl_answer_mean"]),
                str(m["ppl_combined_mean"]),
                str(m["mean_answer_words_all"]),
                str(m["mean_answer_words_valid_only"]),
            ])
        )
    (variant_eval_dir / "lambda_sweep_summary.csv").write_text("\n".join(csv_lines) + "\n")
    _write_json(variant_eval_dir / "lambda_sweep_summary.json", results)
    return results


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/alignment_variants_v4/full_run"),
        help="Existing run root containing probes/ from the earlier pipeline.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=VARIANTS,
        help="Variants to evaluate. Defaults to real_full_pooled masked_pooled.",
    )
    parser.add_argument(
        "--lambda-min",
        type=int,
        default=0,
        help="Minimum DiffMean lambda to test.",
    )
    parser.add_argument(
        "--lambda-max",
        type=int,
        default=20,
        help="Maximum DiffMean lambda to test.",
    )
    parser.add_argument(
        "--lambda-step",
        type=int,
        default=1,
        help="Step size for DiffMean lambda sweep.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=EVAL_START_IDX,
        help="Held-out dataset slice start index.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=EVAL_END_IDX,
        help="Held-out dataset slice end index (exclusive).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Model name to load for generation.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    _seed_everything(SEED)

    if args.lambda_step <= 0:
        raise ValueError("--lambda-step must be positive")
    if args.lambda_max < args.lambda_min:
        raise ValueError("--lambda-max must be >= --lambda-min")

    out_root = args.out_root
    probe_root = out_root / "probes"
    eval_root = out_root / "diffmean_holdout_lambda_sweep"
    eval_root.mkdir(parents=True, exist_ok=True)

    prompts, labels, eval_meta = load_holdout_prompts(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        prompt_words=PROMPT_WORDS,
    )
    _write_json(
        eval_root / "eval_set.json",
        {
            "meta": eval_meta,
            "labels": labels,
            "prompts": prompts,
        },
    )

    lambdas = [float(x) for x in range(args.lambda_min, args.lambda_max + 1, args.lambda_step)]

    print(f"[model] loading {args.model_name} on {DEVICE} ...")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)

    summary: dict[str, Any] = {
        "out_root": str(out_root),
        "probe_root": str(probe_root),
        "eval_root": str(eval_root),
        "model_name": args.model_name,
        "device": DEVICE,
        "method": "mean_diff",
        "variants": list(args.variants),
        "steer_layers": STEER_LAYERS,
        "lambdas": lambdas,
        "evaluation_subset": "valid_generations_only",
        "bad_patterns": BAD_PATTERNS,
        "eval_meta": eval_meta,
        "results": {},
    }

    aggregate_rows = [
        "variant,lambda,n_total,n_valid,n_invalid,invalid_fraction,bad_pattern_fraction,repetition_fraction,empty_fraction,sent_answer_mean,sent_answer_fraction,sent_combined_mean,ppl_answer_mean,ppl_combined_mean,mean_answer_words_all,mean_answer_words_valid_only"
    ]

    for variant in args.variants:
        print(f"\n[eval] variant={variant}")
        variant_results = run_lambda_sweep_for_variant(
            variant=variant,
            lambdas=lambdas,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            probe_root=probe_root,
            eval_root=eval_root,
        )
        summary["results"][variant] = variant_results

        for tag, m in variant_results.items():
            aggregate_rows.append(
                ",".join([
                    variant,
                    str(m["lambda"]),
                    str(m["n_total"]),
                    str(m["n_valid"]),
                    str(m["n_invalid"]),
                    str(m["invalid_fraction"]),
                    str(m["bad_pattern_fraction"]),
                    str(m["repetition_fraction"]),
                    str(m["empty_fraction"]),
                    str(m["sent_answer_mean"]),
                    str(m["sent_answer_fraction"]),
                    str(m["sent_combined_mean"]),
                    str(m["ppl_answer_mean"]),
                    str(m["ppl_combined_mean"]),
                    str(m["mean_answer_words_all"]),
                    str(m["mean_answer_words_valid_only"]),
                ])
            )

    _write_json(eval_root / "summary.json", summary)
    (eval_root / "summary.csv").write_text("\n".join(aggregate_rows) + "\n")
    print(f"\n[done] wrote outputs under {eval_root}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()