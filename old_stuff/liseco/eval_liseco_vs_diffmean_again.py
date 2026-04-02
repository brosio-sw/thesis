from __future__ import annotations

"""
redo_filtered_eval_from_saved_probes.py

Re-run ONLY the generation/evaluation stage of the unified LLaDA steering
comparison, while reusing already-trained probes / mean-diff vectors stored on
 disk.

Changes vs the original eval:
- sentiment and perplexity are computed ONLY on valid generations
- a generation is invalid if it is empty, matches a BAD_PATTERNS regex,
  or contains a simple repetition loop
- the script explicitly reports bad_pattern_fraction (and related fractions)

This script does NOT retrain probes and does NOT recollect activations.
It only:
  1) reloads saved mean-diff vectors / saved probe weights
  2) regenerates completions for the eval prompts
  3) recomputes filtered metrics

Example:
    python redo_filtered_eval_from_saved_probes.py \
        --out-root data/alignment_variants_v4/full_run

Optional:
    python redo_filtered_eval_from_saved_probes.py \
        --out-root data/alignment_variants_v4/full_run \
        --variants real_full_pooled masked_pooled masked_tokenwise
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
from steering.liseco_probe_steering import LiSeCoProbeSteering, ProbeParams


# -----------------------------------------------------------------------------
# Defaults copied from the original evaluation setup
# -----------------------------------------------------------------------------

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
MASK_ID = 126336

PROMPT_WORDS = 20
EVAL_NEG = 25
EVAL_POS = 25
EVAL_ANY = 50

STEER_LAYERS = list(range(9, 25))
MEAN_ALPHA = 8.0
LISECO_INTERVALS = [
    (0.00, 0.20),
    (0.80, 1.00),
]

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
# Helpers
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



def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")



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
# Prompt loading (same logic as the original eval pipeline)
# -----------------------------------------------------------------------------


def load_amazon_prompts_by_label(n: int, label: int, n_words: int = PROMPT_WORDS) -> list[str]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    out: list[str] = []
    for ex in ds:
        if int(ex["label"]) != label:
            continue
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        prompt = _truncate(text, n_words)
        if len(prompt.split()) >= 5:
            out.append(prompt)
        if len(out) >= n:
            break
    return out



def load_amazon_prompts_any(
    n: int,
    skip_neg: int = 0,
    skip_pos: int = 0,
    n_words: int = PROMPT_WORDS,
) -> list[str]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    out: list[str] = []
    seen_neg = 0
    seen_pos = 0
    for ex in ds:
        lab = int(ex["label"])
        if lab == 0 and seen_neg < skip_neg:
            seen_neg += 1
            continue
        if lab == 1 and seen_pos < skip_pos:
            seen_pos += 1
            continue
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        prompt = _truncate(text, n_words)
        if len(prompt.split()) >= 5:
            out.append(prompt)
        if len(out) >= n:
            break
    return out



def load_eval_prompts() -> tuple[list[str], list[str], list[int]]:
    neg_prompts = load_amazon_prompts_by_label(EVAL_NEG, label=0)
    pos_prompts = load_amazon_prompts_by_label(EVAL_POS, label=1)
    any_prompts = load_amazon_prompts_any(EVAL_ANY, skip_neg=EVAL_NEG, skip_pos=EVAL_POS)

    prompts = neg_prompts + pos_prompts + any_prompts
    labels = [0] * len(neg_prompts) + [1] * len(pos_prompts) + [-1] * len(any_prompts)

    rng = random.Random(SEED + 1)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    labels = [labels[i] for i in order]
    raw_texts = [p for p in prompts]
    return prompts, raw_texts, labels


# -----------------------------------------------------------------------------
# Steering / generation
# -----------------------------------------------------------------------------


def load_probe_params_for_variant(
    probe_root: Path,
    variant: str,
    layer_ids: list[int],
) -> dict[int, ProbeParams]:
    probe_by_layer: dict[int, ProbeParams] = {}
    for li in layer_ids:
        p = probe_root / variant / f"layer_{li:02d}" / "probe_weight_bias.pt"
        obj = torch.load(p, map_location="cpu", weights_only=False)
        w = obj["weight"].float().cpu().view(-1)
        b = obj["bias"].float().cpu().view(-1)
        probe_by_layer[li] = ProbeParams(weight=w, bias=b, norm_sq=float((w * w).sum().item()))
    return probe_by_layer



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


# -----------------------------------------------------------------------------
# Filtered evaluation
# -----------------------------------------------------------------------------


def evaluate_texts_filtered(prompts: list[str], answers: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]
    quality = [classify_generation_quality(a) for a in answers]

    valid_indices = [i for i, q in enumerate(quality) if q["is_valid"]]
    invalid_indices = [i for i, q in enumerate(quality) if not q["is_valid"]]

    valid_answers = [answers[i] for i in valid_indices]
    valid_combined = [combined[i] for i in valid_indices]

    n_total = len(answers)
    n_valid = len(valid_indices)
    n_invalid = len(invalid_indices)

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

        sent_ans_scores = {idx: float(score) for idx, score in zip(valid_indices, sent_ans["scores"])}
        sent_comb_scores = {idx: float(score) for idx, score in zip(valid_indices, sent_comb["scores"])}
        ppl_ans_scores = {idx: float(score) for idx, score in zip(valid_indices, ppl_ans["per_text_ppl"])}
        ppl_comb_scores = {idx: float(score) for idx, score in zip(valid_indices, ppl_comb["per_text_ppl"])}
    else:
        sent_ans = {"mean_negative": None, "negative_fraction": None, "scores": []}
        sent_comb = {"mean_negative": None, "negative_fraction": None, "scores": []}
        ppl_ans = {"mean_ppl": None, "per_text_ppl": []}
        ppl_comb = {"mean_ppl": None, "per_text_ppl": []}
        sent_ans_scores = {}
        sent_comb_scores = {}
        ppl_ans_scores = {}
        ppl_comb_scores = {}

    per_row: list[dict[str, Any]] = []
    for i in range(n_total):
        row = {
            "idx": i,
            "prompt": prompts[i],
            "answer": answers[i],
            "combined": combined[i],
            "is_valid": quality[i]["is_valid"],
            "has_bad_pattern": quality[i]["has_bad_pattern"],
            "has_repetition_loop": quality[i]["has_repetition_loop"],
            "is_empty": quality[i]["is_empty"],
            "sent_answer": sent_ans_scores.get(i),
            "sent_combined": sent_comb_scores.get(i),
            "ppl_answer": ppl_ans_scores.get(i),
            "ppl_combined": ppl_comb_scores.get(i),
        }
        per_row.append(row)

    metrics = {
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
        "sent_answer_mean": sent_ans["mean_negative"],
        "sent_answer_fraction": sent_ans["negative_fraction"],
        "sent_combined_mean": sent_comb["mean_negative"],
        "ppl_answer_mean": ppl_ans["mean_ppl"],
        "ppl_combined_mean": ppl_comb["mean_ppl"],
        "mean_answer_words_all": float(sum(len(a.split()) for a in answers) / max(1, len(answers))),
        "mean_answer_words_valid_only": (
            float(sum(len(a.split()) for a in valid_answers) / max(1, len(valid_answers)))
            if n_valid > 0 else None
        ),
    }
    return metrics, per_row


# -----------------------------------------------------------------------------
# Main eval loop
# -----------------------------------------------------------------------------


def infer_variants(probe_root: Path) -> list[str]:
    variants: list[str] = []
    if not probe_root.exists():
        raise FileNotFoundError(f"Probe root not found: {probe_root}")
    for d in sorted(probe_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "mean_diff_vectors.pt").exists():
            variants.append(d.name)
    if not variants:
        raise RuntimeError(f"No variant directories with mean_diff_vectors.pt found under {probe_root}")
    return variants



def run_eval_for_variant(
    variant: str,
    prompts: list[str],
    model,
    tokenizer,
    probe_root: Path,
    eval_root: Path,
) -> dict[str, Any]:
    variant_eval_dir = eval_root / variant
    variant_eval_dir.mkdir(parents=True, exist_ok=True)

    variant_summary: dict[str, Any] = {}

    mean_diff = torch.load(
        probe_root / variant / "mean_diff_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )
    mean_vectors = {int(k): v for k, v in mean_diff.items() if int(k) in STEER_LAYERS}
    mean_steerer = PrecomputedLayerSteering(vectors=mean_vectors, layer_ids=STEER_LAYERS, alpha=MEAN_ALPHA)
    mean_answers = generate_with_steering(model, tokenizer, prompts, mean_steerer, f"mean_diff[{variant}]")
    mean_metrics, mean_rows = evaluate_texts_filtered(prompts, mean_answers)
    mean_metrics["method"] = "mean_diff"
    mean_metrics["variant"] = variant
    _write_json(variant_eval_dir / "mean_diff_metrics_valid_only.json", mean_metrics)
    _write_jsonl(variant_eval_dir / "mean_diff_generations_valid_only.jsonl", mean_rows)
    variant_summary["mean_diff_eval_filtered"] = mean_metrics

    probe_by_layer = load_probe_params_for_variant(probe_root, variant, STEER_LAYERS)
    liseco_runs: dict[str, Any] = {}
    for alpha_min, alpha_max in LISECO_INTERVALS:
        tag = f"amin{alpha_min:.2f}_amax{alpha_max:.2f}"
        steerer = LiSeCoProbeSteering(
            probe_by_layer=probe_by_layer,
            layer_ids=STEER_LAYERS,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mask_id=MASK_ID,
        )
        answers = generate_with_steering(
            model,
            tokenizer,
            prompts,
            steerer,
            f"liseco[{variant} {alpha_min:.2f}-{alpha_max:.2f}]",
        )
        metrics, rows = evaluate_texts_filtered(prompts, answers)
        metrics["method"] = "liseco"
        metrics["variant"] = variant
        metrics["alpha_min"] = alpha_min
        metrics["alpha_max"] = alpha_max
        _write_json(variant_eval_dir / f"liseco_{tag}_metrics_valid_only.json", metrics)
        _write_jsonl(variant_eval_dir / f"liseco_{tag}_generations_valid_only.jsonl", rows)
        liseco_runs[tag] = metrics
    _write_json(variant_eval_dir / "liseco_metrics_valid_only.json", liseco_runs)
    variant_summary["liseco_eval_filtered"] = liseco_runs

    return variant_summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/alignment_variants_v4/full_run"),
        help="Existing run root containing probes/ and alignment/ etc.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional explicit variant list. Defaults to all variants under probes/.",
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

    probe_root = args.out_root / "probes"
    eval_root = args.out_root / "eval_filtered_valid_only"
    eval_root.mkdir(parents=True, exist_ok=True)

    variants = args.variants if args.variants else infer_variants(probe_root)

    prompts, raw_eval_texts, eval_labels = load_eval_prompts()
    eval_set_payload = {
        "n_prompts": len(prompts),
        "prompt_words": PROMPT_WORDS,
        "eval_neg": EVAL_NEG,
        "eval_pos": EVAL_POS,
        "eval_any": EVAL_ANY,
        "labels": eval_labels,
        "prompts": prompts,
        "raw_eval_texts": raw_eval_texts,
    }
    _write_json(eval_root / "eval_set.json", eval_set_payload)

    print(f"[model] loading {args.model_name} on {DEVICE} ...")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)

    summary: dict[str, Any] = {
        "out_root": str(args.out_root),
        "probe_root": str(probe_root),
        "eval_root": str(eval_root),
        "model_name": args.model_name,
        "device": DEVICE,
        "evaluation_subset": "valid_generations_only",
        "bad_patterns": BAD_PATTERNS,
        "variants": {},
    }

    for variant in variants:
        print(f"\n[eval] variant={variant}")
        summary["variants"][variant] = run_eval_for_variant(
            variant=variant,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            probe_root=probe_root,
            eval_root=eval_root,
        )

    _write_json(eval_root / "summary_valid_only.json", summary)
    print(f"\n[done] wrote filtered eval summary to {eval_root / 'summary_valid_only.json'}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()