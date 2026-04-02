from __future__ import annotations

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.fluency_metrics import compute_perplexity
from eval.sentiment_metrics import compute_sentiment_metrics
from fill_scoring.h3_sentiment_grad_fill import SentimentGradientFillScorer
from llada.generate import generate as llada_generate
from llada.model import load_model, load_tokenizer
from steering.precomputed_steering import PrecomputedLayerSteering


SMOKE_TEST = os.getenv("SMOKE_TEST", "0") == "1"

DIFFMEAN_OUT_ROOT = Path("data/liseco_vs_diffmean_alignment/full_run")
DIFFMEAN_VARIANT = "real_full_pooled"
OUT_DIR = Path("data/gradient_filling") / ("smoke_test" if SMOKE_TEST else "full_run")

MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
SENTIMENT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Keep attribution classifier off GPU by default to reduce OOM risk.
SENTIMENT_DEVICE = os.getenv("SENTIMENT_DEVICE", "cpu")
SEED = 42

STEER_LAYERS = list(range(1, 31))
DIFFMEAN_LAMBDAS = [0.0, 10.0, 20.0]
ALPHA_MIX_VALUES = [0.9, 0.8, 0.6]

SMOKE_LAMBDAS = [0.0]
SMOKE_ALPHA_MIX = [0.9, 0.6]
SMOKE_MAX_PROMPTS = 3

EVAL_START_IDX = 50_000
EVAL_END_IDX = 50_250
PROMPT_WORDS = 20
N_DEBUG_EXAMPLES = 2

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
    path.write_text(json.dumps(payload, indent=2))


def has_bad_pattern(text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in BAD_PATTERNS)


def has_repetition_loop(text: str) -> bool:
    toks = text.split()
    if len(toks) < 12:
        return False
    for n in [2, 3, 4]:
        for i in range(len(toks) - 2 * n + 1):
            if toks[i : i + n] == toks[i + n : i + 2 * n]:
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
    }
    return prompts, labels, meta


def load_mean_vectors(probe_root: Path, variant: str, layer_ids: list[int]) -> dict[int, torch.Tensor]:
    candidate_paths = [
        probe_root / variant / "mean_diff_vectors.pt",
        Path("data/liseco_vs_diffmean_alignment/full_run/probes") / variant / "mean_diff_vectors.pt",
        Path("data/alignment_variants_v4/full_run/probes") / variant / "mean_diff_vectors.pt",
    ]
    p = next((cp for cp in candidate_paths if cp.exists()), None)
    if p is None:
        raise FileNotFoundError(
            "Could not find mean_diff_vectors.pt for variant "
            f"{variant}. Tried: {[str(cp) for cp in candidate_paths]}"
        )

    obj = torch.load(p, map_location="cpu", weights_only=False)
    vectors: dict[int, torch.Tensor] = {}
    for k, v in obj.items():
        li = int(k)
        if li in layer_ids:
            vectors[li] = v.float().cpu()

    missing = [li for li in layer_ids if li not in vectors]
    if missing:
        raise RuntimeError(f"Missing DiffMean vectors for layers={missing} in {p}")
    return vectors


def evaluate_texts_filtered(prompts: list[str], answers: list[str]) -> dict[str, Any]:
    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]
    quality = [classify_generation_quality(a) for a in answers]
    valid_idx = [i for i, q in enumerate(quality) if q["is_valid"]]

    valid_answers = [answers[i] for i in valid_idx]
    valid_combined = [combined[i] for i in valid_idx]

    n_total = len(answers)
    n_valid = len(valid_idx)
    n_invalid = n_total - n_valid

    if n_valid > 0:
        # Keep sentiment model off GPU so generation does not OOM between runs.
        sent_ans = compute_sentiment_metrics(valid_answers, device="cpu")
        ppl_ans = compute_perplexity(valid_answers, device="cpu")

        sent_mean = sent_ans["mean_negative"]
        ppl_mean = ppl_ans["mean_ppl"]
    else:
        sent_mean = None
        ppl_mean = None

    bad_pattern_count = int(sum(q["has_bad_pattern"] for q in quality))
    repetition_count = int(sum(q["has_repetition_loop"] for q in quality))
    empty_count = int(sum(q["is_empty"] for q in quality))

    metrics = {
        "evaluation_subset": "valid_generations_only",
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_fraction": float(n_invalid / max(1, n_total)),
        "invalid_percent": float(100.0 * n_invalid / max(1, n_total)),
        "bad_pattern_count": bad_pattern_count,
        "bad_pattern_percent": float(100.0 * bad_pattern_count / max(1, n_total)),
        "sent_answer_mean": sent_mean,
        "ppl_answer_mean": ppl_mean,
        "mean_answer_words_all": float(sum(len(a.split()) for a in answers) / max(1, len(answers))),
        "n_valid_combined": len(valid_combined),
        # Additional diagnostics kept for parity with prior H2/SAE runs.
        "repetition_count": repetition_count,
        "repetition_fraction": float(repetition_count / max(1, n_total)),
        "empty_count": empty_count,
        "empty_fraction": float(empty_count / max(1, n_total)),
    }
    return {"metrics": metrics, "quality": quality}


def generate_h3_grad(
    *,
    model,
    tokenizer,
    prompts: list[str],
    steering_vectors_by_layer: dict[int, torch.Tensor],
    alpha_mix: float,
    sentiment_tokenizer,
    sentiment_model,
    enable_debug: bool,
    desc: str,
):
    answers: list[str] = []
    debug_all: list[list[dict]] = []

    for i, prompt in enumerate(tqdm(prompts, desc=desc, leave=False)):
        enc = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)

        steerer = PrecomputedLayerSteering(
            vectors=steering_vectors_by_layer,
            layer_ids=STEER_LAYERS,
            alpha=1.0,
        )

        debug_this_example = enable_debug and i < N_DEBUG_EXAMPLES
        fill_scorer = SentimentGradientFillScorer(
            llada_tokenizer=tokenizer,
            alpha_mix=alpha_mix,
            sentiment_tokenizer=sentiment_tokenizer,
            sentiment_model=sentiment_model,
            sentiment_device=SENTIMENT_DEVICE,
            enable_debug=debug_this_example,
        )

        out = llada_generate(
            model=model,
            prompt=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            steps=GEN_PARAMS["steps"],
            gen_length=GEN_PARAMS["gen_length"],
            block_length=GEN_PARAMS["block_length"],
            temperature=GEN_PARAMS["temperature"],
            fill_strategy=GEN_PARAMS["fill_strategy"],
            fill_scorer=fill_scorer,
            remasking=None,
            steering=steerer,
            show_progress=False,
        )

        answer_ids = out[:, enc["input_ids"].shape[1] :]
        answers.extend(tokenizer.batch_decode(answer_ids, skip_special_tokens=True))

        if debug_this_example:
            debug_all.append(getattr(fill_scorer, "debug_records", []))
        else:
            debug_all.append([])

        del answer_ids
        del out
        del enc
        del fill_scorer
        del steerer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return answers, debug_all


def save_run(
    prompts: list[str],
    answers: list[str],
    debug_all: list[list[dict]],
    run_tag: str,
    run_info: dict[str, Any],
) -> dict[str, Any]:
    out_dir = OUT_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_payload = evaluate_texts_filtered(prompts, answers)
    metrics = eval_payload["metrics"]
    quality = eval_payload["quality"]

    with (out_dir / "generations.jsonl").open("w") as f:
        for i in range(len(prompts)):
            f.write(
                json.dumps(
                    {
                        "prompt": prompts[i],
                        "answer": answers[i],
                        "is_valid": quality[i]["is_valid"],
                        "has_bad_pattern": quality[i]["has_bad_pattern"],
                        "has_repetition_loop": quality[i]["has_repetition_loop"],
                        "is_empty": quality[i]["is_empty"],
                    }
                )
                + "\n"
            )

    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "run_info.json", run_info)

    for ei, records in enumerate(debug_all):
        if not records:
            continue
        _write_json(
            out_dir / f"debug_example_{ei:03d}.json",
            {
                "example_index": ei,
                "prompt": prompts[ei],
                "answer": answers[ei],
                "diffmean_lambda": run_info["diffmean_lambda"],
                "alpha_mix": run_info["fill_scoring"]["alpha_mix"],
                "fill_steps": records,
            },
        )

    print(
        f"  [done] invalid={metrics['invalid_percent']:.1f}% "
        f"valid={metrics['n_valid']}/{metrics['n_total']} "
        f"sent_valid={metrics['sent_answer_mean']} "
        f"ppl_valid={metrics['ppl_answer_mean']}"
    )
    return metrics


def main() -> None:
    _seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lambdas = SMOKE_LAMBDAS if SMOKE_TEST else DIFFMEAN_LAMBDAS
    alpha_sweep = SMOKE_ALPHA_MIX if SMOKE_TEST else ALPHA_MIX_VALUES

    prompts, labels, eval_meta = load_holdout_prompts(
        start_idx=EVAL_START_IDX,
        end_idx=EVAL_END_IDX,
        prompt_words=PROMPT_WORDS,
    )
    if SMOKE_TEST:
        prompts = prompts[:SMOKE_MAX_PROMPTS]
        labels = labels[:SMOKE_MAX_PROMPTS]

    _write_json(
        OUT_DIR / "eval_set.json",
        {
            "meta": eval_meta,
            "labels": labels,
            "prompts": prompts,
            "smoke_test": SMOKE_TEST,
        },
    )

    probe_root = DIFFMEAN_OUT_ROOT / "probes"
    base_vectors = load_mean_vectors(probe_root, DIFFMEAN_VARIANT, STEER_LAYERS)

    print(f"[model] loading {MODEL_NAME} on {DEVICE}")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    print(f"[sentiment] loading {SENTIMENT_MODEL_NAME} on {SENTIMENT_DEVICE}")
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME, use_fast=True)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_model.to(torch.device(SENTIMENT_DEVICE))
    sentiment_model.eval()

    summary: dict[str, Any] = {
        "mode": "SMOKE" if SMOKE_TEST else "FULL",
        "model": MODEL_NAME,
        "device": DEVICE,
        "sentiment_model": SENTIMENT_MODEL_NAME,
        "sentiment_device": SENTIMENT_DEVICE,
        "diffmean_variant": DIFFMEAN_VARIANT,
        "steer_layers": STEER_LAYERS,
        "lambdas": lambdas,
        "alpha_mix": alpha_sweep,
        "fill_scoring": {
            "heuristic": "H3_sentiment_gradient_fill_rank",
            "score_scalar": "negative_logit",
            "score_direction": "higher gradient norm means higher commit desirability",
            "normalization": "per-step minmax over eligible candidates",
            "token_mapping": "character-span overlap between classifier offsets and LLaDA candidate spans",
        },
        "results": {},
    }

    for lam in lambdas:
        scaled_vectors = {li: lam * v for li, v in base_vectors.items()}
        for alpha_mix in alpha_sweep:
            run_tag = f"h3grad__{DIFFMEAN_VARIANT}__lam{lam:.1f}__mix{alpha_mix:.2f}"
            print(f"[run] {run_tag}")

            answers, debug_all = generate_h3_grad(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                steering_vectors_by_layer=scaled_vectors,
                alpha_mix=alpha_mix,
                sentiment_tokenizer=sentiment_tokenizer,
                sentiment_model=sentiment_model,
                enable_debug=SMOKE_TEST,
                desc=f"H3Grad lam={lam:.1f} mix={alpha_mix:.2f}",
            )

            run_info = {
                "model": MODEL_NAME,
                "device": DEVICE,
                "diffmean_variant": DIFFMEAN_VARIANT,
                "diffmean_lambda": lam,
                "fill_scoring": {
                    "heuristic": "H3_sentiment_gradient_fill_rank",
                    "alpha_mix": alpha_mix,
                    "fill_strategy": GEN_PARAMS["fill_strategy"],
                    "classifier_model": SENTIMENT_MODEL_NAME,
                    "classifier_scalar": "negative_logit",
                    "normalization": "minmax over eligible candidates",
                },
                "n_prompts": len(prompts),
                "smoke_test": SMOKE_TEST,
            }

            metrics = save_run(prompts, answers, debug_all, run_tag, run_info)
            summary["results"][run_tag] = metrics

    _write_json(OUT_DIR / "summary.json", summary)

    del sentiment_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
