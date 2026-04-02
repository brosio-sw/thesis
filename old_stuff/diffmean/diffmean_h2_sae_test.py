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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity
from fill_scoring.h2_sae_fill import H2SAEAwareFillScorer
from steering.precomputed_steering import (
    HiddenStateCaptureSteering,
    SteeringHiddenStateBuffer,
)
from remasking.h2_sae_conf import LayerwiseTopKSAEReconstructor


# =============================================================================
# CONFIG
# =============================================================================

SMOKE_TEST: bool = False

DIFFMEAN_OUT_ROOT = Path("data/alignment_variants_v4/full_run")
DIFFMEAN_VARIANT = "real_full_pooled"
OUT_DIR = Path("data/diffmean_h2_sae") / (
    "smoke_test_all_layers" if SMOKE_TEST else "full_run_all_layers"
)

MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

STEER_LAYERS = list(range(1, 31))  # all layers for LLaDA-8B
DIFFMEAN_LAMBDAS = [4.0, 12.0, 20.0]

MASK_SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
PUBLIC_SAE_LAYERS = [1, 6, 11, 16, 26, 30]
MONITORED_SAE_LAYERS = [1,6,11, 16, 26, 30]
SAE_TRAINER_IDX = 0

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",  # base LLaDA commit policy
)

ALL_ALPHA_MIX = [1.0, 0.875, 0.75, 0.5, 0.25, 0.0]

EVAL_START_IDX = 50_000
EVAL_END_IDX = 50_100
PROMPT_WORDS = 20

N_DEBUG_EXAMPLES = 2

SMOKE_LAMBDAS = [4.0]
SMOKE_ALPHA_MIX = [1.0, 0.0]
SMOKE_MAX_PROMPTS = 5

BAD_PATTERNS = [
    r"Is this .*?\?",
    r"positive or negative\?",
    r"\bAnswer:",
    r"\bYesTitle:",
    r"\bNoTitle:",
    r"\bTitle:",
    r"\bReview:",
]


# =============================================================================
# Helpers
# =============================================================================

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


def load_mean_vectors(probe_root: Path, variant: str, layer_ids: list[int]) -> dict[int, torch.Tensor]:
    p = probe_root / variant / "mean_diff_vectors.pt"
    obj = torch.load(p, map_location="cpu", weights_only=False)
    vectors: dict[int, torch.Tensor] = {}
    for k, v in obj.items():
        li = int(k)
        if li in layer_ids:
            vectors[li] = v.float().cpu()

    missing_layers = [li for li in layer_ids if li not in vectors]
    if missing_layers:
        raise RuntimeError(
            f"Missing DiffMean vectors for layers={missing_layers} in {p}. "
            f"Available layers: {sorted(int(k) for k in obj.keys())}"
        )
    return vectors


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

        sent_details = {
            "answer": {
                "mean_negative": sent_ans["mean_negative"],
                "negative_fraction": sent_ans["negative_fraction"],
                "per_text": sent_ans["scores"].tolist(),
            },
            "combined": {
                "mean_negative": sent_comb["mean_negative"],
                "negative_fraction": sent_comb["negative_fraction"],
                "per_text": sent_comb["scores"].tolist(),
            },
            "valid_indices": valid_indices,
        }
        ppl_details = {
            "answer": {
                "mean_ppl": ppl_ans["mean_ppl"],
                "per_text": ppl_ans["per_text_ppl"],
            },
            "combined": {
                "mean_ppl": ppl_comb["mean_ppl"],
                "per_text": ppl_comb["per_text_ppl"],
            },
            "valid_indices": valid_indices,
        }
    else:
        sent_answer_mean = None
        sent_answer_fraction = None
        sent_combined_mean = None
        ppl_answer_mean = None
        ppl_combined_mean = None
        mean_answer_words_valid_only = None

        sent_details = {"answer": None, "combined": None, "valid_indices": []}
        ppl_details = {"answer": None, "combined": None, "valid_indices": []}

    metrics = {
        "evaluation_subset": "valid_generations_only",
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_fraction": invalid_fraction,
        "invalid_percent": 100.0 * invalid_fraction,
        "bad_pattern_count": bad_pattern_count,
        "bad_pattern_fraction": bad_pattern_fraction,
        "bad_pattern_percent": 100.0 * bad_pattern_fraction,
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
    return {
        "metrics": metrics,
        "quality": quality,
        "valid_indices": valid_indices,
        "sentiment": sent_details,
        "perplexity": ppl_details,
    }


# =============================================================================
# Generation + Save
# =============================================================================

def generate_h2(
    *,
    model,
    tokenizer,
    prompts: list[str],
    steering_vectors_by_layer: dict[int, torch.Tensor],
    sae_reconstructor: LayerwiseTopKSAEReconstructor,
    monitored_layers: list[int],
    alpha_mix: float,
    enable_debug: bool = False,
    desc: str = "H2b",
):
    """
    Generate with:
      - base LLaDA fill policy = low_confidence
      - DiffMean steering (vectors must already be scaled)
      - H2 SAE-aware fill scoring
      - no remasking
    """
    answers: list[str] = []
    debug_all: list[list[dict]] = []

    for i, prompt in enumerate(tqdm(prompts, desc=desc, leave=False)):
        enc = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        hs_buffer = SteeringHiddenStateBuffer()

        steerer = HiddenStateCaptureSteering(
            vectors=steering_vectors_by_layer,
            layer_ids=STEER_LAYERS,
            alpha=1.0,  # vectors are already scaled outside
            capture_layer_ids=monitored_layers,
            hidden_state_buffer=hs_buffer,
        )

        debug_this_example = enable_debug and i < N_DEBUG_EXAMPLES

        fill_scorer = H2SAEAwareFillScorer(
            buffer=hs_buffer,
            sae_reconstructor=sae_reconstructor,
            monitored_layers=monitored_layers,
            alpha_mix=alpha_mix,
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

        answer_ids = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
        answers.extend(decoded)

        if debug_this_example:
            debug_all.append(getattr(fill_scorer, "debug_records", []))
        else:
            debug_all.append([])

        if torch.cuda.is_available() and (i + 1) % 10 == 0:
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

    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]

    with open(out_dir / "generations.jsonl", "w") as f:
        for i in range(len(prompts)):
            q = quality[i]
            f.write(json.dumps({
                "prompt": prompts[i],
                "answer": answers[i],
                "combined": combined[i],
                "is_valid": q["is_valid"],
                "has_bad_pattern": q["has_bad_pattern"],
                "has_repetition_loop": q["has_repetition_loop"],
                "is_empty": q["is_empty"],
            }) + "\n")

    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "run_info.json", run_info)
    _write_json(out_dir / "sentiment.json", eval_payload["sentiment"])
    _write_json(out_dir / "perplexity.json", eval_payload["perplexity"])

    for ei, records in enumerate(debug_all):
        if not records:
            continue
        _write_json(
            out_dir / f"debug_example_{ei:03d}.json",
            {
                "example_index": ei,
                "prompt": prompts[ei],
                "answer": answers[ei],
                "lambda": run_info["diffmean_lambda"],
                "alpha_mix": run_info["fill_scoring"]["alpha_mix"],
                "fill_steps": records,
            },
        )

    print(
        f"  [done] invalid={metrics['invalid_percent']:.1f}% "
        f"bad_pattern={metrics['bad_pattern_percent']:.1f}% "
        f"valid={metrics['n_valid']}/{metrics['n_total']} "
        f"sent_valid={metrics['sent_answer_mean']} "
        f"ppl_valid={metrics['ppl_answer_mean']}"
    )
    return metrics


def _compare_alpha_debug_runs(
    out_dir: Path,
    run_tag_a: str,
    run_tag_b: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_a": run_tag_a,
        "run_b": run_tag_b,
        "examples": [],
        "totals": {},
    }

    total_steps = 0
    same_eligible = 0
    same_chosen = 0
    eligible_size_hist: dict[str, int] = {}
    steps_eligible_ge2 = 0
    steps_sae_nonconstant = 0
    steps_priority_rank_diff = 0
    steps_low_vs_sae_rank_diff = 0

    for ei in range(N_DEBUG_EXAMPLES):
        p_a = dir_a = out_dir / run_tag_a / f"debug_example_{ei:03d}.json"
        p_b = dir_b = out_dir / run_tag_b / f"debug_example_{ei:03d}.json"
        if not p_a.exists() or not p_b.exists():
            continue

        d_a = json.loads(p_a.read_text())
        d_b = json.loads(p_b.read_text())
        s_a = d_a.get("fill_steps", [])
        s_b = d_b.get("fill_steps", [])

        if isinstance(s_a, dict):
            s_a = [s_a]
        if isinstance(s_b, dict):
            s_b = [s_b]

        n = min(len(s_a), len(s_b))

        ex_stats = {
            "example_index": ei,
            "n_steps_compared": n,
            "same_eligible_steps": 0,
            "same_chosen_steps": 0,
            "steps_eligible_ge2": 0,
            "steps_sae_nonconstant": 0,
            "steps_priority_rank_diff": 0,
            "steps_baseline_vs_sae_rank_diff": 0,
        }

        for si in range(n):
            ra = s_a[si]
            rb = s_b[si]
            total_steps += 1

            elig_a = ra.get("candidate_pos", [])
            elig_b = rb.get("candidate_pos", [])
            rem_a = ra.get("chosen_commit_pos", [])
            rem_b = rb.get("chosen_commit_pos", [])

            if elig_a == elig_b:
                same_eligible += 1
                ex_stats["same_eligible_steps"] += 1
            if rem_a == rem_b:
                same_chosen += 1
                ex_stats["same_chosen_steps"] += 1

            n_elig = len(elig_a)
            eligible_size_hist[str(n_elig)] = eligible_size_hist.get(str(n_elig), 0) + 1
            if n_elig >= 2:
                steps_eligible_ge2 += 1
                ex_stats["steps_eligible_ge2"] += 1

            sae_vals = rb.get("sae_fill_score", [])
            if len(sae_vals) >= 2 and (max(sae_vals) - min(sae_vals)) > 1e-8:
                steps_sae_nonconstant += 1
                ex_stats["steps_sae_nonconstant"] += 1

            low_vals = rb.get("baseline_fill_score", [])
            if len(low_vals) >= 2 and len(sae_vals) == len(low_vals):
                low_rank = sorted(range(len(low_vals)), key=lambda i: low_vals[i], reverse=True)
                sae_rank = sorted(range(len(sae_vals)), key=lambda i: sae_vals[i], reverse=True)
                if low_rank != sae_rank:
                    steps_low_vs_sae_rank_diff += 1
                    ex_stats["steps_baseline_vs_sae_rank_diff"] += 1

            pri_a = ra.get("mixed_fill_score", [])
            pri_b = rb.get("mixed_fill_score", [])
            if len(pri_a) >= 2 and len(pri_a) == len(pri_b):
                rank_a = sorted(range(len(pri_a)), key=lambda i: pri_a[i], reverse=True)
                rank_b = sorted(range(len(pri_b)), key=lambda i: pri_b[i], reverse=True)
                if rank_a != rank_b:
                    steps_priority_rank_diff += 1
                    ex_stats["steps_priority_rank_diff"] += 1

        summary["examples"].append(ex_stats)

    summary["totals"] = {
        "steps_compared": total_steps,
        "same_eligible_steps": same_eligible,
        "same_chosen_steps": same_chosen,
        "same_eligible_fraction": float(same_eligible / max(1, total_steps)),
        "same_chosen_fraction": float(same_chosen / max(1, total_steps)),
        "eligible_size_hist": eligible_size_hist,
        "steps_eligible_ge2": steps_eligible_ge2,
        "steps_sae_nonconstant": steps_sae_nonconstant,
        "steps_baseline_vs_sae_rank_diff": steps_low_vs_sae_rank_diff,
        "steps_priority_rank_diff": steps_priority_rank_diff,
    }
    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    _seed_everything(SEED)

    if not set(MONITORED_SAE_LAYERS).issubset(set(STEER_LAYERS)):
        raise ValueError(
            f"MONITORED_SAE_LAYERS={MONITORED_SAE_LAYERS} must be within STEER_LAYERS={STEER_LAYERS}"
        )
    if not set(MONITORED_SAE_LAYERS).issubset(set(PUBLIC_SAE_LAYERS)):
        raise ValueError(
            f"MONITORED_SAE_LAYERS={MONITORED_SAE_LAYERS} not all in PUBLIC_SAE_LAYERS={PUBLIC_SAE_LAYERS}"
        )

    lambdas = SMOKE_LAMBDAS if SMOKE_TEST else DIFFMEAN_LAMBDAS
    alpha_sweep = SMOKE_ALPHA_MIX if SMOKE_TEST else ALL_ALPHA_MIX
    mode = "SMOKE TEST" if SMOKE_TEST else "FULL RUN"

    print(f"\n{'='*78}")
    print(f"  DIFFMEAN + H2b SAE FILL-SCORING — {mode}")
    print(f"  Variant={DIFFMEAN_VARIANT}")
    print(f"  Steer layers={STEER_LAYERS[0]}..{STEER_LAYERS[-1]}  monitored SAE={MONITORED_SAE_LAYERS}")
    print(f"  Lambdas={lambdas}")
    print(f"  alpha_mix sweep={alpha_sweep}")
    print(f"  fill_strategy={GEN_PARAMS['fill_strategy']}  commit scoring=baseline+SAE")
    print(f"{'='*78}\n")

    probe_root = DIFFMEAN_OUT_ROOT / "probes"
    base_vectors = load_mean_vectors(
        probe_root=probe_root,
        variant=DIFFMEAN_VARIANT,
        layer_ids=STEER_LAYERS,
    )

    prompts, labels, eval_meta = load_holdout_prompts(
        start_idx=EVAL_START_IDX,
        end_idx=EVAL_END_IDX,
        prompt_words=PROMPT_WORDS,
    )
    if SMOKE_TEST:
        prompts = prompts[:SMOKE_MAX_PROMPTS]
        labels = labels[:SMOKE_MAX_PROMPTS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        OUT_DIR / "eval_set.json",
        {
            "meta": eval_meta,
            "labels": labels,
            "prompts": prompts,
            "smoke_test": SMOKE_TEST,
        },
    )

    print("[sae] Loading public LLaDA SAEs ...")
    sae_reconstructor = LayerwiseTopKSAEReconstructor(
        repo_id=MASK_SAE_REPO,
        layer_ids=MONITORED_SAE_LAYERS,
        trainer_idx=SAE_TRAINER_IDX,
    )
    print("[sae] Loaded SAE layers:", MONITORED_SAE_LAYERS)

    print(f"[model] loading {MODEL_NAME} on {DEVICE} ...")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    summary: dict[str, Any] = {
        "mode": mode,
        "model": MODEL_NAME,
        "device": DEVICE,
        "diffmean_variant": DIFFMEAN_VARIANT,
        "steer_layers": STEER_LAYERS,
        "lambdas": lambdas,
        "alpha_mix": alpha_sweep,
        "monitored_sae_layers": MONITORED_SAE_LAYERS,
        "mask_sae_repo": MASK_SAE_REPO,
        "sae_trainer_idx": SAE_TRAINER_IDX,
        "results": {},
    }

    total_cfgs = len(lambdas) * len(alpha_sweep)
    cfg_idx = 0

    for lam in lambdas:
        # IMPORTANT FIX: lambda is applied here.
        scaled_vectors = {li: lam * v for li, v in base_vectors.items()}

        for alpha_mix in alpha_sweep:
            cfg_idx += 1
            run_tag = f"h2b__{DIFFMEAN_VARIANT}__lam{lam:.1f}__mix{alpha_mix:.2f}"
            print(f"\n[{cfg_idx}/{total_cfgs}] {run_tag}")

            done_marker = OUT_DIR / run_tag / "metrics.json"
            if done_marker.exists():
                print(f"  [skip] Already done. Delete {done_marker} to rerun.")
                with open(done_marker) as f:
                    summary["results"][run_tag] = json.load(f)
                continue

            answers, debug_all = generate_h2(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                steering_vectors_by_layer=scaled_vectors,
                sae_reconstructor=sae_reconstructor,
                monitored_layers=MONITORED_SAE_LAYERS,
                alpha_mix=alpha_mix,
                enable_debug=SMOKE_TEST,
                desc=f"H2b λ={lam:.1f} mix={alpha_mix:.2f}",
            )

            run_info = {
                "model": MODEL_NAME,
                "device": DEVICE,
                "gen_params": GEN_PARAMS,
                "diffmean_variant": DIFFMEAN_VARIANT,
                "steer_layers": STEER_LAYERS,
                "diffmean_lambda": lam,
                "fill_scoring": {
                    "heuristic": "H2b_SAE_fill_rank",
                    "alpha_mix": alpha_mix,
                    "fill_strategy": GEN_PARAMS["fill_strategy"],
                    "candidate_scope": "currently_masked_generation_in_current_block",
                },
                "sae": {
                    "repo": MASK_SAE_REPO,
                    "trainer_idx": SAE_TRAINER_IDX,
                    "public_layers": PUBLIC_SAE_LAYERS,
                    "monitored_layers": MONITORED_SAE_LAYERS,
                    "aggregation": "per-layer minmax over fill candidates, then mean across layers",
                },
                "n_prompts": len(prompts),
                "smoke_test": SMOKE_TEST,
                "eval_meta": eval_meta,
                "out_root": str(DIFFMEAN_OUT_ROOT),
            }

            metrics = save_run(
                prompts=prompts,
                answers=answers,
                debug_all=debug_all,
                run_tag=run_tag,
                run_info=run_info,
            )
            summary["results"][run_tag] = metrics

    _write_json(OUT_DIR / "summary.json", summary)

    if SMOKE_TEST and 1.0 in alpha_sweep and 0.0 in alpha_sweep:
        for lam in lambdas:
            tag_a = f"h2b__{DIFFMEAN_VARIANT}__lam{lam:.1f}__mix1.00"
            tag_b = f"h2b__{DIFFMEAN_VARIANT}__lam{lam:.1f}__mix0.00"
            p_a = OUT_DIR / tag_a / "debug_example_000.json"
            p_b = OUT_DIR / tag_b / "debug_example_000.json"
            if not p_a.exists() or not p_b.exists():
                continue
            cmp = _compare_alpha_debug_runs(OUT_DIR, tag_a, tag_b)
            out_cmp = OUT_DIR / f"alpha_compare__lam{lam:.1f}.json"
            _write_json(out_cmp, cmp)
            t = cmp["totals"]
            print(
                f"[alpha-compare λ={lam:.1f}] "
                f"same_chosen={t['same_chosen_steps']}/{t['steps_compared']} "
                f"eligible>=2={t['steps_eligible_ge2']} "
                f"sae_nonconstant={t['steps_sae_nonconstant']} "
                f"priority_rank_diff={t['steps_priority_rank_diff']}"
            )
            print(f"[alpha-compare] saved -> {out_cmp}")

    print(f"\n{'='*84}")
    print(f"  H2b SUMMARY — {mode}")
    print(f"{'='*84}")
    hdr = f"{'run_tag':<60} {'invalid%':>8} {'bad%':>7} {'sent':>8} {'ppl':>8}"
    print(hdr)
    print("-" * 84)
    for tag, m in summary["results"].items():
        print(
            f"{tag:<60} "
            f"{m.get('invalid_percent', float('nan')):>8.2f} "
            f"{m.get('bad_pattern_percent', float('nan')):>7.2f} "
            f"{str(m.get('sent_answer_mean')):>8} "
            f"{str(m.get('ppl_answer_mean')):>8}"
        )

    print(f"\n[output] Summary -> {OUT_DIR / 'summary.json'}")

    if SMOKE_TEST:
        print(f"\n{'='*78}")
        print("  SMOKE TEST COMPLETE")
        print("  For full sweep:")
        print("    1. Set SMOKE_TEST = False in diffmean_h2_sae_test.py")
        print("    2. Run:")
        print("       /home/ambroise/miniconda3/envs/thesis/bin/python diffmean_h2_sae_test.py")
        print(f"{'='*78}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()