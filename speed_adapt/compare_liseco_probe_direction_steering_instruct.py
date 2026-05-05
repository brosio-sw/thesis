from __future__ import annotations

"""
compare_liseco_probe_direction_steering_instruct.py

Run LLaDA-Instruct masked-only steering using LiSeCo probe-derived directions.

This script does NOT run LiSeCo live during generation.
It only loads precomputed per-layer vectors and plugs them into
MeanActivationSteeringMaskedOnly.
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
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.fluency_metrics import compute_perplexity
from eval.sentiment_metrics import compute_sentiment_metrics
from llada.model import load_model, load_tokenizer
from steering.sus_fix import MeanActivationSteeringMaskedOnly


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
SENTIMENT_DEVICE = os.getenv("SENTIMENT_DEVICE", "cpu")

PROMPT_WORDS = 25
STEER_LAYERS = list(range(9, 26))
MASK_ID = 126336

ALPHAS = [0.0, 2.0, 4.0, 6.0, 8.0]

SMOKE_TEST = False
SMOKE_MAX_PROMPTS = 4
SMOKE_ALPHAS = [0.0, 4.0]

EVAL_START_IDX = 50_000
EVAL_END_IDX = 50_200
MIN_SOURCE_WORDS = 75

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=50,
    block_length=50,
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
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def format_prompt(raw_prompt: str) -> str:
    return (
        "Continue the following product review naturally.\n"
        "Keep the same topic as the original text.\n"
        "Write 2-4 complete sentences.\n"
        "Do not add headings, labels, or commentary.\n\n"
        f"Review: {raw_prompt}"
    )


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


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits

    logits = logits.float()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _build_default_fill_scores(
    *,
    logits: torch.FloatTensor,
    x0: torch.LongTensor,
    fill_strategy: str,
    device: torch.device,
) -> torch.FloatTensor:
    if fill_strategy == "low_confidence":
        probs = F.softmax(logits.float(), dim=-1)
        return probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    if fill_strategy == "random":
        return torch.rand(x0.shape, device=device)

    raise ValueError(
        f"Unknown fill_strategy {fill_strategy!r}. "
        "Expected 'low_confidence' or 'random'."
    )


def load_holdout_prompts(
    start_idx: int = EVAL_START_IDX,
    end_idx: int = EVAL_END_IDX,
    prompt_words: int = PROMPT_WORDS,
) -> tuple[list[str], list[int], dict[str, Any]]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="train")

    prompts: list[str] = []
    labels: list[int] = []

    target_count = max(1, end_idx - start_idx)
    seen_after_start = 0
    for ex_i, ex in enumerate(ds):
        if ex_i < start_idx:
            continue

        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        if len(text.split()) <= MIN_SOURCE_WORDS:
            continue

        prompt = _truncate(text, prompt_words)
        if len(prompt.split()) < 5:
            continue
        prompts.append(prompt)
        labels.append(int(ex["label"]))
        seen_after_start += 1
        if seen_after_start >= target_count:
            break

    meta = {
        "dataset": "fancyzhx/amazon_polarity",
        "split": "train",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "selection_rule": f"first {target_count} examples after start_idx with > {MIN_SOURCE_WORDS} words",
        "prompt_words": prompt_words,
        "n_prompts": len(prompts),
        "label_counts": {
            "negative_0": int(sum(1 for x in labels if x == 0)),
            "positive_1": int(sum(1 for x in labels if x == 1)),
        },
    }
    return prompts, labels, meta


def ensure_encoder_layer_compat(model) -> None:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        class _EncoderCompat:
            pass

        compat = _EncoderCompat()
        compat.layer = model.model.layers
        model.encoder = compat
        return

    if (
        hasattr(model, "model")
        and hasattr(model.model, "transformer")
        and hasattr(model.model.transformer, "__contains__")
        and "blocks" in model.model.transformer
    ):
        class _EncoderCompat:
            pass

        compat = _EncoderCompat()
        compat.layer = model.model.transformer["blocks"]
        model.encoder = compat
        return

    raise RuntimeError("Could not find transformer layers for steering hook registration")


def load_probe_directions(
    vectors_path: Path,
    layer_ids: list[int],
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing vectors file: {vectors_path}")

    obj = torch.load(vectors_path, map_location="cpu", weights_only=False)
    vectors = {int(k): v.float().cpu() for k, v in obj.items() if int(k) in layer_ids}

    missing = [li for li in layer_ids if li not in vectors]
    if not vectors:
        raise RuntimeError("No requested layers found in vectors file")

    norms = {str(li): float(torch.norm(vectors[li]).item()) for li in sorted(vectors.keys())}
    meta = {
        "source": "probe_directions",
        "vectors_path": str(vectors_path),
        "requested_layers": layer_ids,
        "loaded_layers": sorted(vectors.keys()),
        "missing_layers": missing,
        "layer_norms": norms,
    }
    return vectors, meta


def build_mask_only_steerer(model, tokenizer, alpha: float, vectors: dict[int, torch.Tensor]):
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = MASK_ID

    steerer = MeanActivationSteeringMaskedOnly(
        layer_ids=STEER_LAYERS,
        alpha=float(alpha),
        token_average=True,
        mask_id=int(mask_id),
    )

    model_dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
    if model_dtype is None:
        model_dtype = torch.float32

    steerer.layer_ids = [li for li in STEER_LAYERS if li in vectors]
    steerer.vectors = {
        li: vectors[li].to(dtype=model_dtype, device="cpu")
        for li in steerer.layer_ids
    }

    if not steerer.layer_ids:
        raise RuntimeError("No steering layers available after filtering vectors")

    return steerer


def generate_with_probe_direction_steering(
    *,
    model,
    tokenizer,
    prompt: str,
    steerer,
    enable_debug: bool,
) -> tuple[str, int, list[dict[str, Any]]]:
    formatted_prompt = format_prompt(prompt)
    enc = tokenizer([formatted_prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    gen_length = GEN_PARAMS["gen_length"]
    block_length = GEN_PARAMS["block_length"]
    temperature = GEN_PARAMS["temperature"]
    fill_strategy = GEN_PARAMS["fill_strategy"]

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")

    batch_size, prompt_len = input_ids.shape
    num_blocks = gen_length // block_length

    raw_mask_id = getattr(tokenizer, "mask_token_id", None)
    mask_id = int(raw_mask_id if raw_mask_id is not None else MASK_ID)

    x = torch.full(
        (batch_size, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    x[:, :prompt_len] = input_ids.clone()

    attention_mask_full = torch.cat(
        [
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=attention_mask.device),
        ],
        dim=-1,
    )

    debug_steps: list[dict[str, Any]] = []
    total_denoising_steps = 0

    with torch.inference_mode():
        steerer.register_hooks(model)
        try:
            for block_idx in range(num_blocks):
                block_start = prompt_len + block_idx * block_length
                block_end = prompt_len + (block_idx + 1) * block_length

                while bool((x[:, block_start:block_end] == mask_id).any().item()):
                    total_denoising_steps += 1

                    mask_index = x == mask_id
                    logits = model(x, attention_mask=attention_mask_full).logits.to(x.device)
                    logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
                    x0 = logits_noisy.argmax(dim=-1)

                    x0 = torch.where(mask_index, x0, x)

                    baseline_scores = _build_default_fill_scores(
                        logits=logits,
                        x0=x0,
                        fill_strategy=fill_strategy,
                        device=x.device,
                    )
                    baseline_scores[:, block_end:] = float("-inf")
                    baseline_scores = torch.where(
                        mask_index,
                        baseline_scores,
                        torch.full_like(baseline_scores, float("-inf")),
                    )

                    transfer_index = torch.zeros_like(x, dtype=torch.bool)
                    chosen_u_t = 0
                    for b in range(batch_size):
                        block_mask = x[b, block_start:block_end] == mask_id
                        M_t = int(block_mask.sum().item())
                        if M_t <= 0:
                            continue

                        u_t = 1
                        _, selected_pos = torch.topk(baseline_scores[b], k=u_t)
                        transfer_index[b, selected_pos] = True
                        chosen_u_t = u_t

                    x[transfer_index] = x0[transfer_index]

                    if enable_debug:
                        remaining_after = int((x[:, block_start:block_end] == mask_id).sum().item())
                        debug_steps.append(
                            {
                                "global_step": total_denoising_steps,
                                "block_idx": block_idx,
                                "u_t": int(chosen_u_t),
                                "remaining_masks_after_step": remaining_after,
                            }
                        )
        finally:
            steerer.remove_hooks()

    answer_ids = x[:, prompt_len:]
    answer = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
    return answer, total_denoising_steps, debug_steps


def evaluate_texts_filtered(
    prompts: list[str],
    answers: list[str],
    denoising_steps_per_generation: list[int],
) -> dict[str, Any]:
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
        sent_ans = compute_sentiment_metrics(valid_answers, device=SENTIMENT_DEVICE)
        sent_comb = compute_sentiment_metrics(valid_combined, device=SENTIMENT_DEVICE)
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
        "denoising_steps_per_generation": denoising_steps_per_generation,
        "total_denoising_steps": int(sum(denoising_steps_per_generation)),
        "avg_denoising_steps_per_generation": float(np.mean(denoising_steps_per_generation)) if denoising_steps_per_generation else None,
    }


def run_single_alpha(
    *,
    alpha: float,
    prompts: list[str],
    model,
    tokenizer,
    vectors: dict[int, torch.Tensor],
    out_dir: Path,
    enable_debug: bool,
) -> dict[str, Any]:
    run_tag = f"a{alpha:.1f}"
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    steerer = build_mask_only_steerer(model, tokenizer, alpha=alpha, vectors=vectors)

    answers: list[str] = []
    denoising_steps: list[int] = []
    debug_all: list[list[dict[str, Any]]] = []

    for prompt in tqdm(prompts, desc=f"run[a={alpha:.1f}]", leave=False):
        answer, n_steps, dbg = generate_with_probe_direction_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steerer=steerer,
            enable_debug=enable_debug,
        )
        answers.append(answer)
        denoising_steps.append(n_steps)
        debug_all.append(dbg)

    metrics = evaluate_texts_filtered(prompts, answers, denoising_steps)
    metrics["method"] = "liseco_probe_direction_masked_only"
    metrics["alpha"] = float(alpha)

    combined_answers = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]
    quality = [classify_generation_quality(a) for a in answers]
    with (run_dir / "generations.jsonl").open("w") as f:
        for i in range(len(prompts)):
            formatted_prompt = format_prompt(prompts[i])
            f.write(
                json.dumps(
                    {
                        "prompt": prompts[i],
                        "formatted_prompt": formatted_prompt,
                        "answer": answers[i],
                        "combined_answer": combined_answers[i],
                        "combined_answer_formatted_prompt": f"{formatted_prompt} {answers[i]}".strip(),
                        "denoising_steps": denoising_steps[i],
                        "is_valid": quality[i]["is_valid"],
                        "has_bad_pattern": quality[i]["has_bad_pattern"],
                        "has_repetition_loop": quality[i]["has_repetition_loop"],
                        "is_empty": quality[i]["is_empty"],
                    }
                )
                + "\n"
            )

    _write_json(run_dir / "metrics.json", metrics)
    _write_json(
        run_dir / "run_info.json",
        {
            "model": MODEL_NAME,
            "device": DEVICE,
            "sentiment_device": SENTIMENT_DEVICE,
            "alpha": float(alpha),
            "method": "liseco_probe_direction_masked_only",
            "runtime_liseco": False,
            "gen_params": GEN_PARAMS,
            "steer_layers_requested": STEER_LAYERS,
            "steer_layers_active": [li for li in STEER_LAYERS if li in vectors],
            "prompt_words": PROMPT_WORDS,
            "n_prompts": len(prompts),
        },
    )

    if enable_debug:
        for i, dbg in enumerate(debug_all):
            if not dbg:
                continue
            _write_json(run_dir / f"debug_example_{i:03d}.json", {"steps": dbg})

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--steering-vectors-path",
        type=Path,
        default=Path("data/speed_adapt/liseco_probe_directions_instruct/full_run/steering_vectors.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/speed_adapt/compare_liseco_probe_direction_steering_instruct"),
    )
    parser.add_argument("--alphas", nargs="*", type=float, default=ALPHAS)
    parser.add_argument("--start-idx", type=int, default=EVAL_START_IDX)
    parser.add_argument("--end-idx", type=int, default=EVAL_END_IDX)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-prompts", type=int, default=SMOKE_MAX_PROMPTS)
    parser.set_defaults(smoke_test=SMOKE_TEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(SEED)

    out_dir = args.out_dir / ("smoke_test" if args.smoke_test else "full_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, labels, eval_meta = load_holdout_prompts(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        prompt_words=PROMPT_WORDS,
    )
    if args.smoke_test:
        prompts = prompts[: args.smoke_max_prompts]
        labels = labels[: args.smoke_max_prompts]

    _write_json(
        out_dir / "eval_set.json",
        {
            "meta": eval_meta,
            "labels": labels,
            "prompts": prompts,
            "formatted_prompts": [format_prompt(p) for p in prompts],
            "smoke_test": bool(args.smoke_test),
        },
    )

    vectors, vector_meta = load_probe_directions(
        vectors_path=args.steering_vectors_path,
        layer_ids=STEER_LAYERS,
    )

    print(f"[model] loading {args.model_name} on {DEVICE}")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)
    ensure_encoder_layer_compat(model)

    alphas = SMOKE_ALPHAS if args.smoke_test else [float(a) for a in args.alphas]

    summary: dict[str, Any] = {
        "model_name": args.model_name,
        "device": DEVICE,
        "sentiment_device": SENTIMENT_DEVICE,
        "out_dir": str(out_dir),
        "vector_meta": vector_meta,
        "steer_layers_requested": STEER_LAYERS,
        "steer_layers_active": [li for li in STEER_LAYERS if li in vectors],
        "prompt_words": PROMPT_WORDS,
        "gen_params": GEN_PARAMS,
        "alphas": alphas,
        "evaluation_subset": "valid_generations_only",
        "bad_patterns": BAD_PATTERNS,
        "eval_meta": eval_meta,
        "smoke_test": bool(args.smoke_test),
        "runtime_liseco": False,
        "results": {},
    }

    aggregate_rows = [
        "method,alpha,n_total,n_valid,n_invalid,invalid_fraction,bad_pattern_fraction,repetition_fraction,empty_fraction,sent_answer_mean,sent_answer_fraction,sent_combined_mean,ppl_answer_mean,ppl_combined_mean,mean_answer_words_all,mean_answer_words_valid_only,total_denoising_steps,avg_denoising_steps_per_generation"
    ]

    for alpha in alphas:
        run_key = f"a{alpha:.1f}"
        print(f"[run] alpha={alpha:.1f}")
        metrics = run_single_alpha(
            alpha=float(alpha),
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            vectors=vectors,
            out_dir=out_dir,
            enable_debug=bool(args.smoke_test),
        )
        summary["results"][run_key] = metrics

        aggregate_rows.append(
            ",".join(
                [
                    str(metrics["method"]),
                    str(metrics["alpha"]),
                    str(metrics["n_total"]),
                    str(metrics["n_valid"]),
                    str(metrics["n_invalid"]),
                    str(metrics["invalid_fraction"]),
                    str(metrics["bad_pattern_fraction"]),
                    str(metrics["repetition_fraction"]),
                    str(metrics["empty_fraction"]),
                    str(metrics["sent_answer_mean"]),
                    str(metrics["sent_answer_fraction"]),
                    str(metrics["sent_combined_mean"]),
                    str(metrics["ppl_answer_mean"]),
                    str(metrics["ppl_combined_mean"]),
                    str(metrics["mean_answer_words_all"]),
                    str(metrics["mean_answer_words_valid_only"]),
                    str(metrics["total_denoising_steps"]),
                    str(metrics["avg_denoising_steps_per_generation"]),
                ]
            )
        )

    if summary["results"]:
        vals = [
            float(v["avg_denoising_steps_per_generation"])
            for v in summary["results"].values()
            if v.get("avg_denoising_steps_per_generation") is not None
        ]
        summary["overall_avg_denoising_steps_per_generation"] = float(np.mean(vals)) if vals else None

    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.csv").write_text("\n".join(aggregate_rows) + "\n")
    print(f"[done] wrote outputs under {out_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
