from __future__ import annotations

"""
debug_eos_blocking_test.py

Small non-destructive diagnostic for short visible outputs in LLaDA-Instruct
masked generation.

Conditions:
- baseline: no token banning
- eos_banned: EOS token banned at masked generation positions
- all_special_banned: all tokenizer special tokens banned at masked generation positions

Goal:
Measure whether EOS/special-token selection in generated positions is a major
cause of short visible continuations.
"""

import argparse
import gc
import json
import os
import random
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

from llada.model import load_model, load_tokenizer


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

PROMPT_WORDS = 25
MASK_ID = 126336

SMOKE_TEST = False
SMOKE_MAX_PROMPTS = 20

EVAL_START_IDX = 50_000
EVAL_END_IDX = 50_200
MIN_SOURCE_WORDS = 75

GEN_PARAMS = dict(
    temperature=0.0,
    gen_length=60,
    block_length=60,
    fill_strategy="low_confidence",
)


CONDITIONS = [
    "baseline",
    "eos_banned",
    "eos_last10_only",
    "all_special_banned",
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
        "Write around 10 complete and long sentences.\n"
        "Do not add headings, labels, or commentary.\n\n"
        f"Review: {raw_prompt}"
    )


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

    raise RuntimeError("Could not find transformer layers")


def _apply_banned_ids_to_logits(
    logits: torch.Tensor,
    ban_positions: torch.Tensor,
    banned_ids: list[int],
) -> torch.Tensor:
    if not banned_ids:
        return logits

    logits_mod = logits.clone()
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)

    # Ban selected token ids at explicit positions.
    for tok_id in banned_ids:
        logits_mod[..., tok_id] = torch.where(ban_positions, neg_inf, logits_mod[..., tok_id])

    return logits_mod


def _infer_eot_token_id(tokenizer) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for tok in ["<|eot_id|>", "<|eot|>", "<|end_of_turn|>"]:
            tid = convert(tok)
            if isinstance(tid, int) and tid >= 0:
                return int(tid)

    added = getattr(tokenizer, "added_tokens_encoder", None)
    if isinstance(added, dict):
        for tok in ["<|eot_id|>", "<|eot|>", "<|end_of_turn|>"]:
            tid = added.get(tok)
            if isinstance(tid, int) and tid >= 0:
                return int(tid)

    return None


def _generate_single_condition(
    *,
    model,
    tokenizer,
    prompt: str,
    condition: str,
    eos_id: int | None,
    eot_id: int | None,
    all_special_ids: list[int],
) -> dict[str, Any]:
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

    term_ids = sorted({
        int(t)
        for t in [eos_id, eot_id]
        if t is not None
    })

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

    with torch.inference_mode():
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length

            while bool((x[:, block_start:block_end] == mask_id).any().item()):
                mask_index = x == mask_id
                logits = model(x, attention_mask=attention_mask_full).logits.to(x.device)

                # Position-aware banning in generated region.
                ban_positions = torch.zeros_like(x, dtype=torch.bool)
                banned_ids: list[int] = []
                if condition == "baseline":
                    banned_ids = []
                elif condition == "eos_banned":
                    banned_ids = term_ids
                    ban_positions[:, block_start:block_end] = True
                elif condition == "eos_last10_only":
                    banned_ids = term_ids
                    # Ban EOS/EOT for first 50 generated positions, allow in last 10.
                    early_end = block_start + max(0, gen_length - 10)
                    ban_positions[:, block_start:early_end] = True
                elif condition == "all_special_banned":
                    banned_ids = sorted({int(x) for x in all_special_ids})
                    ban_positions[:, block_start:block_end] = True
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                logits = _apply_banned_ids_to_logits(
                    logits=logits,
                    ban_positions=ban_positions,
                    banned_ids=banned_ids,
                )

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
                for b in range(batch_size):
                    block_mask = x[b, block_start:block_end] == mask_id
                    M_t = int(block_mask.sum().item())
                    if M_t <= 0:
                        continue

                    u_t = 1
                    _, selected_pos = torch.topk(baseline_scores[b], k=u_t)
                    transfer_index[b, selected_pos] = True

                x[transfer_index] = x0[transfer_index]

    gen_ids = x[:, prompt_len:]
    token_ids = [int(t) for t in gen_ids[0].tolist()]

    answer_skip = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    answer_noskip = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]

    special_set = {int(t) for t in all_special_ids}
    eos_count = int(sum(1 for tid in token_ids if eos_id is not None and tid == int(eos_id)))
    eot_count = int(sum(1 for tid in token_ids if eot_id is not None and tid == int(eot_id)))

    non_special_count = int(sum(1 for tid in token_ids if tid not in special_set))
    special_count = int(len(token_ids) - non_special_count)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    mask_tok_id = getattr(tokenizer, "mask_token_id", None)

    pad_count = int(sum(1 for tid in token_ids if pad_id is not None and tid == int(pad_id)))
    bos_count = int(sum(1 for tid in token_ids if bos_id is not None and tid == int(bos_id)))
    mask_count = int(sum(1 for tid in token_ids if mask_tok_id is not None and tid == int(mask_tok_id)))

    answer_words = len(answer_skip.split())

    return {
        "condition": condition,
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "answer_skip_special_tokens_true": answer_skip,
        "answer_skip_special_tokens_false": answer_noskip,
        "generated_token_ids": token_ids,
        "generated_positions": int(len(token_ids)),
        "non_special_generated_token_count": non_special_count,
        "special_generated_token_count": special_count,
        "eos_count": eos_count,
        "eot_count": eot_count,
        "pad_count": pad_count,
        "bos_count": bos_count,
        "mask_count": mask_count,
        "is_empty": not answer_skip.strip(),
        "answer_word_count": int(answer_words),
        "banned_token_ids": banned_ids,
    }


def _aggregate_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n_examples": 0,
            "mean_visible_answer_words": None,
            "empty_fraction": None,
            "mean_non_special_generated_token_count": None,
            "total_eos_count": None,
            "avg_eos_count_per_generation": None,
            "total_eot_count": None,
            "avg_eot_count_per_generation": None,
        }

    mean_words = float(np.mean([r["answer_word_count"] for r in rows]))
    empty_fraction = float(np.mean([1.0 if r["is_empty"] else 0.0 for r in rows]))
    mean_non_special = float(np.mean([r["non_special_generated_token_count"] for r in rows]))
    total_eos = int(sum(int(r["eos_count"]) for r in rows))
    avg_eos = float(total_eos / n)
    total_eot = int(sum(int(r["eot_count"]) for r in rows))
    avg_eot = float(total_eot / n)

    total_special = int(sum(int(r["special_generated_token_count"]) for r in rows))
    avg_special = float(total_special / n)

    return {
        "n_examples": n,
        "mean_visible_answer_words": mean_words,
        "empty_fraction": empty_fraction,
        "mean_non_special_generated_token_count": mean_non_special,
        "total_eos_count": total_eos,
        "avg_eos_count_per_generation": avg_eos,
        "total_eot_count": total_eot,
        "avg_eot_count_per_generation": avg_eot,
        "total_special_generated_token_count": total_special,
        "avg_special_generated_token_count_per_generation": avg_special,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/speed_adapt/debug_eos_blocking_test"),
    )
    parser.add_argument("--start-idx", type=int, default=EVAL_START_IDX)
    parser.add_argument("--end-idx", type=int, default=EVAL_END_IDX)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-prompts", type=int, default=SMOKE_MAX_PROMPTS)
    parser.set_defaults(smoke_test=SMOKE_TEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(SEED)

    run_dir = args.out_dir / ("smoke_test" if args.smoke_test else "full_run")
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts, labels, eval_meta = load_holdout_prompts(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        prompt_words=PROMPT_WORDS,
    )

    if args.smoke_test:
        prompts = prompts[: args.smoke_max_prompts]
        labels = labels[: args.smoke_max_prompts]

    print(f"[model] loading {args.model_name} on {DEVICE}")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)
    ensure_encoder_layer_compat(model)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = _infer_eot_token_id(tokenizer)
    all_special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])

    _write_json(
        run_dir / "eval_set.json",
        {
            "meta": eval_meta,
            "smoke_test": bool(args.smoke_test),
            "n_prompts": len(prompts),
            "labels": labels,
            "prompts": prompts,
            "formatted_prompts": [format_prompt(p) for p in prompts],
            "tokenizer_special_ids": {
                "eos_token_id": eos_id,
                "eot_token_id": eot_id,
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "mask_token_id": getattr(tokenizer, "mask_token_id", None),
                "all_special_ids": all_special_ids,
            },
        },
    )

    per_condition_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"[run] condition={condition}")
        for prompt in tqdm(prompts, desc=f"condition[{condition}]", leave=False):
            row = _generate_single_condition(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                condition=condition,
                eos_id=eos_id,
                eot_id=eot_id,
                all_special_ids=all_special_ids,
            )
            per_condition_rows[condition].append(row)

        with (run_dir / f"{condition}_generations.jsonl").open("w") as f:
            for row in per_condition_rows[condition]:
                f.write(json.dumps(row) + "\n")

    aggregate = {
        c: _aggregate_condition(rows)
        for c, rows in per_condition_rows.items()
    }

    summary = {
        "model_name": args.model_name,
        "device": DEVICE,
        "out_dir": str(run_dir),
        "smoke_test": bool(args.smoke_test),
        "gen_params": GEN_PARAMS,
        "conditions": CONDITIONS,
        "n_prompts": len(prompts),
        "aggregate": aggregate,
        "diagnostic_question": "Are short visible outputs mainly caused by EOS/special token selections in generated positions, and does banning EOS improve visible continuation length?",
    }

    _write_json(run_dir / "summary.json", summary)

    print(f"[done] wrote outputs under {run_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
