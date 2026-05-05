from __future__ import annotations

"""
debug_schedule_sentiment_sanity.py

Sanity-check sentiment dynamics used by
compare_mean_steering_adaptive_schedule_instruct.py.

This script reproduces the same denoising loop style and logs multiple
sentiment scores per step to diagnose why s_t can be high:
- s_provisional_full: score on full provisional decode (current experiment behavior)
- s_provisional_generated_only: score on generated-region provisional decode only
- s_committed_full: score on currently committed full text
- s_prompt_only: score on prompt-only baseline
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.sentiment_metrics import compute_sentiment_metrics
from llada.model import load_model, load_tokenizer
from steering.sus_fix import MeanActivationSteeringMaskedOnly


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
SENTIMENT_DEVICE = os.getenv("SENTIMENT_DEVICE", "cpu")

PROMPT_WORDS = 25
MASK_ID = 126336
STEER_LAYERS = list(range(9, 25))

GEN_PARAMS = dict(
    temperature=0.0,
    gen_length=60,
    block_length=60,
    fill_strategy="low_confidence",
)

NEUTRAL_EVAL_SET_PATH = Path("data/speed_adapt/neutral_prefix_eval_set/full_run/neutral_prefix_eval_set.json")
ACTIVATIONS_ROOT = Path("data/speed_adapt/instruct_real_sentiment_activations/full_run/activations")

SMOKE_MAX_PROMPTS = 2
DEFAULT_SCHEDULE_MODE = "adaptive_123"
DEFAULT_ALPHA = 0.0


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


def format_prompt(raw_prompt: str) -> str:
    return (
        "Continue the following product review naturally.\n"
        "Keep the same topic as the original text.\n"
        "Write around 10 complete and long sentences.\n"
        "Do not add headings, labels, or commentary.\n\n"
        f"Review: {raw_prompt}"
    )


def _prob_negative_single(text: str, device: str) -> float:
    m = compute_sentiment_metrics([text], device=device)
    return float(m["mean_negative"])


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
    raise ValueError(f"Unknown fill_strategy {fill_strategy!r}")


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


def _apply_position_aware_token_ban(
    *,
    logits: torch.Tensor,
    ban_positions: torch.Tensor,
    banned_ids: list[int],
) -> torch.Tensor:
    if not banned_ids:
        return logits
    out = logits.clone()
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    for tok_id in banned_ids:
        out[..., tok_id] = torch.where(ban_positions, neg_inf, out[..., tok_id])
    return out


def _schedule_u_base(schedule_mode: str, s_t: float) -> int:
    if schedule_mode == "fixed_1":
        return 1
    if schedule_mode == "fixed_2":
        return 2
    if schedule_mode == "fixed_3":
        return 3
    if schedule_mode == "adaptive_123":
        if s_t < (1.0 / 3.0):
            return 1
        if s_t < (2.0 / 3.0):
            return 2
        return 3
    raise ValueError(f"Unknown schedule_mode: {schedule_mode}")


def ensure_encoder_layer_compat(model) -> None:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        class _Enc:
            pass
        e = _Enc()
        e.layer = model.model.layers
        model.encoder = e
        return
    if (
        hasattr(model, "model")
        and hasattr(model.model, "transformer")
        and hasattr(model.model.transformer, "__contains__")
        and "blocks" in model.model.transformer
    ):
        class _Enc:
            pass
        e = _Enc()
        e.layer = model.model.transformer["blocks"]
        model.encoder = e
        return
    raise RuntimeError("Could not find transformer layers")


def load_neutral_prompts(path: Path, max_prompts: int) -> list[str]:
    with open(path) as f:
        payload = json.load(f)
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError(f"No rows in neutral set: {path}")
    prompts = [str(r["prompt_text"]) for r in rows[:max_prompts]]
    return prompts


def load_vectors(activations_root: Path, layer_ids: list[int]) -> dict[int, torch.Tensor]:
    neg = torch.load(activations_root / "real_negative.pt", map_location="cpu", weights_only=False)
    pos = torch.load(activations_root / "real_positive.pt", map_location="cpu", weights_only=False)
    vecs: dict[int, torch.Tensor] = {}
    for li in layer_ids:
        vecs[li] = (pos[li].float().mean(dim=0) - neg[li].float().mean(dim=0)).cpu()
    return vecs


def build_steerer(model, tokenizer, alpha: float, vectors: dict[int, torch.Tensor]):
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = MASK_ID
    steerer = MeanActivationSteeringMaskedOnly(
        layer_ids=STEER_LAYERS,
        alpha=float(alpha),
        token_average=True,
        mask_id=int(mask_id),
    )
    model_dtype = getattr(getattr(model, "config", None), "torch_dtype", None) or torch.float32
    steerer.layer_ids = list(STEER_LAYERS)
    steerer.vectors = {li: v.to(dtype=model_dtype, device="cpu") for li, v in vectors.items() if li in STEER_LAYERS}
    return steerer


def run_one_prompt(
    *,
    model,
    tokenizer,
    steerer,
    prompt: str,
    schedule_mode: str,
    termination_token_ids: list[int],
    use_formatted_prompt: bool,
) -> dict[str, Any]:
    formatted_prompt = format_prompt(prompt)
    model_prompt = formatted_prompt if use_formatted_prompt else prompt
    enc = tokenizer([model_prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    gen_length = GEN_PARAMS["gen_length"]
    block_length = GEN_PARAMS["block_length"]
    temperature = GEN_PARAMS["temperature"]
    fill_strategy = GEN_PARAMS["fill_strategy"]

    batch_size, prompt_len = input_ids.shape
    raw_mask_id = getattr(tokenizer, "mask_token_id", None)
    mask_id = int(raw_mask_id if raw_mask_id is not None else MASK_ID)

    x = torch.full((batch_size, prompt_len + gen_length), mask_id, dtype=torch.long, device=input_ids.device)
    x[:, :prompt_len] = input_ids.clone()
    attention_mask_full = torch.cat(
        [
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=attention_mask.device),
        ],
        dim=-1,
    )

    debug_steps: list[dict[str, Any]] = []
    prompt_only_score = _prob_negative_single(model_prompt, device=SENTIMENT_DEVICE)

    with torch.inference_mode():
        steerer.register_hooks(model)
        try:
            block_start = prompt_len
            block_end = prompt_len + block_length
            early_end = block_start + max(0, gen_length - 10)

            global_step = 0
            while bool((x[:, block_start:block_end] == mask_id).any().item()):
                global_step += 1
                mask_index = x == mask_id

                logits = model(x, attention_mask=attention_mask_full).logits.to(x.device)
                ban_positions = torch.zeros_like(x, dtype=torch.bool)
                ban_positions[:, block_start:early_end] = True
                logits = _apply_position_aware_token_ban(
                    logits=logits,
                    ban_positions=ban_positions,
                    banned_ids=termination_token_ids,
                )

                logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
                x0 = logits_noisy.argmax(dim=-1)
                x0 = torch.where(mask_index, x0, x)

                scores = _build_default_fill_scores(
                    logits=logits,
                    x0=x0,
                    fill_strategy=fill_strategy,
                    device=x.device,
                )
                scores[:, block_end:] = float("-inf")
                scores = torch.where(mask_index, scores, torch.full_like(scores, float("-inf")))

                provisional = torch.where(mask_index, x0, x)

                provisional_full = tokenizer.batch_decode(provisional, skip_special_tokens=True)[0]
                provisional_gen_ids = provisional[:, prompt_len:]
                provisional_gen_only = tokenizer.batch_decode(provisional_gen_ids, skip_special_tokens=True)[0]
                committed_full = tokenizer.batch_decode(x, skip_special_tokens=True)[0]

                s_full = _prob_negative_single(provisional_full, device=SENTIMENT_DEVICE)
                s_gen = _prob_negative_single(provisional_gen_only, device=SENTIMENT_DEVICE)
                s_committed = _prob_negative_single(committed_full, device=SENTIMENT_DEVICE)

                u_base = _schedule_u_base(schedule_mode, s_full)
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                chosen_u_t = 0
                for b in range(batch_size):
                    M_t = int((x[b, block_start:block_end] == mask_id).sum().item())
                    if M_t <= 0:
                        continue
                    u_t = min(u_base, M_t)
                    _, selected_pos = torch.topk(scores[b], k=u_t)
                    transfer_index[b, selected_pos] = True
                    chosen_u_t = u_t

                x[transfer_index] = x0[transfer_index]
                remaining_after = int((x[:, block_start:block_end] == mask_id).sum().item())

                debug_steps.append(
                    {
                        "global_step": global_step,
                        "s_prompt_only": float(prompt_only_score),
                        "s_provisional_full": float(s_full),
                        "s_provisional_generated_only": float(s_gen),
                        "s_committed_full": float(s_committed),
                        "schedule_mode": schedule_mode,
                        "u_base": int(u_base),
                        "u_t": int(chosen_u_t),
                        "remaining_masks_after_step": remaining_after,
                        "provisional_full_preview": provisional_full[:260],
                        "provisional_generated_only_preview": provisional_gen_only[:260],
                    }
                )
        finally:
            steerer.remove_hooks()

    answer = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
    return {
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "model_prompt_used": model_prompt,
        "use_formatted_prompt": bool(use_formatted_prompt),
        "answer": answer,
        "n_steps": len(debug_steps),
        "steps": debug_steps,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument("--neutral-eval-set-path", type=Path, default=NEUTRAL_EVAL_SET_PATH)
    p.add_argument("--activations-root", type=Path, default=ACTIVATIONS_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("data/speed_adapt/debug_schedule_sentiment_sanity"))
    p.add_argument("--max-prompts", type=int, default=SMOKE_MAX_PROMPTS)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--schedule-mode", type=str, default=DEFAULT_SCHEDULE_MODE)
    p.add_argument(
        "--no-format-prompt",
        action="store_true",
        help="Use raw prompt directly (no instruction wrapper) for generation/scoring.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(SEED)

    out_dir = args.out_dir / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_neutral_prompts(args.neutral_eval_set_path, max_prompts=args.max_prompts)

    print(f"[model] loading {args.model_name} on {DEVICE}")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)
    ensure_encoder_layer_compat(model)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = _infer_eot_token_id(tokenizer)
    termination_token_ids = sorted({int(t) for t in [eos_id, eot_id] if t is not None})

    vectors = load_vectors(args.activations_root, STEER_LAYERS)
    steerer = build_steerer(model, tokenizer, alpha=float(args.alpha), vectors=vectors)

    all_debug: list[dict[str, Any]] = []
    for i, ptxt in enumerate(prompts):
        print(f"[run] prompt {i+1}/{len(prompts)}")
        obj = run_one_prompt(
            model=model,
            tokenizer=tokenizer,
            steerer=steerer,
            prompt=ptxt,
            schedule_mode=args.schedule_mode,
            termination_token_ids=termination_token_ids,
            use_formatted_prompt=not bool(args.no_format_prompt),
        )
        all_debug.append(obj)
        _write_json(out_dir / f"debug_prompt_{i:03d}.json", obj)

    # Aggregate sanity summary.
    full_scores = []
    gen_scores = []
    committed_scores = []
    prompt_scores = []
    for obj in all_debug:
        for s in obj["steps"]:
            full_scores.append(float(s["s_provisional_full"]))
            gen_scores.append(float(s["s_provisional_generated_only"]))
            committed_scores.append(float(s["s_committed_full"]))
            prompt_scores.append(float(s["s_prompt_only"]))

    summary = {
        "model_name": args.model_name,
        "device": DEVICE,
        "sentiment_device": SENTIMENT_DEVICE,
        "n_prompts": len(prompts),
        "schedule_mode": args.schedule_mode,
        "alpha": float(args.alpha),
        "use_formatted_prompt": not bool(args.no_format_prompt),
        "termination_token_ids": termination_token_ids,
        "termination_policy": "EOS/EOT banned in generated positions 0..49, allowed in 50..59",
        "means": {
            "mean_s_prompt_only": float(np.mean(prompt_scores)) if prompt_scores else None,
            "mean_s_provisional_full": float(np.mean(full_scores)) if full_scores else None,
            "mean_s_provisional_generated_only": float(np.mean(gen_scores)) if gen_scores else None,
            "mean_s_committed_full": float(np.mean(committed_scores)) if committed_scores else None,
        },
        "mins": {
            "min_s_provisional_full": float(np.min(full_scores)) if full_scores else None,
            "min_s_provisional_generated_only": float(np.min(gen_scores)) if gen_scores else None,
        },
        "maxs": {
            "max_s_provisional_full": float(np.max(full_scores)) if full_scores else None,
            "max_s_provisional_generated_only": float(np.max(gen_scores)) if gen_scores else None,
        },
    }

    _write_json(out_dir / "summary.json", summary)
    print(f"[done] wrote outputs under {out_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
