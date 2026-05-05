from __future__ import annotations

"""
Closed-loop causal mixed-commit experiment for LLaDA-Instruct.

This script keeps the same generation/eval setup as
gradient_filling/diffmean_h3_sentiment_grad_test.py, but changes token
commitment in fixed_1 schedule:
- commit exactly 1 token per denoising step
- select by mixed score using baseline low-confidence + normalized online causal score

mixed_score_i = lambda_mix * baseline_low_conf_score_i
                + (1 - lambda_mix) * causal_norm_i

Where causal_norm_i is per-step min-max normalization of
causal_delta_i = S(full_provisional) - S(counterfactual_without_i)
across eligible candidate positions.
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
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
SENTIMENT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_DEVICE = os.getenv("SENTIMENT_DEVICE", "cpu")

PROMPT_WORDS = 25
STEER_LAYERS = list(range(9, 25))
MASK_ID = 126336

ALPHA = 0.50  # Steering alpha only.
SCHEDULE_MODE = "fixed_1"
SCHEDULE_SCORE_SOURCE = "provisional_full"
LAMBDA_MIX_VALUES = [0.5, 0.75, 0.9, 1.0]

SMOKE_TEST = False
SMOKE_MAX_PROMPTS = 2
MAX_PROMPTS = 150

NEUTRAL_EVAL_SET_PATH = Path(
    "data/speed_adapt/neutral_prefix_eval_set/full_run/neutral_prefix_eval_set.json"
)

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=60,
    block_length=60,
    fill_strategy="low_confidence",
)

DEFAULT_OUT_DIR = Path(
    "data/gradient_filling/compare_mean_steering_online_causal_mixed_commit_instruct"
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


def _normalize_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [float((v - vmin) / (vmax - vmin)) for v in values]


def load_neutral_eval_prompts(
    eval_set_path: Path,
) -> tuple[list[str], list[int], list[int], dict[str, Any]]:
    if not eval_set_path.exists():
        raise FileNotFoundError(f"Neutral eval set not found: {eval_set_path}")

    with open(eval_set_path) as f:
        payload = json.load(f)

    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError(f"No rows found in neutral eval set: {eval_set_path}")

    prompts = [str(r["prompt_text"]) for r in rows]
    labels = [int(r.get("label", -1)) for r in rows]
    dataset_indices = [int(r.get("dataset_index", -1)) for r in rows]

    label_counts = {
        "negative_0": int(sum(1 for x in labels if x == 0)),
        "positive_1": int(sum(1 for x in labels if x == 1)),
        "unknown": int(sum(1 for x in labels if x not in [0, 1])),
    }

    meta = {
        "source": "neutral_prefix_eval_set",
        "eval_set_path": str(eval_set_path),
        "builder_meta": payload.get("meta", {}),
        "n_prompts": len(prompts),
        "label_counts": label_counts,
    }
    return prompts, labels, dataset_indices, meta


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

    logits_mod = logits.clone()
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    for tok_id in banned_ids:
        logits_mod[..., tok_id] = torch.where(
            ban_positions, neg_inf, logits_mod[..., tok_id]
        )
    return logits_mod


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


def load_or_build_mean_diff_vectors(
    *,
    vector_path: Path | None,
    activations_root: Path | None,
    layer_ids: list[int],
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if vector_path is not None:
        obj = torch.load(vector_path, map_location="cpu", weights_only=False)
        vectors = {int(k): v.float().cpu() for k, v in obj.items() if int(k) in layer_ids}
        missing = [li for li in layer_ids if li not in vectors]
        if missing:
            raise RuntimeError(f"Missing layers {missing} in vector file {vector_path}")
        return vectors, {"source": "vector_file", "vector_path": str(vector_path)}

    if activations_root is not None:
        neg_path = activations_root / "real_negative.pt"
        pos_path = activations_root / "real_positive.pt"
        if not neg_path.exists() or not pos_path.exists():
            raise FileNotFoundError(
                "Expected activation files not found. Need real_negative.pt and real_positive.pt under "
                f"{activations_root}"
            )

        neg = torch.load(neg_path, map_location="cpu", weights_only=False)
        pos = torch.load(pos_path, map_location="cpu", weights_only=False)

        vectors: dict[int, torch.Tensor] = {}
        for li in layer_ids:
            if li not in neg or li not in pos:
                raise RuntimeError(f"Layer {li} missing from activation files")
            vectors[li] = (pos[li].float().mean(dim=0) - neg[li].float().mean(dim=0)).cpu()

        return vectors, {
            "source": "activation_files",
            "real_negative_path": str(neg_path),
            "real_positive_path": str(pos_path),
        }

    raise ValueError("Provide either --vector-path or --activations-root")


def build_mask_only_steerer(
    model,
    tokenizer,
    alpha: float,
    vectors: dict[int, torch.Tensor],
):
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

    steerer.layer_ids = list(STEER_LAYERS)
    steerer.vectors = {
        li: vec.to(dtype=model_dtype, device="cpu")
        for li, vec in vectors.items()
        if li in STEER_LAYERS
    }
    return steerer


class OnlineSentimentValidator:
    """Compute online full-text negative logits and batched rescoring."""

    def __init__(self, model_name: str = SENTIMENT_MODEL_NAME, device: str = SENTIMENT_DEVICE):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

        label2id = {
            str(k).upper(): int(v)
            for k, v in getattr(self.model.config, "label2id", {}).items()
        }
        self.neg_label_id = int(label2id.get("NEGATIVE", 0))

    def negative_logits_batch(self, texts: list[str], batch_size: int = 64) -> list[float]:
        neg_logits: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = [t if t.strip() else "." for t in texts[start : start + batch_size]]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            neg_logits.extend(logits[:, self.neg_label_id].detach().cpu().tolist())
        return neg_logits

    def negative_logit(self, text: str) -> float:
        safe_text = text if text.strip() else "."
        return float(self.negative_logits_batch([safe_text], batch_size=1)[0])


class LLaDASpanMapper:
    """Map LLaDA token positions to character spans in decoded full text."""

    def __init__(self, llada_tokenizer):
        self.llada_tokenizer = llada_tokenizer

    def decode_llada(self, token_ids: list[int]) -> str:
        return self.llada_tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def position_spans(self, token_ids: list[int], positions: list[int]) -> dict[int, tuple[int, int]]:
        spans: dict[int, tuple[int, int]] = {}
        for p in positions:
            if p < 0 or p >= len(token_ids):
                spans[p] = (0, 0)
                continue
            left = self.decode_llada(token_ids[:p])
            right = self.decode_llada(token_ids[: p + 1])
            spans[p] = (len(left), len(right))
        return spans


def _cleanup_removed_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "."


def _remove_char_span(text: str, span: tuple[int, int]) -> str:
    s, e = span
    s = max(0, min(len(text), s))
    e = max(0, min(len(text), e))
    if e <= s:
        return _cleanup_removed_text(text)
    return _cleanup_removed_text(text[:s] + text[e:])


def generate_with_mixed_causal_commit(
    *,
    model,
    tokenizer,
    prompt: str,
    steerer,
    termination_token_ids: list[int],
    validator: OnlineSentimentValidator,
    span_mapper: LLaDASpanMapper,
    lambda_mix: float,
    enable_debug: bool,
) -> tuple[str, int, list[dict[str, Any]]]:
    model_prompt = prompt
    enc = tokenizer([model_prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)
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

    steerer.register_hooks(model)
    try:
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            early_end = block_start + max(0, gen_length - 10)

            while bool((x[:, block_start:block_end] == mask_id).any().item()):
                total_denoising_steps += 1

                with torch.no_grad():
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

                    provisional = torch.where(mask_index, x0, x)

                transfer_index = torch.zeros_like(x, dtype=torch.bool)

                for b in range(batch_size):
                    block_mask = x[b, block_start:block_end] == mask_id
                    n_masked = int(block_mask.sum().item())
                    if n_masked <= 0:
                        continue

                    candidate_pos = (block_mask.nonzero(as_tuple=True)[0] + block_start).tolist()
                    provisional_ids = provisional[b].detach().cpu().tolist()
                    provisional_full_text = span_mapper.decode_llada(provisional_ids)
                    llada_spans = span_mapper.position_spans(provisional_ids, candidate_pos)

                    full_neg_logit = validator.negative_logit(provisional_full_text)

                    candidate_records: list[dict[str, Any]] = []
                    baseline_vals: list[float] = []

                    for pos in candidate_pos:
                        span = llada_spans.get(pos, (0, 0))
                        baseline_score = float(baseline_scores[b, pos].detach().cpu().item())
                        token_id = int(provisional_ids[pos])
                        token_text = tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )

                        candidate_records.append(
                            {
                                "position": int(pos),
                                "position_in_generated_region": int(pos - prompt_len),
                                "candidate_token_id": token_id,
                                "candidate_token_text": token_text,
                                "llada_char_span": [int(span[0]), int(span[1])],
                                "baseline_low_conf_score": baseline_score,
                            }
                        )
                        baseline_vals.append(baseline_score)

                    cf_texts = [
                        _remove_char_span(provisional_full_text, tuple(rec["llada_char_span"]))
                        for rec in candidate_records
                    ]
                    cf_logits = validator.negative_logits_batch(cf_texts)

                    causal_deltas: list[float] = []
                    for rec, cf_logit, cf_text in zip(candidate_records, cf_logits, cf_texts):
                        delta = float(full_neg_logit - float(cf_logit))
                        rec["counterfactual_negative_logit"] = float(cf_logit)
                        rec["counterfactual_text_preview"] = cf_text[:280]
                        rec["causal_delta"] = delta
                        causal_deltas.append(delta)

                    causal_norm_vals = _normalize_minmax(causal_deltas)
                    mixed_vals = [
                        (float(lambda_mix) * bscore) + ((1.0 - float(lambda_mix)) * causal_norm)
                        for bscore, causal_norm in zip(baseline_vals, causal_norm_vals)
                    ]

                    best_local_idx = int(np.argmax(np.asarray(mixed_vals, dtype=np.float64)))
                    chosen_pos = int(candidate_pos[best_local_idx])
                    transfer_index[b, chosen_pos] = True

                    if enable_debug:
                        for rec, causal_norm, mixed in zip(candidate_records, causal_norm_vals, mixed_vals):
                            rec["causal_norm"] = float(causal_norm)
                            rec["lambda_mix"] = float(lambda_mix)
                            rec["mixed_score"] = float(mixed)

                        step_rec = {
                            "global_step": int(total_denoising_steps),
                            "block_idx": int(block_idx),
                            "batch_idx": int(b),
                            "schedule_mode": SCHEDULE_MODE,
                            "commit_policy": "fixed_1_mixed_baseline_causal",
                            "lambda_mix": float(lambda_mix),
                            "full_provisional_negative_logit": float(full_neg_logit),
                            "full_provisional_text_preview": provisional_full_text[:500],
                            "causal_normalization": {
                                "type": "per_step_minmax_over_eligible_candidates",
                                "constant_case": "if max==min then causal_norm=0 for all candidates",
                            },
                            "chosen_commit_position": chosen_pos,
                            "num_candidates": int(len(candidate_records)),
                            "candidates": candidate_records,
                        }
                        debug_steps.append(step_rec)

                x[transfer_index] = x0[transfer_index]
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
        mean_answer_words_valid_only = float(
            sum(len(a.split()) for a in valid_answers) / max(1, len(valid_answers))
        )
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
        "avg_denoising_steps_per_generation": (
            float(np.mean(denoising_steps_per_generation))
            if denoising_steps_per_generation
            else None
        ),
    }


def _mix_lambda_tag(value: float) -> str:
    return f"{float(value):.2f}"


def run_single_condition(
    *,
    alpha: float,
    lambda_mix: float,
    termination_token_ids: list[int],
    prompts: list[str],
    model,
    tokenizer,
    vectors: dict[int, torch.Tensor],
    out_dir: Path,
    enable_debug: bool,
    validator: OnlineSentimentValidator,
) -> dict[str, Any]:
    run_tag = f"a{alpha:.1f}__causalmix_{_mix_lambda_tag(lambda_mix)}"
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    steerer = build_mask_only_steerer(model, tokenizer, alpha=alpha, vectors=vectors)
    span_mapper = LLaDASpanMapper(tokenizer)

    answers: list[str] = []
    denoising_steps: list[int] = []
    debug_all: list[list[dict[str, Any]]] = []

    for prompt in tqdm(prompts, desc=f"run[a={alpha:.1f}, causalmix={lambda_mix:.2f}]", leave=False):
        answer, n_steps, dbg = generate_with_mixed_causal_commit(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steerer=steerer,
            termination_token_ids=termination_token_ids,
            validator=validator,
            span_mapper=span_mapper,
            lambda_mix=float(lambda_mix),
            enable_debug=enable_debug,
        )
        answers.append(answer)
        denoising_steps.append(n_steps)
        debug_all.append(dbg)

    metrics = evaluate_texts_filtered(prompts, answers, denoising_steps)
    metrics["method"] = "mean_mask_only_causal_mixed_commit"
    metrics["alpha"] = float(alpha)
    metrics["schedule_mode"] = SCHEDULE_MODE
    metrics["schedule_score_source"] = SCHEDULE_SCORE_SOURCE
    metrics["lambda_mix"] = float(lambda_mix)
    metrics["q_conf_mode"] = None
    metrics["mix_lambda"] = None
    metrics["calibration_path"] = None
    metrics["output_subdir"] = str(run_dir.relative_to(out_dir))

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
            "lambda_mix": float(lambda_mix),
            "schedule_mode": SCHEDULE_MODE,
            "schedule_score_source": SCHEDULE_SCORE_SOURCE,
            "q_conf_mode": None,
            "mix_lambda": None,
            "calibration_path": None,
            "output_subdir": str(run_dir.relative_to(out_dir)),
            "commit_policy": "fixed_1_mixed_baseline_causal",
            "schedule": {
                "fixed_1": "always commit 1",
                "ranking": "select by mixed_score = lambda_mix*baseline + (1-lambda_mix)*causal_norm",
                "note": "lambda_mix=1.0 is pure baseline ablation",
                "block_loop": "while masked tokens remain in block",
            },
            "causal_commit_scoring": {
                "baseline_component": "baseline_low_conf_score",
                "causal_component": "causal_delta = S(full_provisional) - S(counterfactual_without_span)",
                "sentiment_scalar": "negative_logit",
                "counterfactual_construction": "remove exact candidate LLaDA char span, then whitespace cleanup",
                "online_scope": "computed on each denoising step using the same provisional text seen by sampler",
                "normalization": {
                    "type": "per_step_minmax_over_eligible_candidates",
                    "constant_case": "if max==min then causal_norm=0 for all candidates",
                },
                "mixed_score": "lambda_mix * baseline_low_conf_score + (1-lambda_mix) * causal_norm",
                "lambda_mix_values": LAMBDA_MIX_VALUES,
            },
            "termination_policy": {
                "termination_token_ids": termination_token_ids,
                "policy": "EOS/EOT banned in generated positions 0..49, allowed in 50..59",
                "gen_length": GEN_PARAMS["gen_length"],
                "block_length": GEN_PARAMS["block_length"],
            },
            "gen_params": GEN_PARAMS,
            "steer_layers": STEER_LAYERS,
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
        "--vector-path",
        type=Path,
        default=None,
        help="Path to mean_diff_vectors.pt. If not provided, --activations-root is used.",
    )
    parser.add_argument(
        "--activations-root",
        type=Path,
        default=Path("data/speed_adapt/instruct_real_sentiment_activations/full_run/activations"),
        help="Directory containing real_negative.pt and real_positive.pt.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--neutral-eval-set-path",
        type=Path,
        default=NEUTRAL_EVAL_SET_PATH,
    )
    parser.add_argument("--max-prompts", type=int, default=MAX_PROMPTS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-prompts", type=int, default=SMOKE_MAX_PROMPTS)
    parser.set_defaults(smoke_test=SMOKE_TEST)
    return parser.parse_args()


def main() -> None:
    print("starting")
    args = parse_args()
    _seed_everything(SEED)

    out_dir = args.out_dir / ("smoke_test" if args.smoke_test else "full_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, labels, dataset_indices, eval_meta = load_neutral_eval_prompts(
        args.neutral_eval_set_path
    )

    if args.smoke_test:
        n_keep = min(args.smoke_max_prompts, len(prompts))
    else:
        n_keep = min(args.max_prompts, len(prompts))

    prompts = prompts[:n_keep]
    labels = labels[:n_keep]
    dataset_indices = dataset_indices[:n_keep]

    _write_json(
        out_dir / "eval_set.json",
        {
            "meta": eval_meta,
            "dataset_indices": dataset_indices,
            "labels": labels,
            "prompts": prompts,
            "formatted_prompts": [format_prompt(p) for p in prompts],
            "smoke_test": bool(args.smoke_test),
            "max_prompts": int(n_keep),
        },
    )

    vectors, vector_meta = load_or_build_mean_diff_vectors(
        vector_path=args.vector_path,
        activations_root=args.activations_root,
        layer_ids=STEER_LAYERS,
    )

    print(f"[model] loading {args.model_name} on {DEVICE}")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)
    ensure_encoder_layer_compat(model)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = _infer_eot_token_id(tokenizer)
    termination_token_ids = sorted({int(t) for t in [eos_id, eot_id] if t is not None})

    print(f"[sentiment] loading {SENTIMENT_MODEL_NAME} on {SENTIMENT_DEVICE}")
    validator = OnlineSentimentValidator(model_name=SENTIMENT_MODEL_NAME, device=SENTIMENT_DEVICE)

    summary: dict[str, Any] = {
        "model_name": args.model_name,
        "device": DEVICE,
        "sentiment_device": SENTIMENT_DEVICE,
        "out_dir": str(out_dir),
        "vector_meta": vector_meta,
        "steer_layers": STEER_LAYERS,
        "prompt_words": PROMPT_WORDS,
        "gen_params": GEN_PARAMS,
        "alpha": ALPHA,
        "schedule_mode": SCHEDULE_MODE,
        "schedule_score_source": SCHEDULE_SCORE_SOURCE,
        "lambda_mix_values": LAMBDA_MIX_VALUES,
        "termination_policy": {
            "termination_token_ids": termination_token_ids,
            "policy": "EOS/EOT banned in generated positions 0..49, allowed in 50..59",
            "gen_length": GEN_PARAMS["gen_length"],
            "block_length": GEN_PARAMS["block_length"],
        },
        "prompting": {
            "use_system_prompt_wrapper": False,
            "generation_prompt_source": "raw_prompt",
        },
        "evaluation_subset": "valid_generations_only",
        "bad_patterns": BAD_PATTERNS,
        "eval_meta": eval_meta,
        "smoke_test": bool(args.smoke_test),
        "results": {},
    }

    aggregate_rows = [
        "method,alpha,schedule_mode,schedule_score_source,lambda_mix,q_conf_mode,mix_lambda,calibration_path,n_total,n_valid,n_invalid,invalid_fraction,bad_pattern_fraction,repetition_fraction,empty_fraction,sent_answer_mean,sent_answer_fraction,sent_combined_mean,ppl_answer_mean,ppl_combined_mean,mean_answer_words_all,mean_answer_words_valid_only,total_denoising_steps,avg_denoising_steps_per_generation"
    ]

    for lambda_mix in LAMBDA_MIX_VALUES:
        print(f"[run] alpha={ALPHA:.1f} lambda_mix={lambda_mix:.2f} schedule={SCHEDULE_MODE}")
        metrics = run_single_condition(
            alpha=float(ALPHA),
            lambda_mix=float(lambda_mix),
            termination_token_ids=termination_token_ids,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            vectors=vectors,
            out_dir=out_dir,
            enable_debug=bool(args.smoke_test),
            validator=validator,
        )

        run_key = f"a{ALPHA:.1f}__causalmix_{_mix_lambda_tag(lambda_mix)}"
        summary["results"][run_key] = metrics

        aggregate_rows.append(
            ",".join(
                [
                    str(metrics["method"]),
                    str(metrics["alpha"]),
                    str(metrics["schedule_mode"]),
                    str(metrics["schedule_score_source"]),
                    str(metrics["lambda_mix"]),
                    str(metrics["q_conf_mode"]),
                    str(metrics["mix_lambda"]),
                    str(metrics["calibration_path"]),
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

    vals = [
        float(v["avg_denoising_steps_per_generation"])
        for v in summary["results"].values()
        if v.get("avg_denoising_steps_per_generation") is not None
    ]
    summary["overall_avg_denoising_steps_per_generation"] = (
        float(np.mean(vals)) if vals else None
    )

    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.csv").write_text("\n".join(aggregate_rows) + "\n")
    print(f"[done] wrote outputs under {out_dir}")

    del model
    del validator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
