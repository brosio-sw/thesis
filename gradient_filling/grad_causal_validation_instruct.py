from __future__ import annotations

"""
Online gradient-vs-causal validation for LLaDA-Instruct denoising.

This script mirrors the generation/eval setup of
speed_adapt/compare_mean_steering_adaptive_schedule_instruct.py, but uses a
single decoding policy:
- alpha = 1.0
- schedule = fixed_1 (commit exactly one token per denoising step)
- commit ranking remains baseline low-confidence only

During generation, for the current provisional full text at each denoising
step, it computes per-candidate (currently masked in active block):
1) gradient-based attribution scores
2) causal leave-one-span-out deltas

The online scores are logged for analysis, but they do NOT affect commitment.
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

ALPHA = 1.0
SCHEDULE_MODE = "fixed_1"
SCHEDULE_SCORE_SOURCE = "provisional_full"

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
    "data/gradient_filling/compare_mean_steering_online_grad_causal_validation_instruct"
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
    """Compute online token-level gradient and causal validation scores."""

    def __init__(self, model_name: str = SENTIMENT_MODEL_NAME, device: str = SENTIMENT_DEVICE):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

        label2id = {
            str(k).upper(): int(v)
            for k, v in getattr(self.model.config, "label2id", {}).items()
        }
        self.neg_label_id = int(label2id.get("NEGATIVE", 0))

        emb_layer = self.model.get_input_embeddings()
        mask_tok_id = self.tokenizer.mask_token_id
        if mask_tok_id is None:
            mask_tok_id = self.tokenizer.unk_token_id
        if mask_tok_id is None:
            mask_tok_id = self.tokenizer.pad_token_id
        if mask_tok_id is None:
            mask_tok_id = 0

        with torch.no_grad():
            mask_ids = torch.tensor([int(mask_tok_id)], device=self.device)
            self.mask_baseline_embedding = emb_layer(mask_ids)[0].detach()

    @staticmethod
    def _shared_chars(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def _sentiment_logits_batch(self, texts: list[str], batch_size: int = 64) -> tuple[list[float], list[float]]:
        neg_logits: list[float] = []
        neg_probs: list[float] = []

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
                probs = torch.softmax(logits, dim=-1)

            neg_logits.extend(logits[:, self.neg_label_id].detach().cpu().tolist())
            neg_probs.extend(probs[:, self.neg_label_id].detach().cpu().tolist())

        return neg_logits, neg_probs

    def full_text_grad_attribution(self, text: str) -> dict[str, Any]:
        safe_text = text if text.strip() else "."

        with torch.enable_grad():
            enc = self.tokenizer(
                safe_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            emb_layer = self.model.get_input_embeddings()
            embeds = emb_layer(input_ids).detach()
            embeds.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(inputs_embeds=embeds, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            neg_logit = logits[0, self.neg_label_id]
            neg_prob = probs[0, self.neg_label_id]
            neg_logit.backward()

            grads = embeds.grad[0].detach()
            vecs = embeds[0].detach()

            grad_norm = grads.norm(dim=-1)
            grad_input = (grads * vecs).sum(dim=-1)
            grad_x_input = (grads * (vecs - self.mask_baseline_embedding)).sum(dim=-1)

            token_ids = input_ids[0].detach().cpu().tolist()
            token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)

        return {
            "negative_logit": float(neg_logit.detach().cpu().item()),
            "negative_prob": float(neg_prob.detach().cpu().item()),
            "classifier_offsets": [(int(a), int(b)) for a, b in offsets],
            "classifier_tokens": token_texts,
            "grad_norm": grad_norm.detach().cpu().tolist(),
            "grad_input": grad_input.detach().cpu().tolist(),
            "grad_x_input": grad_x_input.detach().cpu().tolist(),
        }


class LLaDASpanMapper:
    """Map LLaDA token positions to character spans in the decoded full text."""

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


def _safe_pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return None
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if np.std(arr_a) < 1e-12 or np.std(arr_b) < 1e-12:
        return None
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def generate_with_online_validation(
    *,
    model,
    tokenizer,
    prompt: str,
    steerer,
    termination_token_ids: list[int],
    validator: OnlineSentimentValidator,
    span_mapper: LLaDASpanMapper,
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

                # Keep LLaDA decoding efficient, but do NOT wrap the whole loop in
                # inference_mode: classifier attribution must run in a real grad-enabled context.
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
                    M_t = int(block_mask.sum().item())
                    if M_t <= 0:
                        continue

                    candidate_pos = (block_mask.nonzero(as_tuple=True)[0] + block_start).tolist()
                    provisional_ids = provisional[b].detach().cpu().tolist()
                    provisional_full_text = span_mapper.decode_llada(provisional_ids)
                    llada_spans = span_mapper.position_spans(provisional_ids, candidate_pos)

                    grad_pack = validator.full_text_grad_attribution(provisional_full_text)
                    full_neg_logit = float(grad_pack["negative_logit"])
                    full_neg_prob = float(grad_pack["negative_prob"])

                    cls_offsets = [tuple(xy) for xy in grad_pack["classifier_offsets"]]
                    cls_tokens = list(grad_pack["classifier_tokens"])
                    cls_grad_norm = list(grad_pack["grad_norm"])
                    cls_grad_input = list(grad_pack["grad_input"])
                    cls_grad_x_input = list(grad_pack["grad_x_input"])

                    candidate_records: list[dict[str, Any]] = []
                    gnorm_vals: list[float] = []
                    gxi_vals: list[float] = []

                    for pos in candidate_pos:
                        span = llada_spans.get(pos, (0, 0))
                        span_text = provisional_full_text[span[0] : span[1]] if span[1] > span[0] else ""

                        overlap_idx: list[int] = []
                        overlap_weights: list[float] = []
                        overlap_tokens: list[str] = []
                        overlap_offsets: list[tuple[int, int]] = []

                        w_sum = 0.0
                        gxi_weighted_sum = 0.0
                        gnorm_weighted_num = 0.0
                        ginput_weighted_sum = 0.0

                        unweighted_gnorm_vals: list[float] = []
                        unweighted_gxi_vals: list[float] = []

                        for j, cls_span in enumerate(cls_offsets):
                            cls_len = int(cls_span[1] - cls_span[0])
                            if cls_len <= 0:
                                continue
                            shared = OnlineSentimentValidator._shared_chars(span, cls_span)
                            if shared <= 0:
                                continue
                            w = float(shared / cls_len)

                            overlap_idx.append(j)
                            overlap_weights.append(w)
                            overlap_tokens.append(cls_tokens[j])
                            overlap_offsets.append((int(cls_span[0]), int(cls_span[1])))

                            w_sum += w
                            gxi_weighted_sum += w * float(cls_grad_x_input[j])
                            gnorm_weighted_num += w * float(cls_grad_norm[j])
                            ginput_weighted_sum += w * float(cls_grad_input[j])
                            unweighted_gnorm_vals.append(float(cls_grad_norm[j]))
                            unweighted_gxi_vals.append(float(cls_grad_x_input[j]))

                        if w_sum > 0.0:
                            gnorm_weighted_mean = float(gnorm_weighted_num / w_sum)
                            gxi_weighted_sum_val = float(gxi_weighted_sum)
                            ginput_weighted_sum_val = float(ginput_weighted_sum)
                        else:
                            gnorm_weighted_mean = 0.0
                            gxi_weighted_sum_val = 0.0
                            ginput_weighted_sum_val = 0.0

                        # Optional ablations for debugging.
                        gnorm_unweighted_mean = (
                            float(sum(unweighted_gnorm_vals) / len(unweighted_gnorm_vals))
                            if unweighted_gnorm_vals
                            else 0.0
                        )
                        gxi_unweighted_mean = (
                            float(sum(unweighted_gxi_vals) / len(unweighted_gxi_vals))
                            if unweighted_gxi_vals
                            else 0.0
                        )

                        baseline_score = float(baseline_scores[b, pos].detach().cpu().item())

                        rec = {
                            "position": int(pos),
                            "position_in_generated_region": int(pos - prompt_len),
                            "candidate_token_id": int(provisional_ids[pos]),
                            "candidate_token_text": tokenizer.decode(
                                [int(provisional_ids[pos])],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                            "llada_char_span": [int(span[0]), int(span[1])],
                            "llada_span_text": span_text,
                            "baseline_low_conf_score": baseline_score,
                            "overlap_classifier_token_indices": overlap_idx,
                            "overlap_classifier_token_texts": overlap_tokens,
                            "overlap_classifier_offsets": [[a, b] for a, b in overlap_offsets],
                            "overlap_weights": overlap_weights,
                            "grad_norm_weighted_mean": gnorm_weighted_mean,
                            "grad_x_input_weighted_sum": gxi_weighted_sum_val,
                            "grad_input_weighted_sum": ginput_weighted_sum_val,
                            "grad_norm_unweighted_mean": gnorm_unweighted_mean,
                            "grad_x_input_unweighted_mean": gxi_unweighted_mean,
                        }
                        candidate_records.append(rec)
                        gnorm_vals.append(gnorm_weighted_mean)
                        gxi_vals.append(gxi_weighted_sum_val)

                    cf_texts = [
                        _remove_char_span(provisional_full_text, tuple(rec["llada_char_span"]))
                        for rec in candidate_records
                    ]
                    cf_logits, cf_probs = validator._sentiment_logits_batch(cf_texts)

                    causal_vals: list[float] = []
                    for rec, cf_logit, cf_prob, cf_text in zip(candidate_records, cf_logits, cf_probs, cf_texts):
                        delta = float(full_neg_logit - float(cf_logit))
                        rec["counterfactual_text_preview"] = cf_text[:280]
                        rec["counterfactual_negative_logit"] = float(cf_logit)
                        rec["counterfactual_negative_prob"] = float(cf_prob)
                        rec["causal_delta"] = delta
                        causal_vals.append(delta)

                    corr_gnorm = _safe_pearson(gnorm_vals, causal_vals)
                    corr_gxi = _safe_pearson(gxi_vals, causal_vals)

                    _, selected_pos = torch.topk(baseline_scores[b], k=1)
                    transfer_index[b, selected_pos] = True

                    if enable_debug:
                        step_rec = {
                            "global_step": int(total_denoising_steps),
                            "block_idx": int(block_idx),
                            "batch_idx": int(b),
                            "schedule_mode": SCHEDULE_MODE,
                            "commit_policy": "baseline_low_confidence_fixed_1",
                            "score_object": "full_provisional_text",
                            "sentiment_scalar_main": "negative_logit",
                            "full_provisional_negative_logit": full_neg_logit,
                            "full_provisional_negative_prob": full_neg_prob,
                            "full_provisional_text_preview": provisional_full_text[:500],
                            "num_candidates": int(len(candidate_records)),
                            "chosen_commit_position": int(selected_pos[0].detach().cpu().item()),
                            "correlations": {
                                "pearson_grad_norm_weighted_mean_vs_causal_delta": corr_gnorm,
                                "pearson_grad_x_input_weighted_sum_vs_causal_delta": corr_gxi,
                            },
                            "classifier_token_debug": {
                                "tokens": cls_tokens,
                                "offsets": [[a, b] for a, b in cls_offsets],
                            },
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


def run_single_condition(
    *,
    alpha: float,
    termination_token_ids: list[int],
    prompts: list[str],
    model,
    tokenizer,
    vectors: dict[int, torch.Tensor],
    out_dir: Path,
    enable_debug: bool,
    validator: OnlineSentimentValidator,
) -> dict[str, Any]:
    run_tag = f"a{alpha:.1f}__sched_fixed1__online_grad_causal_validation"
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    steerer = build_mask_only_steerer(model, tokenizer, alpha=alpha, vectors=vectors)
    span_mapper = LLaDASpanMapper(tokenizer)

    answers: list[str] = []
    denoising_steps: list[int] = []
    debug_all: list[list[dict[str, Any]]] = []

    for prompt in tqdm(prompts, desc=f"run[a={alpha:.1f}, sched=fixed_1]", leave=False):
        answer, n_steps, dbg = generate_with_online_validation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steerer=steerer,
            termination_token_ids=termination_token_ids,
            validator=validator,
            span_mapper=span_mapper,
            enable_debug=enable_debug,
        )
        answers.append(answer)
        denoising_steps.append(n_steps)
        debug_all.append(dbg)

    metrics = evaluate_texts_filtered(prompts, answers, denoising_steps)
    metrics["method"] = "mean_mask_only_online_grad_causal_validation"
    metrics["alpha"] = float(alpha)
    metrics["schedule_mode"] = SCHEDULE_MODE
    metrics["schedule_score_source"] = SCHEDULE_SCORE_SOURCE
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
            "schedule_mode": SCHEDULE_MODE,
            "schedule_score_source": SCHEDULE_SCORE_SOURCE,
            "q_conf_mode": None,
            "mix_lambda": None,
            "calibration_path": None,
            "output_subdir": str(run_dir.relative_to(out_dir)),
            "schedule": {
                "fixed_1": "always commit 1",
                "ranking": "standard low_confidence baseline",
                "note": "online gradient/causal scores are measured only; they do not affect commitment",
                "block_loop": "while masked tokens remain in block",
            },
            "online_validation": {
                "sentiment_model": SENTIMENT_MODEL_NAME,
                "sentiment_scalar_main": "negative_logit",
                "also_logged": ["negative_prob"],
                "candidate_set": "currently masked positions in active block",
                "grad_scores": [
                    "grad_norm_weighted_mean",
                    "grad_x_input_weighted_sum",
                    "grad_input_weighted_sum",
                ],
                "causal_score": "causal_delta = S(full_provisional) - S(counterfactual_without_span)",
                "counterfactual": "remove exact candidate char span, then whitespace cleanup",
                "mapping": {
                    "source": "LLaDA span vs classifier token offsets",
                    "weight": "shared_chars / classifier_token_char_len",
                    "aggregation": {
                        "grad_x_input_weighted_sum": "sum_j w_ij * gxi_j",
                        "grad_norm_weighted_mean": "sum_j w_ij * gnorm_j / sum_j w_ij",
                    },
                    "zero_overlap_policy": "score=0.0",
                },
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
        "method,alpha,schedule_mode,schedule_score_source,q_conf_mode,mix_lambda,calibration_path,n_total,n_valid,n_invalid,invalid_fraction,bad_pattern_fraction,repetition_fraction,empty_fraction,sent_answer_mean,sent_answer_fraction,sent_combined_mean,ppl_answer_mean,ppl_combined_mean,mean_answer_words_all,mean_answer_words_valid_only,total_denoising_steps,avg_denoising_steps_per_generation"
    ]

    metrics = run_single_condition(
        alpha=float(ALPHA),
        termination_token_ids=termination_token_ids,
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        out_dir=out_dir,
        enable_debug=bool(args.smoke_test),
        validator=validator,
    )

    run_key = f"a{ALPHA:.1f}__sched_fixed1__online_grad_causal_validation"
    summary["results"][run_key] = metrics

    aggregate_rows.append(
        ",".join(
            [
                str(metrics["method"]),
                str(metrics["alpha"]),
                str(metrics["schedule_mode"]),
                str(metrics["schedule_score_source"]),
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

    summary["overall_avg_denoising_steps_per_generation"] = metrics.get(
        "avg_denoising_steps_per_generation"
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
