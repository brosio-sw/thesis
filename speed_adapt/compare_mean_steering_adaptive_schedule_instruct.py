from __future__ import annotations

"""
compare_mean_steering_adaptive_schedule_instruct.py

LLaDA-Instruct masked-only DiffMean schedule comparison.

Matrix:
- alpha in {0, 4, 10}
- schedule_mode in {fixed_1, fixed_2, fixed_3, adaptive_123, mix_adaptive}

Commit selection policy:
- Keep standard low-confidence ranking for *which* positions to commit.
- Only schedule changes *how many* positions are committed per denoising step.

EOS/EOT policy:
- generated region length = 60
- EOS/EOT-like tokens banned for generated positions 0..49
- EOS/EOT-like tokens allowed again for generated positions 50..59

Prompting:
- generation uses raw neutral-prefix prompt directly (no instruction wrapper)

Score-source handling:
- adaptive_123 runs once per requested score source
- fixed schedules run only once, using the canonical score tag "provisional_full"
- mix_adaptive always uses provisional_generated_only for s_t
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
STEER_LAYERS = list(range(9, 25))
MASK_ID = 126336

ALPHAS = [0]
ALL_SCHEDULE_MODES = ["fixed_1", "fixed_2", "fixed_3", "adaptive_123", "mix_adaptive", "calibrated_mix"]
# SCHEDULE_MODES = ["fixed_1", "fixed_2", "fixed_3", "adaptive_123", "mix_adaptive", "calibrated_mix"]
SCHEDULE_MODES = ["fixed_1"]
SCHEDULE_SCORE_SOURCES = ["provisional_full", "provisional_generated_only"]
ALL_Q_CONF_MODES = ["mean_top3", "min_top3", "mean_masked_block", "mean_all_block"]
CALIBRATED_MIX_Q_CONF_MODES = ["mean_top3", "min_top3"]
Q_CONF_MODES = CALIBRATED_MIX_Q_CONF_MODES
LAMBDA_VALUES = [0.0, 0.5, 1.0]
Q_THRESHOLDS = {
    "mean_top3": {"q_mid": 0.11, "q_high": 0.25},
    "min_top3": {"q_mid": 0.07, "q_high": 0.16},
    "mean_masked_block": {"q_mid": 0.05, "q_high": 0.11},
    "mean_all_block": {"q_mid": 0.22, "q_high": 0.41},
}
Q_MID = 0.4
Q_HIGH = 0.65

SMOKE_TEST = False
SMOKE_MAX_PROMPTS = 4
SMOKE_ALPHAS = [1.0]
SMOKE_SCHEDULE_MODES = ["calibrated_mix"]
SMOKE_SCHEDULE_SCORE_SOURCES = ["provisional_full", "provisional_generated_only"]
SMOKE_Q_CONF_MODES = CALIBRATED_MIX_Q_CONF_MODES
SMOKE_LAMBDAS = LAMBDA_VALUES

CALIBRATION_BASE_DIR = Path("data/speed_adapt/debug_calibrate_speed_signals/full_run")

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


def _prob_negative_single(text: str, device: str) -> float:
    m = compute_sentiment_metrics([text], device=device)
    return float(m["mean_negative"])


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
    if schedule_mode == "mix_adaptive":
        raise ValueError("mix_adaptive uses q_t-dependent rule; handle in generation loop")
    if schedule_mode == "calibrated_mix":
        raise ValueError("calibrated_mix uses calibrated mix score; handle in generation loop")
    raise ValueError(f"Unknown schedule_mode: {schedule_mode}")


def _schedule_tag(schedule_mode: str) -> str:
    return {
        "fixed_1": "fixed1",
        "fixed_2": "fixed2",
        "fixed_3": "fixed3",
        "adaptive_123": "adapt123",
        "mix_adaptive": "mixadapt",
        "calibrated_mix": "calmix",
    }[schedule_mode]


def _q_conf_tag(q_conf_mode: str | None) -> str:
    if q_conf_mode is None:
        return "na"
    return {
        "mean_top3": "mtop3",
        "min_top3": "mintop3",
        "mean_masked_block": "mmasked",
        "mean_all_block": "mall",
    }[q_conf_mode]


def _score_source_tag(schedule_score_source: str) -> str:
    return {
        "provisional_full": "full",
        "provisional_generated_only": "genonly",
    }[schedule_score_source]


def _score_sources_for_mode(
    schedule_mode: str,
    requested_sources: list[str],
) -> list[str]:
    """
    Fixed schedules do not depend on score source, so run them only once.
    Use 'provisional_full' as the canonical tag when available.
    """
    if schedule_mode == "adaptive_123":
        return list(requested_sources)

    if schedule_mode == "mix_adaptive":
        return ["provisional_generated_only"]

    if schedule_mode == "calibrated_mix":
        return ["provisional_generated_only"]

    if "provisional_full" in requested_sources:
        return ["provisional_full"]

    return [requested_sources[0]]


def _q_conf_modes_for_mode(schedule_mode: str, requested_q_modes: list[str]) -> list[str | None]:
    if schedule_mode in {"mix_adaptive", "calibrated_mix"}:
        return list(requested_q_modes)
    return [None]


def _lambdas_for_mode(schedule_mode: str, requested_lambdas: list[float]) -> list[float | None]:
    if schedule_mode == "calibrated_mix":
        return list(requested_lambdas)
    return [None]


def _compute_q_t(
    *,
    q_conf_mode: str,
    baseline_scores_raw_row: torch.Tensor,
    block_start: int,
    block_end: int,
    block_mask_row: torch.Tensor,
) -> float:
    block_scores = baseline_scores_raw_row[block_start:block_end]

    if q_conf_mode == "mean_all_block":
        return float(block_scores.float().mean().item())

    masked_vals = block_scores[block_mask_row]
    if masked_vals.numel() <= 0:
        return 0.0

    if q_conf_mode == "mean_masked_block":
        return float(masked_vals.float().mean().item())

    k = min(3, int(masked_vals.numel()))
    top_vals, _ = torch.topk(masked_vals.float(), k=k)
    if q_conf_mode == "mean_top3":
        return float(top_vals.mean().item())
    if q_conf_mode == "min_top3":
        return float(top_vals.min().item())

    raise ValueError(f"Unknown q_conf_mode: {q_conf_mode}")


def _resolve_q_thresholds(
    *,
    schedule_mode: str,
    q_conf_mode: str | None,
    q_mid_default: float,
    q_high_default: float,
) -> tuple[float, float]:
    if schedule_mode != "mix_adaptive":
        return float(q_mid_default), float(q_high_default)

    if q_conf_mode is None:
        raise ValueError("q_conf_mode is required for mix_adaptive")

    if q_conf_mode not in Q_THRESHOLDS:
        raise ValueError(
            f"No thresholds configured for q_conf_mode={q_conf_mode!r}. "
            f"Expected one of: {sorted(Q_THRESHOLDS.keys())}"
        )

    cfg = Q_THRESHOLDS[q_conf_mode]
    return float(cfg["q_mid"]), float(cfg["q_high"])


def _float_tag(value: float) -> str:
    txt = f"{value:.3f}".rstrip("0").rstrip(".")
    return txt.replace(".", "p")


def _lambda_tag(value: float | None) -> str:
    if value is None:
        return "na"
    return _float_tag(float(value))


def _threshold_subdir_name(*, q_mid: float, q_high: float) -> str:
    return f"qthr_mid{_float_tag(q_mid)}__high{_float_tag(q_high)}"


def _phase_from_masks(remaining_before: int, total_masks: int) -> str:
    if total_masks <= 0:
        return "unknown"
    progress = 1.0 - (float(remaining_before) / float(total_masks))
    if progress < (1.0 / 3.0):
        return "early"
    if progress < (2.0 / 3.0):
        return "middle"
    return "late"


def _alpha_tag(alpha: float) -> str:
    return f"a{float(alpha):.1f}"


def _empirical_quantile_from_sorted(raw_value: float, sorted_values: list[float]) -> float:
    if not sorted_values:
        return float("nan")
    arr = np.asarray(sorted_values, dtype=np.float64)
    idx = int(np.searchsorted(arr, raw_value, side="right"))
    return float(idx / arr.size)


def _calibrated_quantile(
    *,
    calibration_payload: dict[str, Any],
    signal_key: str,
    raw_value: float,
    phase: str,
) -> float:
    by_phase = calibration_payload.get("by_phase", {})
    phase_sorted = (
        by_phase.get(phase, {})
        .get(signal_key, {})
        .get("empirical_cdf", {})
        .get("sorted_values", [])
    )
    if phase_sorted:
        q = _empirical_quantile_from_sorted(raw_value, [float(v) for v in phase_sorted])
        if not np.isnan(q):
            return float(np.clip(q, 0.0, 1.0))

    overall_sorted = (
        calibration_payload.get("overall", {})
        .get(signal_key, {})
        .get("empirical_cdf", {})
        .get("sorted_values", [])
    )
    q = _empirical_quantile_from_sorted(raw_value, [float(v) for v in overall_sorted])
    if np.isnan(q):
        raise RuntimeError(
            f"Calibration sorted_values missing for signal={signal_key!r}, phase={phase!r}"
        )
    return float(np.clip(q, 0.0, 1.0))


def _load_calibration_for_alpha(calibration_base_dir: Path, alpha: float) -> tuple[dict[str, Any], Path]:
    cal_path = calibration_base_dir / _alpha_tag(alpha) / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found for alpha={alpha}: {cal_path}")
    with open(cal_path) as f:
        payload = json.load(f)

    for key in ["s_t", "mean_top3", "min_top3"]:
        vals = payload.get("overall", {}).get(key, {}).get("empirical_cdf", {}).get("sorted_values", [])
        if not vals:
            raise RuntimeError(f"Calibration file missing sorted_values for {key}: {cal_path}")
    return payload, cal_path


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


def generate_with_schedule_mode(
    *,
    model,
    tokenizer,
    prompt: str,
    steerer,
    schedule_mode: str,
    schedule_score_source: str,
    q_conf_mode: str | None,
    mix_lambda: float | None,
    calibration_payload: dict[str, Any] | None,
    calibration_path: str | None,
    q_mid: float,
    q_high: float,
    termination_token_ids: list[int],
    enable_debug: bool,
) -> tuple[str, int, list[dict[str, Any]]]:
    q_mid_effective, q_high_effective = _resolve_q_thresholds(
        schedule_mode=schedule_mode,
        q_conf_mode=q_conf_mode,
        q_mid_default=q_mid,
        q_high_default=q_high,
    )

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

    with torch.inference_mode():
        steerer.register_hooks(model)
        try:
            for block_idx in range(num_blocks):
                block_start = prompt_len + block_idx * block_length
                block_end = prompt_len + (block_idx + 1) * block_length
                early_end = block_start + max(0, gen_length - 10)

                while bool((x[:, block_start:block_end] == mask_id).any().item()):
                    total_denoising_steps += 1

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
                    baseline_scores_raw = baseline_scores.clone()
                    baseline_scores[:, block_end:] = float("-inf")
                    baseline_scores = torch.where(
                        mask_index,
                        baseline_scores,
                        torch.full_like(baseline_scores, float("-inf")),
                    )

                    provisional = torch.where(mask_index, x0, x)
                    provisional_full = tokenizer.batch_decode(
                        provisional, skip_special_tokens=True
                    )[0]
                    provisional_generated_only = tokenizer.batch_decode(
                        provisional[:, prompt_len:],
                        skip_special_tokens=True,
                    )[0]

                    s_t_full = _prob_negative_single(provisional_full, device=SENTIMENT_DEVICE)
                    s_t_generated_only = _prob_negative_single(
                        provisional_generated_only,
                        device=SENTIMENT_DEVICE,
                    )

                    if schedule_score_source == "provisional_full":
                        s_t = s_t_full
                    elif schedule_score_source == "provisional_generated_only":
                        s_t = s_t_generated_only
                    else:
                        raise ValueError(f"Unknown schedule_score_source: {schedule_score_source}")

                    transfer_index = torch.zeros_like(x, dtype=torch.bool)
                    chosen_u_t = 0
                    chosen_q_t = None
                    chosen_phase = None
                    chosen_sent_cal = None
                    chosen_conf_cal = None
                    chosen_mix_score = None
                    for b in range(batch_size):
                        block_mask = x[b, block_start:block_end] == mask_id
                        M_t = int(block_mask.sum().item())
                        if M_t <= 0:
                            continue

                        if schedule_mode == "mix_adaptive":
                            if q_conf_mode is None:
                                raise ValueError("q_conf_mode is required for mix_adaptive")

                            q_t = _compute_q_t(
                                q_conf_mode=q_conf_mode,
                                baseline_scores_raw_row=baseline_scores_raw[b],
                                block_start=block_start,
                                block_end=block_end,
                                block_mask_row=block_mask,
                            )
                            if s_t >= (2.0 / 3.0) and q_t >= q_high_effective:
                                u_base = 3
                            elif s_t >= (1.0 / 3.0) and q_t >= q_mid_effective:
                                u_base = 2
                            else:
                                u_base = 1
                            chosen_q_t = float(q_t)
                        elif schedule_mode == "calibrated_mix":
                            if q_conf_mode not in CALIBRATED_MIX_Q_CONF_MODES:
                                raise ValueError(
                                    "calibrated_mix requires q_conf_mode in "
                                    f"{CALIBRATED_MIX_Q_CONF_MODES}, got {q_conf_mode!r}"
                                )
                            if mix_lambda is None:
                                raise ValueError("mix_lambda is required for calibrated_mix")
                            if calibration_payload is None:
                                raise ValueError("calibration_payload is required for calibrated_mix")

                            q_t = _compute_q_t(
                                q_conf_mode=q_conf_mode,
                                baseline_scores_raw_row=baseline_scores_raw[b],
                                block_start=block_start,
                                block_end=block_end,
                                block_mask_row=block_mask,
                            )
                            phase = _phase_from_masks(remaining_before=M_t, total_masks=int(block_length))
                            sent_cal = _calibrated_quantile(
                                calibration_payload=calibration_payload,
                                signal_key="s_t",
                                raw_value=float(s_t_generated_only),
                                phase=phase,
                            )
                            conf_cal = _calibrated_quantile(
                                calibration_payload=calibration_payload,
                                signal_key=q_conf_mode,
                                raw_value=float(q_t),
                                phase=phase,
                            )
                            mix_score = (float(mix_lambda) * sent_cal) + (
                                (1.0 - float(mix_lambda)) * conf_cal
                            )
                            mix_score = float(np.clip(mix_score, 0.0, 1.0))
                            u_base = 1 + int(np.floor(3.0 * min(mix_score, 1.0 - 1e-8)))

                            chosen_q_t = float(q_t)
                            chosen_phase = phase
                            chosen_sent_cal = float(sent_cal)
                            chosen_conf_cal = float(conf_cal)
                            chosen_mix_score = float(mix_score)
                        else:
                            u_base = _schedule_u_base(schedule_mode=schedule_mode, s_t=s_t)

                        u_t = min(u_base, M_t)
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
                                "s_t": float(s_t),
                                "s_t_full": float(s_t_full),
                                "s_t_generated_only": float(s_t_generated_only),
                                "schedule_score_source": schedule_score_source,
                                "schedule_mode": schedule_mode,
                                "q_conf_mode": q_conf_mode,
                                "mix_lambda": mix_lambda,
                                "calibration_path": calibration_path,
                                "q_mid": float(q_mid_effective),
                                "q_high": float(q_high_effective),
                                "q_t": chosen_q_t,
                                "phase": chosen_phase,
                                "sent_cal": chosen_sent_cal,
                                "conf_cal": chosen_conf_cal,
                                "mix_score": chosen_mix_score,
                                "u_base": int(u_base),
                                "u_t": int(chosen_u_t),
                                "remaining_masks_after_step": remaining_after,
                                "provisional_text_preview": provisional_full[:280],
                                "provisional_generated_only_preview": provisional_generated_only[:280],
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
    schedule_mode: str,
    schedule_score_source: str,
    q_conf_mode: str | None,
    mix_lambda: float | None,
    calibration_payload: dict[str, Any] | None,
    calibration_path: str | None,
    q_mid: float,
    q_high: float,
    termination_token_ids: list[int],
    prompts: list[str],
    model,
    tokenizer,
    vectors: dict[int, torch.Tensor],
    out_dir: Path,
    enable_debug: bool,
) -> dict[str, Any]:
    q_mid_effective, q_high_effective = _resolve_q_thresholds(
        schedule_mode=schedule_mode,
        q_conf_mode=q_conf_mode,
        q_mid_default=q_mid,
        q_high_default=q_high,
    )

    if schedule_mode == "calibrated_mix":
        run_tag = (
            f"a{alpha:.1f}__sched_{_schedule_tag(schedule_mode)}"
            f"__q_{_q_conf_tag(q_conf_mode)}"
            f"__lam_{_lambda_tag(mix_lambda)}"
        )
    else:
        run_tag = (
            f"a{alpha:.1f}__sched_{_schedule_tag(schedule_mode)}"
            f"__score_{_score_source_tag(schedule_score_source)}"
            f"__q_{_q_conf_tag(q_conf_mode)}"
        )
    run_base_dir = out_dir / run_tag
    if schedule_mode == "mix_adaptive":
        run_dir = run_base_dir / _threshold_subdir_name(
            q_mid=q_mid_effective,
            q_high=q_high_effective,
        )
    else:
        run_dir = run_base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    steerer = build_mask_only_steerer(model, tokenizer, alpha=alpha, vectors=vectors)

    answers: list[str] = []
    denoising_steps: list[int] = []
    debug_all: list[list[dict[str, Any]]] = []

    for prompt in tqdm(prompts, desc=f"run[a={alpha:.1f}, sched={schedule_mode}]", leave=False):
        answer, n_steps, dbg = generate_with_schedule_mode(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steerer=steerer,
            schedule_mode=schedule_mode,
            schedule_score_source=schedule_score_source,
            q_conf_mode=q_conf_mode,
            mix_lambda=mix_lambda,
            calibration_payload=calibration_payload,
            calibration_path=calibration_path,
            q_mid=q_mid_effective,
            q_high=q_high_effective,
            termination_token_ids=termination_token_ids,
            enable_debug=enable_debug,
        )
        answers.append(answer)
        denoising_steps.append(n_steps)
        debug_all.append(dbg)

    metrics = evaluate_texts_filtered(prompts, answers, denoising_steps)
    metrics["method"] = "mean_mask_only_schedule_compare"
    metrics["alpha"] = float(alpha)
    metrics["schedule_mode"] = schedule_mode
    metrics["schedule_score_source"] = schedule_score_source
    metrics["q_conf_mode"] = q_conf_mode
    metrics["mix_lambda"] = (None if mix_lambda is None else float(mix_lambda))
    metrics["calibration_path"] = calibration_path
    metrics["q_mid"] = float(q_mid_effective)
    metrics["q_high"] = float(q_high_effective)
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
            "schedule_mode": schedule_mode,
            "schedule_score_source": schedule_score_source,
            "q_conf_mode": q_conf_mode,
            "mix_lambda": (None if mix_lambda is None else float(mix_lambda)),
            "calibration_path": calibration_path,
            "q_mid": float(q_mid_effective),
            "q_high": float(q_high_effective),
            "output_subdir": str(run_dir.relative_to(out_dir)),
            "schedule": {
                "fixed_1": "always commit 1",
                "fixed_2": "always commit 2",
                "fixed_3": "always commit 3",
                "adaptive_123": "commit 1/2/3 by thirds of P(NEGATIVE)",
                "mix_adaptive": "if s_t>=2/3 and q_t>=q_high ->3, elif s_t>=1/3 and q_t>=q_mid ->2, else 1",
                "calibrated_mix": "u_base from thirds of lambda*cal(sentiment) + (1-lambda)*cal(confidence), where calibration is phase-aware empirical quantile",
                "score_source": schedule_score_source,
                "mix_lambda": (None if mix_lambda is None else float(mix_lambda)),
                "calibration_path": calibration_path,
                "scores_computed": ["provisional_full", "provisional_generated_only"],
                "q_thresholds": Q_THRESHOLDS,
                "q_t_definition": {
                    "mean_top3": "mean of top-3 baseline_scores among masked positions in current block",
                    "min_top3": "min of top-3 baseline_scores among masked positions in current block",
                    "mean_masked_block": "mean baseline_scores across masked positions in current block",
                    "mean_all_block": "mean baseline_scores across all positions in current block",
                },
                "calibrated_mix_q_modes": CALIBRATED_MIX_Q_CONF_MODES,
                "ranking": "standard low_confidence baseline",
                "block_loop": "while masked tokens remain in block",
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/speed_adapt/compare_mean_steering_adaptive_schedule_instruct"),
    )
    parser.add_argument(
        "--calibration-base-dir",
        type=Path,
        default=CALIBRATION_BASE_DIR,
        help="Base directory containing a{alpha}/calibration.json files from debug_calibrate_speed_signals.py",
    )
    parser.add_argument("--alphas", nargs="*", type=float, default=ALPHAS)
    parser.add_argument("--schedule-modes", nargs="*", type=str, default=SCHEDULE_MODES)
    parser.add_argument(
        "--schedule-score-sources",
        nargs="*",
        type=str,
        default=SCHEDULE_SCORE_SOURCES,
    )
    parser.add_argument("--q-conf-modes", nargs="*", type=str, default=Q_CONF_MODES)
    parser.add_argument("--lambdas", nargs="*", type=float, default=LAMBDA_VALUES)
    parser.add_argument("--q-mid", type=float, default=Q_MID)
    parser.add_argument("--q-high", type=float, default=Q_HIGH)
    parser.add_argument(
        "--neutral-eval-set-path",
        type=Path,
        default=NEUTRAL_EVAL_SET_PATH,
    )
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
        prompts = prompts[: args.smoke_max_prompts]
        labels = labels[: args.smoke_max_prompts]
        dataset_indices = dataset_indices[: args.smoke_max_prompts]

    _write_json(
        out_dir / "eval_set.json",
        {
            "meta": eval_meta,
            "dataset_indices": dataset_indices,
            "labels": labels,
            "prompts": prompts,
            "formatted_prompts": [format_prompt(p) for p in prompts],
            "smoke_test": bool(args.smoke_test),
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

    alphas = SMOKE_ALPHAS if args.smoke_test else [float(a) for a in args.alphas]
    schedule_modes = SMOKE_SCHEDULE_MODES if args.smoke_test else list(args.schedule_modes)
    schedule_score_sources = (
        SMOKE_SCHEDULE_SCORE_SOURCES
        if args.smoke_test
        else list(args.schedule_score_sources)
    )
    q_conf_modes = SMOKE_Q_CONF_MODES if args.smoke_test else list(args.q_conf_modes)
    lambdas = SMOKE_LAMBDAS if args.smoke_test else [float(x) for x in args.lambdas]

    allowed_modes = set(ALL_SCHEDULE_MODES)
    bad_modes = [m for m in schedule_modes if m not in allowed_modes]
    if bad_modes:
        raise ValueError(f"Unknown schedule modes: {bad_modes}. Allowed: {sorted(allowed_modes)}")

    allowed_score_sources = set(SCHEDULE_SCORE_SOURCES)
    bad_score_sources = [s for s in schedule_score_sources if s not in allowed_score_sources]
    if bad_score_sources:
        raise ValueError(
            f"Unknown schedule score sources: {bad_score_sources}. Allowed: {sorted(allowed_score_sources)}"
        )

    allowed_q_modes = set(ALL_Q_CONF_MODES)
    bad_q_modes = [q for q in q_conf_modes if q not in allowed_q_modes]
    if bad_q_modes:
        raise ValueError(f"Unknown q_conf_modes: {bad_q_modes}. Allowed: {sorted(allowed_q_modes)}")

    bad_lambdas = [lam for lam in lambdas if lam < 0.0 or lam > 1.0]
    if bad_lambdas:
        raise ValueError(f"All lambdas must be in [0, 1], got: {bad_lambdas}")

    calibration_by_alpha: dict[float, dict[str, Any]] = {}
    calibration_path_by_alpha: dict[float, str] = {}
    if "calibrated_mix" in schedule_modes:
        for alpha in alphas:
            payload, path = _load_calibration_for_alpha(args.calibration_base_dir, float(alpha))
            calibration_by_alpha[float(alpha)] = payload
            calibration_path_by_alpha[float(alpha)] = str(path)

    summary: dict[str, Any] = {
        "model_name": args.model_name,
        "device": DEVICE,
        "sentiment_device": SENTIMENT_DEVICE,
        "out_dir": str(out_dir),
        "vector_meta": vector_meta,
        "steer_layers": STEER_LAYERS,
        "prompt_words": PROMPT_WORDS,
        "gen_params": GEN_PARAMS,
        "alphas": alphas,
        "schedule_modes": schedule_modes,
        "q_conf_modes": q_conf_modes,
        "lambdas": lambdas,
        "q_mid": float(args.q_mid),
        "q_high": float(args.q_high),
        "calibration_base_dir": str(args.calibration_base_dir),
        "calibration_by_alpha": calibration_path_by_alpha,
        "q_thresholds": Q_THRESHOLDS,
        "q_thresholds_note": "mix_adaptive uses per-q_conf_mode thresholds from q_thresholds; non-mix modes use q_mid/q_high",
        "schedule_score_sources_requested": schedule_score_sources,
        "schedule_score_sources_note": "fixed schedules run once only (provisional_full); mix_adaptive and calibrated_mix always use provisional_generated_only",
        "scores_computed": ["provisional_full", "provisional_generated_only"],
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
        "method,alpha,schedule_mode,schedule_score_source,q_conf_mode,mix_lambda,calibration_path,q_mid,q_high,n_total,n_valid,n_invalid,invalid_fraction,bad_pattern_fraction,repetition_fraction,empty_fraction,sent_answer_mean,sent_answer_fraction,sent_combined_mean,ppl_answer_mean,ppl_combined_mean,mean_answer_words_all,mean_answer_words_valid_only,total_denoising_steps,avg_denoising_steps_per_generation"
    ]

    for alpha in alphas:
        for schedule_mode in schedule_modes:
            score_sources_this_mode = _score_sources_for_mode(
                schedule_mode=schedule_mode,
                requested_sources=schedule_score_sources,
            )
            q_modes_this_mode = _q_conf_modes_for_mode(
                schedule_mode=schedule_mode,
                requested_q_modes=q_conf_modes,
            )
            lambdas_this_mode = _lambdas_for_mode(
                schedule_mode=schedule_mode,
                requested_lambdas=lambdas,
            )

            for schedule_score_source in score_sources_this_mode:
                for q_conf_mode in q_modes_this_mode:
                    for mix_lambda in lambdas_this_mode:
                        if schedule_mode == "calibrated_mix" and q_conf_mode not in CALIBRATED_MIX_Q_CONF_MODES:
                            continue

                        if schedule_mode == "calibrated_mix":
                            run_key = (
                                f"a{alpha:.1f}__sched_{_schedule_tag(schedule_mode)}"
                                f"__q_{_q_conf_tag(q_conf_mode)}"
                                f"__lam_{_lambda_tag(mix_lambda)}"
                            )
                        else:
                            run_key = (
                                f"a{alpha:.1f}__sched_{_schedule_tag(schedule_mode)}"
                                f"__score_{_score_source_tag(schedule_score_source)}"
                                f"__q_{_q_conf_tag(q_conf_mode)}"
                            )

                        print(
                            f"[run] alpha={alpha:.1f} schedule_mode={schedule_mode} "
                            f"schedule_score_source={schedule_score_source} q_conf_mode={q_conf_mode} "
                            f"mix_lambda={mix_lambda}"
                        )
                        metrics = run_single_condition(
                            alpha=float(alpha),
                            schedule_mode=schedule_mode,
                            schedule_score_source=schedule_score_source,
                            q_conf_mode=q_conf_mode,
                            mix_lambda=mix_lambda,
                            calibration_payload=calibration_by_alpha.get(float(alpha)),
                            calibration_path=calibration_path_by_alpha.get(float(alpha)),
                            q_mid=float(args.q_mid),
                            q_high=float(args.q_high),
                            termination_token_ids=termination_token_ids,
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
                                    str(metrics["schedule_mode"]),
                                    str(metrics["schedule_score_source"]),
                                    str(metrics["q_conf_mode"]),
                                    str(metrics["mix_lambda"]),
                                    str(metrics["calibration_path"]),
                                    str(metrics["q_mid"]),
                                    str(metrics["q_high"]),
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
        summary["overall_avg_denoising_steps_per_generation"] = (
            float(np.mean(vals)) if vals else None
        )

    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.csv").write_text("\n".join(aggregate_rows) + "\n")
    print(f"[done] wrote outputs under {out_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()