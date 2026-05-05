from __future__ import annotations

"""
Build calibration data for no-manual-bins scheduling.

Collect per-step signals under fixed_2 denoising:
- s_t (P(NEGATIVE) on provisional_generated_only)
- mean_top3
- min_top3

The script saves raw records plus reusable calibration artifacts (stats + empirical CDF)
so later inference can map raw values to quantiles in [0, 1].
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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.sentiment_metrics import compute_sentiment_metrics
from llada.model import load_model, load_tokenizer
from steering.sus_fix import MeanActivationSteeringMaskedOnly


SEED = 42
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
SENTIMENT_DEVICE = os.getenv("SENTIMENT_DEVICE", "cpu")

PROMPT_WORDS = 25
STEER_LAYERS = list(range(9, 25))
MASK_ID = 126336

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=60,
    block_length=60,
    fill_strategy="low_confidence",
)

SCHEDULE_MODE = "fixed_2"
FIXED_U = 2
DEFAULT_ALPHAS = [1]

NEUTRAL_EVAL_SET_PATH = Path(
    "data/speed_adapt/neutral_prefix_eval_set/full_run/neutral_prefix_eval_set.json"
)
DEFAULT_OUT_DIR = Path("data/speed_adapt/debug_calibrate_speed_signals")
DEFAULT_ACTIVATIONS_ROOT = Path(
    "data/speed_adapt/instruct_real_sentiment_activations/full_run/activations"
)

SIGNAL_KEYS = ["s_t", "mean_top3", "min_top3"]


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

    raise ValueError(
        f"Unknown fill_strategy {fill_strategy!r}. "
        "Expected 'low_confidence' or 'random'."
    )


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
            ban_positions,
            neg_inf,
            logits_mod[..., tok_id],
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


def _compute_top3_signals(
    *,
    baseline_scores_raw_row: torch.Tensor,
    block_start: int,
    block_end: int,
    block_mask_row: torch.Tensor,
) -> dict[str, float]:
    block_scores = baseline_scores_raw_row[block_start:block_end]
    masked_vals = block_scores[block_mask_row]
    if masked_vals.numel() <= 0:
        return {
            "mean_top3": 0.0,
            "min_top3": 0.0,
        }

    k = min(3, int(masked_vals.numel()))
    top_vals, _ = torch.topk(masked_vals.float(), k=k)
    return {
        "mean_top3": float(top_vals.mean().item()),
        "min_top3": float(top_vals.min().item()),
    }


def _phase_from_masks(remaining_before: int, total_masks: int) -> str:
    if total_masks <= 0:
        return "unknown"
    progress = 1.0 - (float(remaining_before) / float(total_masks))
    if progress < (1.0 / 3.0):
        return "early"
    if progress < (2.0 / 3.0):
        return "middle"
    return "late"


def _compute_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def empirical_quantile_from_sorted(raw_value: float, sorted_values: list[float]) -> float:
    """
    Map a raw value to its empirical quantile in [0, 1] using sorted calibration values.
    """
    if not sorted_values:
        return float("nan")
    arr = np.asarray(sorted_values, dtype=np.float64)
    idx = int(np.searchsorted(arr, raw_value, side="right"))
    return float(idx / arr.size)


def _build_empirical_cdf_artifact(values: list[float], grid_size: int = 201) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "sorted_values": [],
            "cdf_grid": [],
        }

    arr = np.sort(np.asarray(values, dtype=np.float64))
    quantiles = np.linspace(0.0, 1.0, num=grid_size)
    cdf_grid = []
    for q in quantiles:
        idx = min(int(round(q * (arr.size - 1))), arr.size - 1)
        cdf_grid.append({"quantile": float(q), "value": float(arr[idx])})

    return {
        "n": int(arr.size),
        "sorted_values": [float(v) for v in arr.tolist()],
        "cdf_grid": cdf_grid,
    }


def _plot_hist(values: list[float], title: str, out_path: Path, bins: int) -> None:
    if not HAS_MATPLOTLIB:
        return
    if not values:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, alpha=0.85)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_hist_by_phase(
    records: list[dict[str, Any]],
    signal_key: str,
    out_path: Path,
    bins: int,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    phase_to_values: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    for rec in records:
        phase = rec.get("phase")
        if phase not in phase_to_values:
            continue
        val = rec.get(signal_key)
        if val is None:
            continue
        phase_to_values[phase].append(float(val))

    if not any(phase_to_values.values()):
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for phase, color in [("early", "tab:blue"), ("middle", "tab:orange"), ("late", "tab:green")]:
        vals = phase_to_values[phase]
        if not vals:
            continue
        plt.hist(vals, bins=bins, alpha=0.4, label=f"{phase} (n={len(vals)})", color=color)
    plt.title(f"{signal_key} histogram by denoising phase")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def generate_and_collect_signals(
    *,
    model,
    tokenizer,
    prompt: str,
    prompt_index: int,
    dataset_index: int,
    steerer,
    termination_token_ids: list[int],
    preview_chars: int,
    sentiment_device: str,
) -> list[dict[str, Any]]:
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

    records: list[dict[str, Any]] = []
    global_step = 0

    with torch.inference_mode():
        steerer.register_hooks(model)
        try:
            for block_idx in range(num_blocks):
                block_start = prompt_len + block_idx * block_length
                block_end = prompt_len + (block_idx + 1) * block_length
                early_end = block_start + max(0, gen_length - 10)
                block_step = 0

                while bool((x[:, block_start:block_end] == mask_id).any().item()):
                    global_step += 1
                    block_step += 1

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
                    provisional_generated_only = tokenizer.batch_decode(
                        provisional[:, prompt_len:],
                        skip_special_tokens=True,
                    )[0]
                    s_t = _prob_negative_single(provisional_generated_only, device=sentiment_device)

                    transfer_index = torch.zeros_like(x, dtype=torch.bool)
                    for b in range(batch_size):
                        block_mask = x[b, block_start:block_end] == mask_id
                        remaining_before = int(block_mask.sum().item())
                        if remaining_before <= 0:
                            continue

                        signals = _compute_top3_signals(
                            baseline_scores_raw_row=baseline_scores_raw[b],
                            block_start=block_start,
                            block_end=block_end,
                            block_mask_row=block_mask,
                        )

                        u_t = min(FIXED_U, remaining_before)
                        _, selected_pos = torch.topk(baseline_scores[b], k=u_t)
                        transfer_index[b, selected_pos] = True

                        records.append(
                            {
                                "prompt_index": int(prompt_index),
                                "dataset_index": int(dataset_index),
                                "global_step": int(global_step),
                                "block_step": int(block_step),
                                "block_idx": int(block_idx),
                                "schedule_mode": SCHEDULE_MODE,
                                "u_t": int(u_t),
                                "remaining_masks_before_step": int(remaining_before),
                                "total_masks_in_block": int(block_length),
                                "phase": _phase_from_masks(remaining_before, int(block_length)),
                                "s_t": float(s_t),
                                "mean_top3": float(signals["mean_top3"]),
                                "min_top3": float(signals["min_top3"]),
                                "provisional_generated_only_preview": provisional_generated_only[:preview_chars],
                            }
                        )

                    x[transfer_index] = x0[transfer_index]
                    remaining_after = int((x[:, block_start:block_end] == mask_id).sum().item())
                    if records:
                        records[-1]["remaining_masks_after_step"] = int(remaining_after)
        finally:
            steerer.remove_hooks()

    return records


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
        default=DEFAULT_ACTIVATIONS_ROOT,
        help="Directory containing real_negative.pt and real_positive.pt.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--neutral-eval-set-path", type=Path, default=NEUTRAL_EVAL_SET_PATH)
    parser.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-prompts", type=int, default=4)
    parser.add_argument("--plot-bins", type=int, default=60)
    parser.add_argument("--preview-chars", type=int, default=280)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--sentiment-device", type=str, default=SENTIMENT_DEVICE)
    parser.add_argument("--cdf-grid-size", type=int, default=201)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    model_device = DEFAULT_DEVICE if args.device == "auto" else args.device

    out_root = args.out_dir / ("smoke_test" if args.smoke_test else "full_run")
    out_root.mkdir(parents=True, exist_ok=True)

    prompts, labels, dataset_indices, eval_meta = load_neutral_eval_prompts(args.neutral_eval_set_path)
    n_target = min(len(prompts), args.smoke_max_prompts if args.smoke_test else args.max_prompts)
    prompts = prompts[:n_target]
    labels = labels[:n_target]
    dataset_indices = dataset_indices[:n_target]

    _write_json(
        out_root / "eval_set.json",
        {
            "meta": eval_meta,
            "n_selected_prompts": int(n_target),
            "dataset_indices": dataset_indices,
            "labels": labels,
            "prompts": prompts,
            "prompt_words": PROMPT_WORDS,
            "smoke_test": bool(args.smoke_test),
        },
    )

    vectors, vector_meta = load_or_build_mean_diff_vectors(
        vector_path=args.vector_path,
        activations_root=args.activations_root,
        layer_ids=STEER_LAYERS,
    )

    print(f"[model] loading {args.model_name} on {model_device}")
    model = load_model(model_name=args.model_name, device=model_device)
    tokenizer = load_tokenizer(model_name=args.model_name)
    ensure_encoder_layer_compat(model)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = _infer_eot_token_id(tokenizer)
    termination_token_ids = sorted({int(t) for t in [eos_id, eot_id] if t is not None})

    all_alpha_summaries: dict[str, Any] = {}

    for alpha in [float(a) for a in args.alphas]:
        alpha_tag = f"a{alpha:.1f}"
        alpha_dir = out_root / alpha_tag
        alpha_dir.mkdir(parents=True, exist_ok=True)

        steerer = build_mask_only_steerer(
            model=model,
            tokenizer=tokenizer,
            alpha=alpha,
            vectors=vectors,
        )

        all_records: list[dict[str, Any]] = []
        signals_path = alpha_dir / "signals.jsonl"
        with signals_path.open("w") as f:
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"calibrate_signals[{alpha_tag}]", leave=False)):
                prompt_records = generate_and_collect_signals(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    prompt_index=prompt_idx,
                    dataset_index=dataset_indices[prompt_idx],
                    steerer=steerer,
                    termination_token_ids=termination_token_ids,
                    preview_chars=int(args.preview_chars),
                    sentiment_device=args.sentiment_device,
                )
                all_records.extend(prompt_records)
                for rec in prompt_records:
                    f.write(json.dumps(rec) + "\n")

        signal_values = {
            key: [float(rec[key]) for rec in all_records if rec.get(key) is not None]
            for key in SIGNAL_KEYS
        }

        stats = {key: _compute_stats(vals) for key, vals in signal_values.items()}

        phase_stats: dict[str, dict[str, Any]] = {}
        calibration_by_phase: dict[str, dict[str, Any]] = {}
        for phase in ["early", "middle", "late"]:
            phase_rows = [rec for rec in all_records if rec.get("phase") == phase]
            phase_stats[phase] = {
                key: _compute_stats([float(rec[key]) for rec in phase_rows if rec.get(key) is not None])
                for key in SIGNAL_KEYS
            }
            calibration_by_phase[phase] = {
                key: _build_empirical_cdf_artifact(
                    [float(rec[key]) for rec in phase_rows if rec.get(key) is not None],
                    grid_size=int(args.cdf_grid_size),
                )
                for key in SIGNAL_KEYS
            }

        calibration_payload = {
            "alpha": alpha,
            "signal_definitions": {
                "s_t": "P(NEGATIVE) on provisional_generated_only",
                "mean_top3": "mean of top-3 baseline_scores among masked positions in current block",
                "min_top3": "min of top-3 baseline_scores among masked positions in current block",
            },
            "overall": {
                key: {
                    "stats": stats[key],
                    "empirical_cdf": _build_empirical_cdf_artifact(
                        signal_values[key],
                        grid_size=int(args.cdf_grid_size),
                    ),
                }
                for key in SIGNAL_KEYS
            },
            "by_phase": {
                phase: {
                    key: {
                        "stats": phase_stats[phase][key],
                        "empirical_cdf": calibration_by_phase[phase][key],
                    }
                    for key in SIGNAL_KEYS
                }
                for phase in ["early", "middle", "late"]
            },
            "quantile_helper": {
                "name": "empirical_quantile_from_sorted",
                "formula": "quantile = searchsorted(sorted_values, raw_value, side='right') / n",
            },
        }
        _write_json(alpha_dir / "calibration.json", calibration_payload)

        plots_dir = alpha_dir / "plots"
        if HAS_MATPLOTLIB:
            for key in SIGNAL_KEYS:
                _plot_hist(
                    signal_values[key],
                    title=f"{key} histogram ({alpha_tag})",
                    out_path=plots_dir / f"hist_{key}.png",
                    bins=int(args.plot_bins),
                )
                _plot_hist_by_phase(
                    all_records,
                    signal_key=key,
                    out_path=plots_dir / f"hist_{key}_by_phase.png",
                    bins=max(20, int(args.plot_bins) // 2),
                )
        else:
            print("[warn] matplotlib not available; skipping plot generation")

        summary = {
            "model_name": args.model_name,
            "model_device": model_device,
            "sentiment_device": args.sentiment_device,
            "out_dir": str(alpha_dir),
            "vector_meta": vector_meta,
            "schedule_mode": SCHEDULE_MODE,
            "fixed_u": FIXED_U,
            "alpha": alpha,
            "prompt_words": PROMPT_WORDS,
            "steer_layers": STEER_LAYERS,
            "gen_params": GEN_PARAMS,
            "termination_policy": {
                "termination_token_ids": termination_token_ids,
                "policy": "EOS/EOT banned in generated positions 0..49, allowed in 50..59",
                "gen_length": GEN_PARAMS["gen_length"],
                "block_length": GEN_PARAMS["block_length"],
            },
            "score_definition": {
                "s_t": "P(NEGATIVE) on provisional_generated_only",
                "mean_top3": "mean of top-3 baseline_scores among masked positions in current block",
                "min_top3": "min of top-3 baseline_scores among masked positions in current block",
            },
            "eval_meta": eval_meta,
            "n_selected_prompts": int(n_target),
            "n_step_records": int(len(all_records)),
            "signals_stats": stats,
            "signals_stats_by_phase": phase_stats,
            "files": {
                "signals_jsonl": str(signals_path),
                "calibration_json": str(alpha_dir / "calibration.json"),
                "plots_dir": str(plots_dir),
            },
            "plotting": {
                "matplotlib_available": bool(HAS_MATPLOTLIB),
                "plots_generated": bool(HAS_MATPLOTLIB),
            },
            "smoke_test": bool(args.smoke_test),
        }
        _write_json(alpha_dir / "summary.json", summary)
        all_alpha_summaries[alpha_tag] = summary

    _write_json(
        out_root / "summary_all_alphas.json",
        {
            "model_name": args.model_name,
            "model_device": model_device,
            "sentiment_device": args.sentiment_device,
            "schedule_mode": SCHEDULE_MODE,
            "fixed_u": FIXED_U,
            "alphas": [float(a) for a in args.alphas],
            "alpha_summaries": all_alpha_summaries,
            "smoke_test": bool(args.smoke_test),
        },
    )

    print(f"[done] wrote outputs under {out_root}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
