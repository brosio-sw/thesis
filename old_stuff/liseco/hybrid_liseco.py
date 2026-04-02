from __future__ import annotations

"""
eval_liseco_fixed_diffmean_direction.py

Hybrid LiSeCo experiment (Option 1): keep the LiSeCo controller, but replace the
learned probe direction with the saved DiffMean direction.

For each layer and variant:
  1) load the already-collected activations from out_root/activations/<variant>
  2) load the saved mean-diff vector v from out_root/probes/<variant>/mean_diff_vectors.pt
  3) fit a 1D calibrator on the scalar score s = v^T x
       p(negative | x) = sigmoid(a * s + b)
  4) convert that back into an ordinary linear probe
       w_eff = a * v,  b_eff = b
  5) reuse the standard LiSeCoProbeSteering controller with those fixed-direction
     probes

This script does NOT recollect activations.
It does retrain the scalar calibrators from the saved activations.

Evaluation:
- same test set logic as eval_liseco_vs_diffmean_again.py
- filtered metrics: sentiment/perplexity on valid generations only
- compare:
    * DiffMean, alpha=8.0, layers 9..24
    * Hybrid LiSeCo (fixed DiffMean direction), layers 9..24
    * Hybrid LiSeCo (fixed DiffMean direction), all available probe layers
- variants default to:
    * real_full_pooled
    * masked_pooled

Example:
    python eval_liseco_fixed_diffmean_direction.py \
        --out-root data/alignment_variants_v4/full_run
"""

import argparse
import gc
import json
import os
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity
from steering.precomputed_steering import PrecomputedLayerSteering
from steering.liseco_probe_steering import LiSeCoProbeSteering, ProbeParams


# -----------------------------------------------------------------------------
# Defaults aligned with the user's existing scripts
# -----------------------------------------------------------------------------

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
MASK_ID = 126336

PROMPT_WORDS = 20
EVAL_NEG = 25
EVAL_POS = 25
EVAL_ANY = 50

VARIANTS = ["real_full_pooled", "masked_pooled"]
STEER_LAYERS = list(range(9, 25))
MEAN_ALPHA = 8.0
LISECO_INTERVALS = [
    (0.00, 0.20),
    (0.40, 0.60),
    (0.80, 1.00),
]

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",
)

PROBE_VAL_FRAC = 0.2
PROBE_EPOCHS = 40
PROBE_BATCH_SIZE = 256
PROBE_LR = 2e-3
PROBE_WEIGHT_DECAY = 1e-4

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
# Basic helpers
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



def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)



def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    na = torch.linalg.norm(a).item()
    nb = torch.linalg.norm(b).item()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))



def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot



def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = float(np.sqrt((yt ** 2).sum()) * np.sqrt((yp ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((yt * yp).sum() / denom)



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


# -----------------------------------------------------------------------------
# Prompt loading (same logic as eval_liseco_vs_diffmean_again.py)
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
# Saved activation loading + targets (same conventions as liseco_vs_diffmean.py)
# -----------------------------------------------------------------------------


def load_variant_dataset(act_root: Path, variant: str) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    variant_dir = act_root / variant
    part_files = sorted(variant_dir.glob("part*.pt"))
    if not part_files:
        raise RuntimeError(f"No activation parts found for {variant} in {variant_dir}")

    act_parts: list[dict[int, torch.Tensor]] = []
    rows_all: list[dict[str, Any]] = []
    for pt in part_files:
        meta = _read_json(pt.with_name(pt.stem + "_meta.json"))
        act = torch.load(pt, map_location="cpu", weights_only=False)
        act = {int(k): v.float().cpu() for k, v in act.items()}
        act_parts.append(act)
        rows_all.extend(meta["rows"])

    layer_ids = sorted(act_parts[0].keys())
    acts = {li: torch.cat([part[li] for part in act_parts], dim=0) for li in layer_ids}
    return acts, rows_all



def build_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(r["target_negative"]) for r in rows], dtype=np.float32)



def grouped_split_by_text(rows: list[dict[str, Any]], val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    by_group: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        gid = f"text::{row['source_text_idx']}"
        by_group.setdefault(gid, []).append(i)
    gids = list(by_group.keys())
    rng.shuffle(gids)
    n_val = max(1, int(round(len(gids) * val_frac)))
    if len(gids) - n_val < 1:
        n_val = len(gids) - 1
    val_gids = set(gids[:n_val])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for gid, idxs in by_group.items():
        if gid in val_gids:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


# -----------------------------------------------------------------------------
# Option 1 training: fixed-direction 1D calibrator on saved activations
# -----------------------------------------------------------------------------


def train_fixed_direction_single_layer(
    X: torch.Tensor,
    v: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], ProbeParams]:
    """
    Fit p(negative|x) = sigmoid(a * (v^T x) + b), where v is fixed.
    Returns metrics, raw calibrator params, and effective ProbeParams with
    w_eff = a * v, b_eff = b.
    """
    _seed_everything(seed)

    device = "cpu"
    v = v.float().cpu().view(-1)
    X = X.float().cpu()
    y = y.float().cpu()

    s_all = (X @ v).view(-1, 1)
    s_train = s_all[train_idx].to(device)
    y_train = y[train_idx].to(device)
    s_val = s_all[val_idx].to(device)
    y_val = y[val_idx].to(device)

    ds = TensorDataset(s_train, y_train)
    dl = DataLoader(ds, batch_size=min(PROBE_BATCH_SIZE, len(ds)), shuffle=True)
    model = nn.Linear(1, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best = {"val_mse": float("inf"), "state_dict": None, "epoch": -1}
    for epoch in range(PROBE_EPOCHS):
        model.train()
        for sb, yb in dl:
            pred = torch.sigmoid(model(sb)).squeeze(-1)
            loss = criterion(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = torch.sigmoid(model(s_val)).squeeze(-1)
            val_mse = float(nn.functional.mse_loss(pred_val, y_val).item())
        if val_mse < best["val_mse"]:
            best["val_mse"] = val_mse
            best["state_dict"] = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
            best["epoch"] = epoch

    model.load_state_dict(best["state_dict"])
    model.eval()

    with torch.no_grad():
        pred_val = torch.sigmoid(model(s_val)).squeeze(-1).cpu().numpy().astype(np.float32)
        y_val_np = y_val.cpu().numpy().astype(np.float32)

    a = model.weight.detach().cpu().view(-1)[0].float()
    b = model.bias.detach().cpu().view(-1)[0].float()
    w_eff = (a * v).float().cpu().view(-1)
    b_eff = torch.tensor([float(b.item())], dtype=torch.float32)
    probe = ProbeParams(weight=w_eff, bias=b_eff, norm_sq=float((w_eff * w_eff).sum().item()))

    metrics = {
        "best_epoch": int(best["epoch"]),
        "val_mse": float(best["val_mse"]),
        "val_mae": float(np.mean(np.abs(pred_val - y_val_np))),
        "val_r2": float(_r2_score(y_val_np, pred_val)),
        "val_pearson": float(_pearson(y_val_np, pred_val)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "direction_scale_a": float(a.item()),
        "bias_b": float(b.item()),
        "effective_weight_norm": float(torch.linalg.norm(w_eff).item()),
        "cosine_effective_weight_vs_mean_diff": _cosine(w_eff, v),
    }
    artifact = {
        "direction_scale_a": a.view(1).cpu(),
        "bias": b_eff.cpu(),
        "effective_weight": w_eff.cpu(),
        "fixed_direction": v.cpu(),
    }
    return metrics, artifact, probe



def train_fixed_direction_probes_for_variant(
    out_root: Path,
    variant: str,
) -> tuple[dict[int, ProbeParams], list[int], dict[str, Any]]:
    act_root = out_root / "activations"
    probe_root = out_root / "probes"
    save_root = out_root / "probes_fixed_diffmean_direction" / variant
    save_root.mkdir(parents=True, exist_ok=True)

    acts, rows = load_variant_dataset(act_root=act_root, variant=variant)
    y_np = build_targets(rows)
    y = torch.tensor(y_np, dtype=torch.float32)
    train_idx, val_idx = grouped_split_by_text(rows, PROBE_VAL_FRAC, SEED)

    mean_diff = torch.load(
        probe_root / variant / "mean_diff_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )
    mean_diff = {int(k): v.float().cpu().view(-1) for k, v in mean_diff.items()}

    layer_ids = sorted(set(acts.keys()) & set(mean_diff.keys()))
    if not layer_ids:
        raise RuntimeError(f"No overlapping activation layers / mean-diff layers for {variant}")

    probe_by_layer: dict[int, ProbeParams] = {}
    layer_rows: list[dict[str, Any]] = []

    for li in layer_ids:
        X = acts[li]
        v = mean_diff[li]
        metrics, artifact, probe = train_fixed_direction_single_layer(
            X=X,
            v=v,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            seed=SEED + li,
        )
        probe_by_layer[li] = probe

        layer_dir = save_root / f"layer_{li:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, layer_dir / "probe_fixed_direction.pt")
        torch.save({"weight": probe.weight, "bias": probe.bias}, layer_dir / "probe_weight_bias.pt")
        _write_json(layer_dir / "metrics.json", metrics)

        layer_rows.append({"layer": li, **metrics})

    summary = {
        "variant": variant,
        "source": "saved_activations_plus_saved_mean_diff",
        "training_objective": "fit sigmoid(a * (v^T x) + b) to target_negative with fixed v = mean_diff",
        "n_rows": len(rows),
        "n_layers": len(layer_ids),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "target_definition": "target_negative from saved activation metadata",
        "layer_rows": layer_rows,
    }
    _write_json(save_root / "variant_summary.json", summary)
    return probe_by_layer, layer_ids, summary


# -----------------------------------------------------------------------------
# Generation / filtered evaluation
# -----------------------------------------------------------------------------


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
        generated_ids = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers.extend(decoded)
    return answers



def evaluate_texts_filtered(prompts: list[str], answers: list[str]) -> dict[str, Any]:
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
    }


# -----------------------------------------------------------------------------
# Per-variant eval
# -----------------------------------------------------------------------------


def run_eval_for_variant(
    variant: str,
    prompts: list[str],
    model,
    tokenizer,
    out_root: Path,
    eval_root: Path,
) -> dict[str, Any]:
    variant_eval_dir = eval_root / variant
    variant_eval_dir.mkdir(parents=True, exist_ok=True)

    hybrid_probe_by_layer, all_probe_layers, train_summary = train_fixed_direction_probes_for_variant(
        out_root=out_root,
        variant=variant,
    )
    _write_json(variant_eval_dir / "hybrid_probe_training_summary.json", train_summary)

    probe_root = out_root / "probes"
    mean_diff = torch.load(
        probe_root / variant / "mean_diff_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )
    mean_vectors_subset = {int(k): v for k, v in mean_diff.items() if int(k) in STEER_LAYERS}
    if len(mean_vectors_subset) != len(STEER_LAYERS):
        missing = [li for li in STEER_LAYERS if li not in mean_vectors_subset]
        raise RuntimeError(f"Variant {variant} missing mean-diff vectors for STEER_LAYERS: {missing}")

    variant_summary: dict[str, Any] = {
        "variant": variant,
        "fixed_direction_probe_training": {
            "train_rows": train_summary["train_rows"],
            "val_rows": train_summary["val_rows"],
            "n_layers": train_summary["n_layers"],
            "saved_dir": str(out_root / "probes_fixed_diffmean_direction" / variant),
        },
        "runs": {},
    }

    # DiffMean baseline on layers 9..24 only
    diffmean_tag = "diffmean__layers9-24__alpha8.0"
    diffmean_steerer = PrecomputedLayerSteering(
        vectors=mean_vectors_subset,
        layer_ids=STEER_LAYERS,
        alpha=MEAN_ALPHA,
    )
    diffmean_answers = generate_with_steering(
        model,
        tokenizer,
        prompts,
        diffmean_steerer,
        f"diffmean[{variant}]",
    )
    diffmean_metrics = evaluate_texts_filtered(prompts, diffmean_answers)
    diffmean_metrics.update(
        {
            "method": "diffmean",
            "variant": variant,
            "alpha": MEAN_ALPHA,
            "layer_config_name": "layers9-24",
            "layer_ids": STEER_LAYERS,
        }
    )
    _write_json(variant_eval_dir / f"{diffmean_tag}_metrics_valid_only.json", diffmean_metrics)
    variant_summary["runs"][diffmean_tag] = diffmean_metrics
    _write_json(variant_eval_dir / "summary_so_far.json", variant_summary)

    # Hybrid LiSeCo on two layer configs
    layer_configs = [
        ("layers9-24", [li for li in STEER_LAYERS if li in hybrid_probe_by_layer]),
        ("all_layers", list(sorted(all_probe_layers))),
    ]

    for layer_config_name, layer_ids in layer_configs:
        if not layer_ids:
            continue
        probe_subset = {li: hybrid_probe_by_layer[li] for li in layer_ids}
        for alpha_min, alpha_max in LISECO_INTERVALS:
            tag = (
                f"liseco_fixed_diffmean_direction__{layer_config_name}"
                f"__amin{alpha_min:.2f}_amax{alpha_max:.2f}"
            )
            steerer = LiSeCoProbeSteering(
                probe_by_layer=probe_subset,
                layer_ids=layer_ids,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                mask_id=MASK_ID,
            )
            answers = generate_with_steering(
                model,
                tokenizer,
                prompts,
                steerer,
                f"hybrid_liseco[{variant} {layer_config_name} {alpha_min:.2f}-{alpha_max:.2f}]",
            )
            metrics = evaluate_texts_filtered(prompts, answers)
            metrics.update(
                {
                    "method": "liseco_fixed_diffmean_direction",
                    "variant": variant,
                    "layer_config_name": layer_config_name,
                    "layer_ids": layer_ids,
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                }
            )
            _write_json(variant_eval_dir / f"{tag}_metrics_valid_only.json", metrics)
            variant_summary["runs"][tag] = metrics
            _write_json(variant_eval_dir / "summary_so_far.json", variant_summary)

    _write_json(variant_eval_dir / "summary.json", variant_summary)
    return variant_summary


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/alignment_variants_v4/full_run"),
        help="Existing run root containing activations/ and probes/.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=VARIANTS,
        help="Variants to evaluate. Defaults to real_full_pooled masked_pooled.",
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

    out_root = args.out_root
    eval_root = out_root / "eval_liseco_fixed_diffmean_direction_valid_only"
    eval_root.mkdir(parents=True, exist_ok=True)

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
        "out_root": str(out_root),
        "eval_root": str(eval_root),
        "model_name": args.model_name,
        "device": DEVICE,
        "evaluation_subset": "valid_generations_only",
        "diffmean_baseline": {
            "alpha": MEAN_ALPHA,
            "layer_ids": STEER_LAYERS,
        },
        "hybrid_liseco": {
            "training_objective": "fit sigmoid(a * (v^T x) + b) with fixed v = mean_diff",
            "intervals": LISECO_INTERVALS,
            "layer_configs": {
                "layers9-24": STEER_LAYERS,
                "all_layers": "all available layers from saved activations / mean_diff",
            },
        },
        "bad_patterns": BAD_PATTERNS,
        "variants": {},
    }
    _write_json(eval_root / "summary_so_far.json", summary)

    for variant in args.variants:
        print(f"\n[eval] variant={variant}")
        summary["variants"][variant] = run_eval_for_variant(
            variant=variant,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            out_root=out_root,
            eval_root=eval_root,
        )
        _write_json(eval_root / "summary_so_far.json", summary)

    _write_json(eval_root / "summary.json", summary)
    print(f"\n[done] summary -> {eval_root / 'summary.json'}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()