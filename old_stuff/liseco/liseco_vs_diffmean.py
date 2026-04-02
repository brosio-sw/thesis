from __future__ import annotations

"""
unified_steering_alignment_pipeline_v4.py

Unified diagnostic pipeline to compare mean-diff steering and LiSeCo-style
probe steering on LLaDA across three activation variants:

  1) real_full_pooled   : mean over all tokens of real comments
  2) masked_pooled      : mean over originally masked positions of corrupted inputs
  3) masked_tokenwise   : one row per originally masked token of corrupted inputs

Important change in this version
--------------------------------
masked_pooled and masked_tokenwise now use VERSION B:

- activations are captured DURING reconstruction,
- specifically from the FIRST denoising forward on the original masked inputs,
- then kept only for rows whose final reconstruction is valid.

So there is no extra second forward pass for masked variants.

This version follows the patterns that already work in your repo:
- real_full_pooled uses LABEL-BALANCED loading like get_activations.py
- eval prompts use a fixed balanced subset like mean_steering_test.py
- masked variants use MULTIPLE MASK LEVELS, reconstruction-time sentiment,
  and per-noise-sample filtering like get_masked_activations.py
- probe training follows train_probe_llada.py (grouped split by original text id,
  sigmoid linear probe trained with MSE)

Crash test
----------
Set CRASH_TEST = True for a tiny end-to-end run:
- real train texts: 1 negative + 1 positive  (2 total)
- eval prompts:     1 negative + 1 positive  (2 total)
"""

import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from llada.generate import _add_gumbel_noise, _get_num_transfer_tokens
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity
from steering.precomputed_steering import PrecomputedLayerSteering
from steering.liseco_probe_steering import LiSeCoProbeSteering, ProbeParams


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CRASH_TEST = False
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
MASK_ID = 126336

# Data sizes
REAL_TRAIN_EACH = 500
REAL_TRAIN_EACH_CRASH = 1
EVAL_NEG = 25
EVAL_POS = 25
EVAL_ANY = 50
EVAL_NEG_CRASH = 1
EVAL_POS_CRASH = 1
EVAL_ANY_CRASH = 0

N_WORDS = 50
PROMPT_WORDS = 20
CHUNK_SIZE = 5
NOISE_LEVELS = [0.25, 0.5, 0.75, 0.9]

RECON_PARAMS = dict(
    temperature=0.0,
    steps=30,
    fill_strategy="low_confidence",
)

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

STEER_LAYERS = list(range(9, 25))
MEAN_ALPHA = 8.0
LISECO_INTERVALS = [
    (0.00, 0.20),
    (0.80, 1.00),
]

OUT_ROOT = Path("data/alignment_variants_v4") / ("crash_test" if CRASH_TEST else "full_run")
ACT_ROOT = OUT_ROOT / "activations"
PROBE_ROOT = OUT_ROOT / "probes"
ALIGN_ROOT = OUT_ROOT / "alignment"
EVAL_ROOT = OUT_ROOT / "eval"

BAD_PATTERNS = [
    r"Is this .*?\?",
    r"positive or negative\?",
    r"\bAnswer:",
    r"\bYesTitle:",
    r"\bNoTitle:",
    r"\bTitle:",
    r"\bReview:",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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


def _read_json(path: Path) -> dict:
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
            if toks[i:i+n] == toks[i+n:i+2*n]:
                return True
    return False


def is_valid_completion(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if has_bad_pattern(t):
        return False
    if has_repetition_loop(t):
        return False
    return True


def iter_chunks(items: list, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        yield start, end, items[start:end]


def _get_transformer_layers(model) -> list:
    if hasattr(model, "model") and hasattr(model.model, "transformer") and "blocks" in model.model.transformer:
        return list(model.model.transformer["blocks"])
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError("Cannot locate transformer layers on model.")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_amazon_by_label(n: int, label: int, n_words: int) -> list[str]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    out = []
    for ex in ds:
        if int(ex["label"]) != label:
            continue
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        text = _truncate(text, n_words)
        if len(text.split()) >= max(5, PROMPT_WORDS + 5):
            out.append(text)
        if len(out) >= n:
            break
    return out


def load_amazon_prompts_by_label(n: int, label: int, n_words: int = PROMPT_WORDS) -> list[str]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    out = []
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


def load_amazon_prompts_any(n: int, skip_neg: int = 0, skip_pos: int = 0, n_words: int = PROMPT_WORDS) -> list[str]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    out = []
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


def load_real_train_texts() -> tuple[list[str], list[int]]:
    n_each = REAL_TRAIN_EACH_CRASH if CRASH_TEST else REAL_TRAIN_EACH
    neg = load_amazon_by_label(n_each, label=0, n_words=N_WORDS)
    pos = load_amazon_by_label(n_each, label=1, n_words=N_WORDS)
    texts = neg + pos
    labels = [0] * len(neg) + [1] * len(pos)
    rng = random.Random(SEED)
    order = list(range(len(texts)))
    rng.shuffle(order)
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    return texts, labels


def load_eval_prompts() -> tuple[list[str], list[str], list[int]]:
    n_neg = EVAL_NEG_CRASH if CRASH_TEST else EVAL_NEG
    n_pos = EVAL_POS_CRASH if CRASH_TEST else EVAL_POS
    n_any = EVAL_ANY_CRASH if CRASH_TEST else EVAL_ANY

    neg_prompts = load_amazon_prompts_by_label(n_neg, label=0)
    pos_prompts = load_amazon_prompts_by_label(n_pos, label=1)
    any_prompts = load_amazon_prompts_any(n_any, skip_neg=n_neg, skip_pos=n_pos)

    prompts = neg_prompts + pos_prompts + any_prompts
    labels = [0] * len(neg_prompts) + [1] * len(pos_prompts) + [-1] * len(any_prompts)

    rng = random.Random(SEED + 1)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    labels = [labels[i] for i in order]

    raw_texts = [p for p in prompts]
    return prompts, raw_texts, labels


# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION WITH VERSION-B CACHE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reconstruct_one_pass_with_cache(
    batch_input_ids: torch.Tensor,
    tokenizer,
    model,
    layers: list,
) -> tuple[list[str], dict[int, torch.Tensor]]:
    """
    One-pass masked reconstruction.

    For each masked input:
    - run exactly one forward pass
    - cache layer outputs from that pass
    - fill all masked positions at once using argmax logits
    - decode the completed sequence

    Returns:
        recon_texts: list[str] length B
        cache: {layer_idx: Tensor[B, L, D]} from the one forward pass
    """
    x = batch_input_ids.clone()
    device = x.device
    attention_mask = torch.ones_like(x, device=device)

    mask_index = (x == MASK_ID)
    if not mask_index.any():
        decoded = tokenizer.batch_decode(x, skip_special_tokens=True)
        return [t.strip() for t in decoded], {}

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()
        return hook

    hooks = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]

    try:
        logits = model(x, attention_mask=attention_mask).logits
        cache = {li: captured[li].clone() for li in range(len(layers))}
        captured.clear()

        logits_noisy = _add_gumbel_noise(logits, temperature=RECON_PARAMS["temperature"])
        x0 = logits_noisy.argmax(dim=-1)

        # fill all masked positions at once
        x_completed = torch.where(mask_index, x0, x)

        decoded = tokenizer.batch_decode(x_completed, skip_special_tokens=True)
        return [t.strip() for t in decoded], cache

    finally:
        for h in hooks:
            h.remove()
        captured.clear()


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_real_full_pooled_chunk(
    texts: list[str],
    labels: list[int],
    global_text_offset: int,
    model,
    tokenizer,
    device: str,
) -> tuple[dict[int, torch.Tensor], dict]:
    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    all_rows: dict[int, list[np.ndarray]] = {li: [] for li in range(n_layers)}
    meta_rows = []
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()
        return hook

    hooks = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]
    model.eval()
    try:
        for local_i, (text, label) in enumerate(tqdm(list(zip(texts, labels)), desc="[real_full] chunk", leave=False)):
            global_i = global_text_offset + local_i
            enc = tokenizer([text], add_special_tokens=False, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(device)

            _ = model(input_ids, attention_mask=attention_mask)
            for li in range(n_layers):
                hs = captured[li][0]
                all_rows[li].append(hs.mean(dim=0).numpy())
            meta_rows.append({
                "source_text_idx": global_i,
                "text": text,
                "label": int(label),
                "target_negative": float(1 - label),
            })
            captured.clear()
    finally:
        for h in hooks:
            h.remove()
        captured.clear()

    acts = {
        li: torch.tensor(np.stack(v), dtype=torch.float32) if len(v) > 0 else torch.empty((0, 0), dtype=torch.float32)
        for li, v in all_rows.items()
    }
    return acts, {"rows": meta_rows}


@torch.no_grad()
def extract_masked_variants_chunk(
    texts: list[str],
    global_text_offset: int,
    model,
    tokenizer,
    device: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict]:
    layers = _get_transformer_layers(model)
    n_layers = len(layers)

    pooled_rows: dict[int, list[np.ndarray]] = {li: [] for li in range(n_layers)}
    tokenwise_rows: dict[int, list[np.ndarray]] = {li: [] for li in range(n_layers)}
    meta_pooled = []
    meta_tokenwise = []

    model.eval()
    for local_ti, text in enumerate(tqdm(texts, desc="[masked] chunk", leave=False)):
        global_ti = global_text_offset + local_ti
        words = text.split()
        prompt_text = " ".join(words[:PROMPT_WORDS])

        enc_full = tokenizer([text], add_special_tokens=False, return_tensors="pt")
        enc_prompt = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")

        input_ids = enc_full["input_ids"][0]
        prompt_len = enc_prompt["input_ids"].shape[1]
        seq_len = input_ids.shape[0]
        prompt_len = min(prompt_len, seq_len - 5)
        cont_indices = list(range(prompt_len, seq_len))
        if len(cont_indices) < 2:
            continue

        batch_input_ids = []
        batch_mask_pos_list = []
        batch_ratio_list = []

        for ratio in NOISE_LEVELS:
            n_mask = max(1, int(len(cont_indices) * ratio))
            mask_pos = np.random.choice(cont_indices, size=n_mask, replace=False)

            corr = input_ids.clone()
            corr[mask_pos] = MASK_ID

            batch_input_ids.append(corr)
            batch_mask_pos_list.append(mask_pos)
            batch_ratio_list.append(float(ratio))

        batch_input_ids = torch.stack(batch_input_ids).to(device)

        # one-pass reconstruction + one-pass activation cache
        recon_texts, cache = reconstruct_one_pass_with_cache(
            batch_input_ids=batch_input_ids,
            tokenizer=tokenizer,
            model=model,
            layers=layers,
        )
        if not cache:
            continue

        valid_indices = [i for i, recon in enumerate(recon_texts) if is_valid_completion(recon)]
        if not valid_indices:
            continue

        valid_recons = [recon_texts[i] for i in valid_indices]
        sent = compute_sentiment_metrics(valid_recons, device=device)
        neg_scores = sent["scores"].tolist()

        for out_i, orig_i in enumerate(valid_indices):
            mask_pos = batch_mask_pos_list[orig_i]
            ratio = batch_ratio_list[orig_i]
            target_negative = float(neg_scores[out_i])
            reconstruction = valid_recons[out_i]

            for li in range(n_layers):
                hs_batch = cache[li]                    # [n_noise, L, D]
                hs_masked = hs_batch[orig_i, mask_pos]  # [n_mask, D]

                pooled_rows[li].append(hs_masked.mean(dim=0).numpy())

                for token_pos in mask_pos.tolist():
                    tokenwise_rows[li].append(hs_batch[orig_i, token_pos].numpy())

            meta_pooled.append({
                "source_text_idx": global_ti,
                "text": text,
                "mask_ratio": float(ratio),
                "target_negative": target_negative,
                "reconstruction": reconstruction,
                "target_label": int(target_negative >= 0.5),
            })

            for token_pos in mask_pos.tolist():
                meta_tokenwise.append({
                    "source_text_idx": global_ti,
                    "text": text,
                    "mask_ratio": float(ratio),
                    "token_position": int(token_pos),
                    "target_negative": target_negative,
                    "reconstruction": reconstruction,
                    "target_label": int(target_negative >= 0.5),
                })

    pooled_acts = {
        li: torch.tensor(np.stack(v), dtype=torch.float32) if len(v) > 0 else torch.empty((0, 0), dtype=torch.float32)
        for li, v in pooled_rows.items()
    }
    tokenwise_acts = {
        li: torch.tensor(np.stack(v), dtype=torch.float32) if len(v) > 0 else torch.empty((0, 0), dtype=torch.float32)
        for li, v in tokenwise_rows.items()
    }
    return pooled_acts, tokenwise_acts, {"pooled_rows": meta_pooled, "tokenwise_rows": meta_tokenwise}


def save_variant_chunk(
    variant_dir: Path,
    chunk_idx: int,
    acts: dict[int, torch.Tensor],
    rows: list[dict],
    start_idx: int,
    end_idx: int,
) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    acts_path = variant_dir / f"part{chunk_idx:03d}.pt"
    meta_path = variant_dir / f"part{chunk_idx:03d}_meta.json"
    torch.save(acts, acts_path)
    _write_json(meta_path, {
        "chunk_idx": chunk_idx,
        "text_start_idx": start_idx,
        "text_end_idx_exclusive": end_idx,
        "n_rows": len(rows),
        "rows": rows,
    })


def collect_all_variants(real_texts: list[str], real_labels: list[int], model, tokenizer) -> None:
    for variant in ["real_full_pooled", "masked_pooled", "masked_tokenwise"]:
        (ACT_ROOT / variant).mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "crash_test": CRASH_TEST,
        "n_real_train_texts": len(real_texts),
        "n_words": N_WORDS,
        "prompt_words": PROMPT_WORDS,
        "noise_levels": NOISE_LEVELS,
        "recon_params": RECON_PARAMS,
        "masked_variant_mode": "version_B_first_denoising_forward_cache",
        "variants": {k: {"files": []} for k in ["real_full_pooled", "masked_pooled", "masked_tokenwise"]},
    }

    for chunk_idx, (start_idx, end_idx, idxs) in enumerate(iter_chunks(list(range(len(real_texts))), CHUNK_SIZE)):
        chunk_texts = [real_texts[i] for i in idxs]
        chunk_labels = [real_labels[i] for i in idxs]

        real_acts, real_meta = extract_real_full_pooled_chunk(
            chunk_texts, chunk_labels, start_idx, model, tokenizer, DEVICE
        )
        save_variant_chunk(ACT_ROOT / "real_full_pooled", chunk_idx, real_acts, real_meta["rows"], start_idx, end_idx)
        manifest["variants"]["real_full_pooled"]["files"].append({
            "chunk_idx": chunk_idx,
            "acts_file": f"part{chunk_idx:03d}.pt",
            "meta_file": f"part{chunk_idx:03d}_meta.json",
            "n_rows": len(real_meta["rows"]),
        })

        masked_pooled_acts, masked_tokenwise_acts, masked_meta = extract_masked_variants_chunk(
            chunk_texts, start_idx, model, tokenizer, DEVICE
        )
        save_variant_chunk(ACT_ROOT / "masked_pooled", chunk_idx, masked_pooled_acts, masked_meta["pooled_rows"], start_idx, end_idx)
        save_variant_chunk(ACT_ROOT / "masked_tokenwise", chunk_idx, masked_tokenwise_acts, masked_meta["tokenwise_rows"], start_idx, end_idx)
        manifest["variants"]["masked_pooled"]["files"].append({
            "chunk_idx": chunk_idx,
            "acts_file": f"part{chunk_idx:03d}.pt",
            "meta_file": f"part{chunk_idx:03d}_meta.json",
            "n_rows": len(masked_meta["pooled_rows"]),
        })
        manifest["variants"]["masked_tokenwise"]["files"].append({
            "chunk_idx": chunk_idx,
            "acts_file": f"part{chunk_idx:03d}.pt",
            "meta_file": f"part{chunk_idx:03d}_meta.json",
            "n_rows": len(masked_meta["tokenwise_rows"]),
        })

        del real_acts, masked_pooled_acts, masked_tokenwise_acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_json(ACT_ROOT / "manifest.json", manifest)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD COLLECTED VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def load_variant_dataset(variant: str) -> tuple[dict[int, torch.Tensor], list[dict]]:
    variant_dir = ACT_ROOT / variant
    part_files = sorted(variant_dir.glob("part*.pt"))
    if not part_files:
        raise RuntimeError(f"No activation parts found for {variant} in {variant_dir}")

    act_parts = []
    rows_all = []
    for pt in part_files:
        meta = _read_json(pt.with_name(pt.stem + "_meta.json"))
        act = torch.load(pt, map_location="cpu", weights_only=False)
        act = {int(k): v.float().cpu() for k, v in act.items()}
        act_parts.append(act)
        rows_all.extend(meta["rows"])

    layer_ids = sorted(act_parts[0].keys())
    acts = {li: torch.cat([part[li] for part in act_parts], dim=0) for li in layer_ids}
    return acts, rows_all


# ══════════════════════════════════════════════════════════════════════════════
# PROBES + MEAN-DIFF
# ══════════════════════════════════════════════════════════════════════════════

def grouped_split_by_text(rows: list[dict], val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    by_group = {}
    for i, row in enumerate(rows):
        gid = f"text::{row['source_text_idx']}"
        by_group.setdefault(gid, []).append(i)
    gids = list(by_group.keys())
    rng.shuffle(gids)
    n_val = max(1, int(round(len(gids) * val_frac)))
    if len(gids) - n_val < 1:
        n_val = len(gids) - 1
    val_gids = set(gids[:n_val])
    train_idx, val_idx = [], []
    for gid, idxs in by_group.items():
        if gid in val_gids:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def build_targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(r["target_negative"]) for r in rows], dtype=np.float32)


def train_probe_single_layer(
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> tuple[dict, dict]:
    _seed_everything(seed)
    device = "cpu"
    X_train = X[train_idx].to(device)
    y_train = y[train_idx].to(device)
    X_val = X[val_idx].to(device)
    y_val = y[val_idx].to(device)

    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=min(PROBE_BATCH_SIZE, len(ds)), shuffle=True)
    model = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best = {"val_mse": float("inf"), "state_dict": None, "epoch": -1}
    for epoch in range(PROBE_EPOCHS):
        model.train()
        for xb, yb in dl:
            pred = torch.sigmoid(model(xb)).squeeze(-1)
            loss = criterion(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = torch.sigmoid(model(X_val)).squeeze(-1)
            val_mse = float(nn.functional.mse_loss(pred_val, y_val).item())

        if val_mse < best["val_mse"]:
            best["val_mse"] = val_mse
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch

    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_val = torch.sigmoid(model(X_val)).squeeze(-1).cpu().numpy().astype(np.float32)
        y_val_np = y_val.cpu().numpy().astype(np.float32)

    weight = model.weight.detach().cpu().view(-1).float()
    bias = model.bias.detach().cpu().view(-1).float()
    metrics = {
        "best_epoch": int(best["epoch"]),
        "val_mse": float(best["val_mse"]),
        "val_mae": float(np.mean(np.abs(pred_val - y_val_np))),
        "val_r2": float(_r2_score(y_val_np, pred_val)),
        "val_pearson": float(_pearson(y_val_np, pred_val)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }
    artifact = {
        "weight": weight,
        "bias": bias,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    return metrics, artifact


def compute_mean_diff_vectors(acts: dict[int, torch.Tensor], rows: list[dict], variant: str) -> dict[int, torch.Tensor]:
    if variant == "real_full_pooled":
        neg_idx = [i for i, r in enumerate(rows) if int(r["label"]) == 0]
        pos_idx = [i for i, r in enumerate(rows) if int(r["label"]) == 1]
    else:
        neg_idx = [i for i, r in enumerate(rows) if float(r["target_negative"]) >= 0.5]
        pos_idx = [i for i, r in enumerate(rows) if float(r["target_negative"]) < 0.5]
        if len(neg_idx) == 0 or len(pos_idx) == 0:
            scores = np.asarray([float(r["target_negative"]) for r in rows], dtype=np.float32)
            order = np.argsort(scores)
            half = max(1, len(order) // 2)
            pos_idx = order[:half].tolist()
            neg_idx = order[-half:].tolist()

    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise RuntimeError(f"Variant {variant} has empty neg or pos split for mean-diff.")

    vectors = {}
    for li, X in acts.items():
        neg_mean = X[neg_idx].mean(dim=0)
        pos_mean = X[pos_idx].mean(dim=0)
        v = pos_mean - neg_mean
        n = torch.linalg.norm(v).item()
        if n > 1e-12:
            v = v / n
        vectors[li] = v.float().cpu()
    return vectors


def train_variant_and_alignment(variant: str) -> None:
    acts, rows = load_variant_dataset(variant)
    y_np = build_targets(rows)
    y = torch.tensor(y_np, dtype=torch.float32)
    train_idx, val_idx = grouped_split_by_text(rows, PROBE_VAL_FRAC, SEED)

    variant_probe_dir = PROBE_ROOT / variant
    variant_probe_dir.mkdir(parents=True, exist_ok=True)
    mean_diff = compute_mean_diff_vectors(acts, rows, variant)
    torch.save({k: v.cpu() for k, v in mean_diff.items()}, variant_probe_dir / "mean_diff_vectors.pt")

    alignment_rows = []
    for li in sorted(acts.keys()):
        X = acts[li]
        metrics, artifact = train_probe_single_layer(X, y, train_idx, val_idx, seed=SEED + li)
        layer_dir = variant_probe_dir / f"layer_{li:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, layer_dir / "probe.pt")
        torch.save({"weight": artifact["weight"], "bias": artifact["bias"]}, layer_dir / "probe_weight_bias.pt")
        _write_json(layer_dir / "metrics.json", metrics)

        alignment_rows.append({
            "layer": li,
            "cosine_probe_vs_mean_diff": _cosine(artifact["weight"], mean_diff[li]),
            "probe_norm": float(torch.linalg.norm(artifact["weight"]).item()),
            "mean_diff_norm": float(torch.linalg.norm(mean_diff[li]).item()),
            **metrics,
        })

    _write_json(ALIGN_ROOT / f"{variant}_alignment.json", {"variant": variant, "rows": alignment_rows})
    _write_json(PROBE_ROOT / f"{variant}_dataset_summary.json", {
        "variant": variant,
        "n_rows": len(rows),
        "n_layers": len(acts),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "target_definition": "original label for real_full_pooled; reconstruction negativity for masked variants",
    })


# ══════════════════════════════════════════════════════════════════════════════
# EVAL
# ══════════════════════════════════════════════════════════════════════════════

def generate_with_steering(model, tokenizer, prompts: list[str], steerer, desc: str) -> list[str]:
    answers = []
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
        generated_ids = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers.extend(decoded)
    return answers


def evaluate_texts(prompts: list[str], answers: list[str]) -> dict:
    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]
    sent_ans = compute_sentiment_metrics(answers, device=DEVICE)
    sent_comb = compute_sentiment_metrics(combined, device=DEVICE)
    ppl_ans = compute_perplexity(answers, device="cpu")
    ppl_comb = compute_perplexity(combined, device="cpu")
    return {
        "sent_answer_mean": sent_ans["mean_negative"],
        "sent_answer_fraction": sent_ans["negative_fraction"],
        "sent_combined_mean": sent_comb["mean_negative"],
        "ppl_answer_mean": ppl_ans["mean_ppl"],
        "ppl_combined_mean": ppl_comb["mean_ppl"],
        "mean_answer_words": float(sum(len(a.split()) for a in answers) / max(1, len(answers))),
    }


def load_probe_params_for_variant(variant: str, layer_ids: list[int]) -> dict[int, ProbeParams]:
    probe_by_layer = {}
    for li in layer_ids:
        p = PROBE_ROOT / variant / f"layer_{li:02d}" / "probe_weight_bias.pt"
        obj = torch.load(p, map_location="cpu", weights_only=False)
        w = obj["weight"].float().cpu().view(-1)
        b = obj["bias"].float().cpu().view(-1)
        probe_by_layer[li] = ProbeParams(weight=w, bias=b, norm_sq=float((w * w).sum().item()))
    return probe_by_layer


def run_eval_for_variant(variant: str, prompts: list[str], model, tokenizer) -> None:
    variant_eval_dir = EVAL_ROOT / variant
    variant_eval_dir.mkdir(parents=True, exist_ok=True)

    mean_diff = torch.load(PROBE_ROOT / variant / "mean_diff_vectors.pt", map_location="cpu", weights_only=False)
    mean_vectors = {int(k): v for k, v in mean_diff.items() if int(k) in STEER_LAYERS}
    mean_steerer = PrecomputedLayerSteering(vectors=mean_vectors, layer_ids=STEER_LAYERS, alpha=MEAN_ALPHA)
    mean_answers = generate_with_steering(model, tokenizer, prompts, mean_steerer, f"mean_diff[{variant}]")
    _write_json(variant_eval_dir / "mean_diff_metrics.json", evaluate_texts(prompts, mean_answers))

    probe_by_layer = load_probe_params_for_variant(variant, STEER_LAYERS)
    liseco_runs = {}
    for alpha_min, alpha_max in LISECO_INTERVALS:
        steerer = LiSeCoProbeSteering(
            probe_by_layer=probe_by_layer,
            layer_ids=STEER_LAYERS,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mask_id=MASK_ID,
        )
        answers = generate_with_steering(
            model, tokenizer, prompts, steerer,
            f"liseco[{variant} {alpha_min:.2f}-{alpha_max:.2f}]"
        )
        liseco_runs[f"amin{alpha_min:.2f}_amax{alpha_max:.2f}"] = evaluate_texts(prompts, answers)

    _write_json(variant_eval_dir / "liseco_metrics.json", liseco_runs)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _seed_everything(SEED)
    for d in [OUT_ROOT, ACT_ROOT, PROBE_ROOT, ALIGN_ROOT, EVAL_ROOT]:
        d.mkdir(parents=True, exist_ok=True)

    real_texts, real_labels = load_real_train_texts()
    prompts, raw_eval_texts, eval_labels = load_eval_prompts()

    _write_json(OUT_ROOT / "run_config.json", {
        "crash_test": CRASH_TEST,
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "n_real_train_texts": len(real_texts),
        "n_eval_prompts": len(prompts),
        "noise_levels": NOISE_LEVELS,
        "recon_params": RECON_PARAMS,
        "steer_layers": STEER_LAYERS,
        "mean_alpha": MEAN_ALPHA,
        "liseco_intervals": LISECO_INTERVALS,
    })
    _write_json(OUT_ROOT / "eval_prompts.json", {
        "prompts": prompts,
        "raw_texts": raw_eval_texts,
        "labels": eval_labels,
    })

    print("[model] loading model/tokenizer ...")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    print("[step] collecting activations ...")
    collect_all_variants(real_texts, real_labels, model, tokenizer)

    for variant in ["real_full_pooled", "masked_pooled", "masked_tokenwise"]:
        print(f"[step] training + alignment for {variant} ...")
        train_variant_and_alignment(variant)

    for variant in ["real_full_pooled", "masked_pooled", "masked_tokenwise"]:
        print(f"[step] eval for {variant} ...")
        run_eval_for_variant(variant, prompts, model, tokenizer)

    summary = {}
    for variant in ["real_full_pooled", "masked_pooled", "masked_tokenwise"]:
        summary[variant] = {
            "alignment": _read_json(ALIGN_ROOT / f"{variant}_alignment.json"),
            "mean_diff_eval": _read_json(EVAL_ROOT / variant / "mean_diff_metrics.json"),
            "liseco_eval": _read_json(EVAL_ROOT / variant / "liseco_metrics.json"),
        }
    _write_json(OUT_ROOT / "summary.json", summary)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[done] outputs saved under {OUT_ROOT}")


if __name__ == "__main__":
    main()