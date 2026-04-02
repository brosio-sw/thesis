from __future__ import annotations

"""
plot_alignment_and_probe_accuracy.py

Utilities for the LLaDA diffusion steering setup used in liseco_vs_diffmean.py.

This script does two things:

1) Plot reversed cosine alignment by layer
   - reads the stored alignment JSON files (or summary.json)
   - plots -cosine(probe, mean_diff) so that stronger expected anti-alignment
     appears as larger positive values.

2) Reproduce a sentiment-only probe-accuracy plot on held-out Amazon reviews
   - loads the already-trained probes for the three activation variants:
       * real_full_pooled
       * masked_pooled
       * masked_tokenwise
   - samples held-out Amazon Polarity examples from train[50000:50200]
   - runs LLaDA forward passes
   - evaluates the stored probes layer-by-layer against the gold binary labels
   - makes a plot similar in spirit to the one you shared.

Important note about sign conventions
-------------------------------------
The probes in liseco_vs_diffmean.py are trained to predict NEGATIVITY
(target_negative = 1 - label for real_full_pooled). Meanwhile mean-diff is
constructed as pos_mean - neg_mean. Therefore the expected cosine between the
probe weight and mean-diff is NEGATIVE. The alignment plot below flips the sign:

    reversed_alignment = -cosine_probe_vs_mean_diff

so "better expected alignment" is shown as a larger positive number.

Important note about held-out probe evaluation
----------------------------------------------
For held-out evaluation, this script compares probe predictions against the
ORIGINAL Amazon binary label:
    label == 0  -> negative review
    label == 1  -> positive review

Since the stored probes predict NEGATIVITY, we convert with:
    predicted_positive = (pred_negative < 0.5)

For masked variants, the training code created multiple rows per text. Here we
aggregate back to ONE prediction per held-out text so the three methods are
comparable at the sentence level:
    * masked_pooled: average probabilities across mask ratios
    * masked_tokenwise: average probabilities across masked tokens and ratios

The masking / activation extraction matches the logic used in
liseco_vs_diffmean.py as closely as possible. See that file for the original
pipeline details.  # noqa: E501
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# Make local repo imports work in the same way as liseco_vs_diffmean.py
_THIS_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(_THIS_DIR))

from llada.model import load_model, load_tokenizer
from llada.generate import _add_gumbel_noise


# -----------------------------------------------------------------------------
# Defaults copied from liseco_vs_diffmean.py where relevant
# -----------------------------------------------------------------------------

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
MASK_ID = 126336

OUT_ROOT_DEFAULT = Path("data/alignment_variants_v4/full_run")
SUMMARY_DEFAULT = Path("summary.json")

VARIANTS = [
    "real_full_pooled",
    "masked_pooled",
    "masked_tokenwise",
]

PROMPT_WORDS = 20
N_WORDS = 50
NOISE_LEVELS = [0.25, 0.5, 0.75, 0.9]
RECON_TEMPERATURE = 0.0


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)



def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)



def truncate_words(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()



def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))



def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))



def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    radius = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, centre - radius)
    hi = min(1.0, centre + radius)
    return lo, hi


# -----------------------------------------------------------------------------
# Model / probe helpers
# -----------------------------------------------------------------------------


@dataclass
class Probe:
    weight: np.ndarray
    bias: float

    def predict_negative_prob(self, x: np.ndarray) -> float:
        # Probe predicts NEGATIVITY, same convention as training in liseco_vs_diffmean.py.
        logit = float(np.dot(self.weight, x) + self.bias)
        return float(1.0 / (1.0 + math.exp(-logit)))



def get_transformer_layers(model) -> list:
    if hasattr(model, "model") and hasattr(model.model, "transformer") and "blocks" in model.model.transformer:
        return list(model.model.transformer["blocks"])
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError("Cannot locate transformer layers on model.")



def load_probe_bank(out_root: Path) -> Dict[str, Dict[int, Probe]]:
    probe_root = out_root / "probes"
    bank: Dict[str, Dict[int, Probe]] = {}
    for variant in VARIANTS:
        variant_dir = probe_root / variant
        if not variant_dir.exists():
            raise FileNotFoundError(f"Missing probe directory: {variant_dir}")
        probes: Dict[int, Probe] = {}
        for layer_dir in sorted(variant_dir.glob("layer_*")):
            layer = int(layer_dir.name.split("_")[-1])
            obj = torch.load(layer_dir / "probe_weight_bias.pt", map_location="cpu", weights_only=False)
            w = obj["weight"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            b_arr = obj["bias"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            b = float(b_arr[0]) if len(b_arr) else 0.0
            probes[layer] = Probe(weight=w, bias=b)
        if not probes:
            raise FileNotFoundError(f"No layer probes found for variant={variant} under {variant_dir}")
        bank[variant] = probes
    return bank


# -----------------------------------------------------------------------------
# Alignment loading / plotting
# -----------------------------------------------------------------------------


def load_alignment_rows(out_root: Path, summary_path: Optional[Path]) -> Dict[str, List[dict]]:
    alignment_root = out_root / "alignment"
    rows_by_variant: Dict[str, List[dict]] = {}

    have_alignment_files = all((alignment_root / f"{v}_alignment.json").exists() for v in VARIANTS)
    if have_alignment_files:
        for variant in VARIANTS:
            payload = read_json(alignment_root / f"{variant}_alignment.json")
            rows_by_variant[variant] = payload["rows"]
        return rows_by_variant

    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(
            "Could not find alignment JSON files under out_root/alignment and no usable summary.json was provided."
        )

    summary = read_json(summary_path)
    for variant in VARIANTS:
        if variant not in summary:
            raise KeyError(f"Variant {variant} missing from summary file: {summary_path}")
        rows_by_variant[variant] = summary[variant]["alignment"]["rows"]
    return rows_by_variant



def plot_reversed_alignment(rows_by_variant: Dict[str, List[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined figure
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, variant in zip(axes, VARIANTS):
        rows = rows_by_variant[variant]
        layers = [int(r["layer"]) for r in rows]
        rev_cos = [-float(r["cosine_probe_vs_mean_diff"]) for r in rows]
        ax.plot(layers, rev_cos, marker="o")
        ax.set_title(variant)
        ax.set_xlabel("layer")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel("reversed cosine alignment")
    fig.suptitle("Probe vs mean-diff alignment (-cosine)")
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_reversed_cosine_combined.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # One file per method, as requested / allowed.
    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        layers = [int(r["layer"]) for r in rows]
        rev_cos = [-float(r["cosine_probe_vs_mean_diff"]) for r in rows]

        fig = plt.figure(figsize=(6.5, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(layers, rev_cos, marker="o")
        ax.set_title(f"{variant}: reversed cosine alignment")
        ax.set_xlabel("layer")
        ax.set_ylabel("reversed cosine alignment")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"alignment_reversed_cosine_{variant}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Held-out Amazon data
# -----------------------------------------------------------------------------


def load_heldout_amazon_slice(start_idx: int, end_idx: int, n_words: int) -> List[dict]:
    split = f"train[{start_idx}:{end_idx}]"
    ds = load_dataset("fancyzhx/amazon_polarity", split=split)
    rows: List[dict] = []
    for ex in ds:
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        text = truncate_words(text, n_words)
        if len(text.split()) < max(5, PROMPT_WORDS + 5):
            continue
        rows.append({
            "text": text,
            "label_positive": int(ex["label"]),  # 0 = negative, 1 = positive
        })
    return rows


# -----------------------------------------------------------------------------
# Forward-pass helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def forward_with_cache(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layers: list) -> Dict[int, torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()
        return hook

    hooks = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        _ = model(input_ids, attention_mask=attention_mask)
        out = {li: captured[li].clone() for li in range(len(layers))}
    finally:
        for h in hooks:
            h.remove()
        captured.clear()
    return out


@torch.no_grad()
def reconstruct_one_pass_with_cache(
    batch_input_ids: torch.Tensor,
    tokenizer,
    model,
    layers: list,
    mask_id: int,
    temperature: float,
) -> Tuple[List[str], Dict[int, torch.Tensor]]:
    """
    Mirrors liseco_vs_diffmean.py:
      - exactly one forward on masked inputs
      - cache layer outputs from that forward
      - fill all masked positions at once via argmax over noisy logits
    """
    x = batch_input_ids.clone()
    attention_mask = torch.ones_like(x, device=x.device)
    mask_index = (x == mask_id)

    if not mask_index.any():
        decoded = tokenizer.batch_decode(x, skip_special_tokens=True)
        return [t.strip() for t in decoded], {}

    captured: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()
        return hook

    hooks = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        logits = model(x, attention_mask=attention_mask).logits
        cache = {li: captured[li].clone() for li in range(len(layers))}
        logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
        x0 = logits_noisy.argmax(dim=-1)
        x_completed = torch.where(mask_index, x0, x)
        decoded = tokenizer.batch_decode(x_completed, skip_special_tokens=True)
        return [t.strip() for t in decoded], cache
    finally:
        for h in hooks:
            h.remove()
        captured.clear()


# -----------------------------------------------------------------------------
# Probe evaluation on held-out texts
# -----------------------------------------------------------------------------


@dataclass
class EvalResult:
    accuracy: Dict[str, List[float]]
    ci_low: Dict[str, List[float]]
    ci_high: Dict[str, List[float]]
    n_examples: int
    layers: List[int]



def evaluate_probes_on_heldout(
    model,
    tokenizer,
    probe_bank: Dict[str, Dict[int, Probe]],
    examples: List[dict],
    prompt_words: int,
    noise_levels: List[float],
    mask_id: int,
    recon_temperature: float,
    seed: int,
) -> EvalResult:
    layers = get_transformer_layers(model)
    n_layers = len(layers)

    # Store one sentence-level prediction per held-out text and layer.
    preds: Dict[str, List[List[int]]] = {
        variant: [[] for _ in range(n_layers)]
        for variant in VARIANTS
    }
    golds: List[int] = []

    rng = np.random.default_rng(seed)
    model.eval()

    for ex in tqdm(examples, desc="held-out probe eval"):
        text = ex["text"]
        gold_positive = int(ex["label_positive"])
        golds.append(gold_positive)

        # ---------------------------------------------------------------
        # 1) real_full_pooled from a full-text forward
        # ---------------------------------------------------------------
        enc_full = tokenizer([text], add_special_tokens=False, return_tensors="pt")
        input_ids_full = enc_full["input_ids"].to(DEVICE)
        attention_mask_full = enc_full.get("attention_mask")
        if attention_mask_full is None:
            attention_mask_full = torch.ones_like(input_ids_full)
        else:
            attention_mask_full = attention_mask_full.to(DEVICE)

        full_cache = forward_with_cache(model, input_ids_full, attention_mask_full, layers)
        valid_tok_mask = attention_mask_full[0].detach().cpu().numpy().astype(bool)

        for li in range(n_layers):
            hs = full_cache[li][0].numpy()  # [L, D]
            pooled = hs[valid_tok_mask].mean(axis=0)
            neg_prob = probe_bank["real_full_pooled"][li].predict_negative_prob(pooled)
            pred_positive = int(neg_prob < 0.5)
            preds["real_full_pooled"][li].append(pred_positive)

        # ---------------------------------------------------------------
        # 2) masked variants from a single masked forward batch
        #    shared by masked_pooled + masked_tokenwise
        # ---------------------------------------------------------------
        words = text.split()
        prompt_text = " ".join(words[:prompt_words])

        enc_text = tokenizer([text], add_special_tokens=False, return_tensors="pt")
        enc_prompt = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")

        input_ids = enc_text["input_ids"][0]
        prompt_len = int(enc_prompt["input_ids"].shape[1])
        seq_len = int(input_ids.shape[0])
        prompt_len = min(prompt_len, seq_len - 5)
        cont_indices = list(range(prompt_len, seq_len))

        if len(cont_indices) < 2:
            # Fallback: if text is too short, reuse the real_full prediction so this
            # example does not silently disappear.
            for li in range(n_layers):
                preds["masked_pooled"][li].append(preds["real_full_pooled"][li][-1])
                preds["masked_tokenwise"][li].append(preds["real_full_pooled"][li][-1])
            continue

        batch_mask_pos: List[np.ndarray] = []
        corrupted_inputs: List[torch.Tensor] = []

        for ratio in noise_levels:
            n_mask = max(1, int(len(cont_indices) * ratio))
            chosen = np.sort(rng.choice(cont_indices, size=n_mask, replace=False))
            corr = input_ids.clone()
            corr[torch.tensor(chosen, dtype=torch.long)] = mask_id
            corrupted_inputs.append(corr)
            batch_mask_pos.append(chosen)

        batch_input_ids = torch.stack(corrupted_inputs).to(DEVICE)
        _, masked_cache = reconstruct_one_pass_with_cache(
            batch_input_ids=batch_input_ids,
            tokenizer=tokenizer,
            model=model,
            layers=layers,
            mask_id=mask_id,
            temperature=recon_temperature,
        )

        # Sentence-level aggregation:
        #   masked_pooled     -> mean across mask ratios of pooled masked-position probs
        #   masked_tokenwise  -> mean across all masked tokens and ratios
        for li in range(n_layers):
            ratio_probs: List[float] = []
            token_probs: List[float] = []
            hs_batch = masked_cache[li].numpy()  # [n_ratios, L, D]
            for ridx, pos in enumerate(batch_mask_pos):
                hs_masked = hs_batch[ridx, pos, :]  # [n_mask, D]
                pooled = hs_masked.mean(axis=0)
                ratio_probs.append(probe_bank["masked_pooled"][li].predict_negative_prob(pooled))
                for j in range(hs_masked.shape[0]):
                    token_probs.append(probe_bank["masked_tokenwise"][li].predict_negative_prob(hs_masked[j]))

            pooled_neg = float(np.mean(ratio_probs)) if ratio_probs else 0.5
            tokenwise_neg = float(np.mean(token_probs)) if token_probs else 0.5
            preds["masked_pooled"][li].append(int(pooled_neg < 0.5))
            preds["masked_tokenwise"][li].append(int(tokenwise_neg < 0.5))

        del full_cache, masked_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert to accuracy + confidence interval per layer.
    accuracy: Dict[str, List[float]] = {v: [] for v in VARIANTS}
    ci_low: Dict[str, List[float]] = {v: [] for v in VARIANTS}
    ci_high: Dict[str, List[float]] = {v: [] for v in VARIANTS}

    n = len(golds)
    gold_arr = np.asarray(golds, dtype=np.int64)
    for variant in VARIANTS:
        for li in range(n_layers):
            pred_arr = np.asarray(preds[variant][li], dtype=np.int64)
            if len(pred_arr) != n:
                raise RuntimeError(
                    f"Prediction length mismatch for variant={variant}, layer={li}: "
                    f"got {len(pred_arr)} predictions, expected {n}."
                )
            correct = int((pred_arr == gold_arr).sum())
            acc = correct / max(1, n)
            lo, hi = wilson_interval(correct, n)
            accuracy[variant].append(acc)
            ci_low[variant].append(lo)
            ci_high[variant].append(hi)

    return EvalResult(
        accuracy=accuracy,
        ci_low=ci_low,
        ci_high=ci_high,
        n_examples=n,
        layers=list(range(n_layers)),
    )


# -----------------------------------------------------------------------------
# Plot sentiment-only held-out probe accuracy
# -----------------------------------------------------------------------------



def plot_sentiment_accuracy(result: EvalResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "real_full_pooled": "real full pooled",
        "masked_pooled": "masked pooled",
        "masked_tokenwise": "masked tokenwise",
    }

    fig = plt.figure(figsize=(8.6, 4.8))
    ax = fig.add_subplot(111)

    for variant in VARIANTS:
        y = np.asarray(result.accuracy[variant], dtype=np.float64)
        lo = np.asarray(result.ci_low[variant], dtype=np.float64)
        hi = np.asarray(result.ci_high[variant], dtype=np.float64)
        ax.plot(result.layers, y, label=label_map[variant])
        ax.fill_between(result.layers, lo, hi, alpha=0.18)

    ax.set_title("sentiment")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out probe accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "sentiment_probe_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT_DEFAULT), help="Root dir from liseco_vs_diffmean.py")
    parser.add_argument("--summary-json", type=str, default=str(SUMMARY_DEFAULT), help="Optional summary.json fallback")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--start-idx", type=int, default=50_000)
    parser.add_argument("--end-idx", type=int, default=50_200)
    parser.add_argument("--prompt-words", type=int, default=PROMPT_WORDS)
    parser.add_argument("--n-words", type=int, default=N_WORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-heldout-eval", action="store_true", help="Only plot reversed cosine alignment")
    parser.add_argument("--plots-dir", type=str, default="plots_alignment_and_probe_accuracy")
    args = parser.parse_args()

    seed_everything(args.seed)

    out_root = Path(args.out_root)
    summary_path = Path(args.summary_json) if args.summary_json else None
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) reversed cosine alignment plot(s)
    rows_by_variant = load_alignment_rows(out_root=out_root, summary_path=summary_path)
    plot_reversed_alignment(rows_by_variant, out_dir=plots_dir)

    if args.skip_heldout_eval:
        print(f"[done] wrote alignment plots to {plots_dir}")
        return

    # 2) held-out sentiment probe accuracy plot
    examples = load_heldout_amazon_slice(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        n_words=args.n_words,
    )
    if not examples:
        raise RuntimeError("No usable held-out Amazon examples were loaded.")
    print(f"[data] loaded {len(examples)} held-out Amazon examples from train[{args.start_idx}:{args.end_idx}]")

    probe_bank = load_probe_bank(out_root=out_root)

    print("[model] loading model + tokenizer ...")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)

    result = evaluate_probes_on_heldout(
        model=model,
        tokenizer=tokenizer,
        probe_bank=probe_bank,
        examples=examples,
        prompt_words=args.prompt_words,
        noise_levels=NOISE_LEVELS,
        mask_id=MASK_ID,
        recon_temperature=RECON_TEMPERATURE,
        seed=args.seed,
    )

    plot_sentiment_accuracy(result, out_dir=plots_dir)

    payload = {
        "n_examples": result.n_examples,
        "layers": result.layers,
        "accuracy": result.accuracy,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "notes": {
            "probe_target": "stored probes predict negativity",
            "dataset_label": "Amazon label is 0=negative, 1=positive",
            "prediction_rule": "predicted_positive = int(pred_negative < 0.5)",
            "masked_aggregation": {
                "masked_pooled": "average probabilities across noise levels, then threshold",
                "masked_tokenwise": "average probabilities across masked tokens and noise levels, then threshold",
            },
            "alignment_plot": "plots -cosine(probe, mean_diff) so expected anti-alignment appears positive",
        },
    }
    write_json(plots_dir / "sentiment_probe_accuracy_metrics.json", payload)

    print(f"[done] wrote plots and metrics to {plots_dir}")


if __name__ == "__main__":
    main()