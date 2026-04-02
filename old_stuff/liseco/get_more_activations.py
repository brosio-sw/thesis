"""
get_activations_tail_all.py – Extract per-layer mean activations from the tail
of Amazon Polarity, without generation or label splits.

Goal
----
Take the last N examples from the Amazon Polarity train split, compute:
- text
- sentiment score
- per-layer mean hidden activations from GSAI-ML/LLaDA-8B-Base

No generation is performed.

Outputs
-------
Saved under:
    data/steering_test_tail_all/

Files:
    texts/
      tail_all.json

    scores/
      tail_all_sentiment.json

    activations/
      tail_all.pt

Each .pt file has structure:
    { layer_idx (int): torch.Tensor of shape [N_examples, hidden_dim] }

Scale
-----
- If SMOKE_TEST = True: use 5 examples
- If SMOKE_TEST = False: use N_EXAMPLES examples

Usage
-----
    python get_activations_tail_all.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from eval.sentiment_metrics import compute_sentiment_metrics


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"

SMOKE_TEST = False
N_EXAMPLES = 20000
N_SMOKE = 5
N_WORDS = 50

OUT_DIR = Path("data/steering_test_tail_all")


# ── Data loading ──────────────────────────────────────────────────────────────

def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()


def load_amazon_tail_examples(n: int, n_words: int) -> list[str]:
    """
    Stream the Amazon Polarity train split and keep the LAST `n` examples,
    regardless of label.
    """
    print(f"[data] Loading LAST {n} examples from Amazon Polarity ({n_words} words each) …")
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

    tail = deque(maxlen=n)

    for ex in tqdm(ds, desc="[data] scanning dataset tail"):
        text = ex.get("content") or ex.get("text") or (
            f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        )
        truncated = _truncate(text, n_words)
        if len(truncated.split()) >= 5:
            tail.append(truncated)

    tail_list = list(tail)
    print(f"[data] Loaded {len(tail_list)} tail examples.")
    return tail_list


# ── Activation extraction ─────────────────────────────────────────────────────

def _get_transformer_layers(model) -> list:
    if (
        hasattr(model, "model")
        and hasattr(model.model, "transformer")
        and "blocks" in model.model.transformer
    ):
        return list(model.model.transformer["blocks"])

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)

    if hasattr(model, "layers"):
        return list(model.layers)

    raise RuntimeError(
        "Cannot locate transformer layers on model. "
        "Tried model.model.transformer['blocks'], model.model.layers, model.layers."
    )


@torch.no_grad()
def extract_mean_layer_activations(
    texts: list[str],
    model,
    tokenizer,
    device: str,
) -> dict[int, torch.Tensor]:
    """
    Forward-pass each text through the model and compute mean-pooled hidden
    activations for every transformer layer.

    Returns:
        {layer_idx: Tensor[N_examples, hidden_dim]}
    """
    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    n_texts = len(texts)

    all_full: list[list[np.ndarray]] = []
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()
        return hook

    hooks = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]

    model.eval()
    try:
        for text in tqdm(texts, desc="[act] Extracting activations"):
            enc = tokenizer(
                [text],
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
            ).to(device)

            input_ids = enc["input_ids"]
            attn_mask = enc.get("attention_mask")

            _ = model(input_ids, attention_mask=attn_mask)

            seq_len = input_ids.shape[1]
            attn_float = (
                attn_mask.float().cpu()
                if attn_mask is not None
                else torch.ones(1, seq_len, dtype=torch.float32)
            )

            per_layer_full = []
            for li in range(n_layers):
                hs = captured[li]                     # [1, L, D]
                mask = attn_float.unsqueeze(-1)      # [1, L, 1]
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                per_layer_full.append(pooled.squeeze(0).numpy())

            captured.clear()
            all_full.append(per_layer_full)

    finally:
        for h in hooks:
            h.remove()
        captured.clear()

    return {
        li: torch.tensor(
            np.stack([all_full[ti][li] for ti in range(n_texts)]),
            dtype=torch.float32,
        )
        for li in range(n_layers)
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    n_use = N_SMOKE if SMOKE_TEST else N_EXAMPLES

    out_texts = OUT_DIR / "texts"
    out_scores = OUT_DIR / "scores"
    out_act = OUT_DIR / "activations"
    for d in [out_texts, out_scores, out_act]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STEP 1 — Loading tail examples (N={n_use})")
    print(f"{'='*60}")
    texts = load_amazon_tail_examples(n_use, n_words=N_WORDS)

    with open(out_texts / "tail_all.json", "w") as f:
        json.dump(
            {"texts": texts, "n_words": N_WORDS, "tail": True},
            f,
            indent=2,
        )

    print(f"\n{'='*60}")
    print("STEP 2 — Scoring sentiment")
    print(f"{'='*60}")
    sent = compute_sentiment_metrics(texts, device=DEVICE)
    with open(out_scores / "tail_all_sentiment.json", "w") as f:
        json.dump(
            {
                "mean_negative": sent["mean_negative"],
                "negative_fraction": sent["negative_fraction"],
                "max_negative": sent["max_negative"],
                "per_text": sent["scores"].tolist(),
            },
            f,
            indent=2,
        )

    print(f"\n{'='*60}")
    print(f"STEP 3 — Loading model: {MODEL_NAME}")
    print(f"{'='*60}")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    print(f"\n{'='*60}")
    print("STEP 4 — Extracting activations")
    print(f"{'='*60}")
    acts = extract_mean_layer_activations(texts, model, tokenizer, DEVICE)
    torch.save(acts, out_act / "tail_all.pt")

    n_layers = len(acts)
    ex_shape = tuple(acts[0].shape)

    print(f"\n{'='*60}")
    print("TAIL-ALL ACTIVATION EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Smoke test: {SMOKE_TEST}")
    print(f"Examples: {n_use}")
    print(f"Layers extracted: {n_layers}")
    print(f"Shape per layer: {ex_shape}  (N_examples × hidden_dim)")
    print(f"Mean negativity: {sent['mean_negative']:.4f}")
    print(f"Outputs saved under: {OUT_DIR.resolve()}/")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()