from __future__ import annotations

"""
collect_instruct_real_sentiment_activations.py

Collect real-positive and real-negative hidden activations for LLaDA-Instruct,
for downstream DiffMean steering experiments.

Outputs (under --out-dir):
- texts/real_negative.json
- texts/real_positive.json
- activations/real_negative.pt
- activations/real_positive.pt
- activations/chunks/<class>/chunk_XXXX.pt (intermediate)
- summary.json
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
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.sentiment_metrics import compute_sentiment_metrics
from llada.model import load_model, load_tokenizer


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
SENTIMENT_DEVICE = os.getenv("SENTIMENT_DEVICE", "cpu")

TARGET_PER_CLASS = 5000
TEXT_MAX_CHARS = 4000
TOKEN_MAX_LENGTH = 256
CHUNK_SIZE = 1000
SAVE_DTYPE = "float16"

SMOKE_TEST = False
SMOKE_TARGET_PER_CLASS = 32



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
        "Cannot locate transformer layers on model. Tried "
        "model.model.transformer['blocks'], model.model.layers, model.layers."
    )



def _cast_tensor_dtype(x: torch.Tensor, save_dtype: str) -> torch.Tensor:
    if save_dtype == "float16":
        return x.to(torch.float16)
    if save_dtype == "float32":
        return x.to(torch.float32)
    raise ValueError(f"Unsupported save dtype: {save_dtype}")



def _extract_means_for_texts(
    texts: list[str],
    model,
    tokenizer,
    *,
    token_max_length: int,
    save_dtype: str,
) -> dict[int, torch.Tensor]:
    layers = _get_transformer_layers(model)
    n_layers = len(layers)

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()

        return hook

    hooks = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]

    rows_by_layer: dict[int, list[torch.Tensor]] = {li: [] for li in range(n_layers)}

    model.eval()
    try:
        with torch.inference_mode():
            for text in tqdm(texts, desc="[act] chunk", leave=False):
                enc = tokenizer(
                    [text],
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=token_max_length,
                ).to(next(model.parameters()).device)

                input_ids = enc["input_ids"]
                attn_mask = enc.get("attention_mask")
                _ = model(input_ids, attention_mask=attn_mask)

                seq_len = input_ids.shape[1]
                attn_float = (
                    attn_mask.float().cpu()
                    if attn_mask is not None
                    else torch.ones(1, seq_len, dtype=torch.float32)
                )

                for li in range(n_layers):
                    hs = captured[li]
                    mask = attn_float.unsqueeze(-1)
                    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    rows_by_layer[li].append(_cast_tensor_dtype(pooled.squeeze(0), save_dtype))

                captured.clear()

    finally:
        for h in hooks:
            h.remove()
        captured.clear()

    out: dict[int, torch.Tensor] = {}
    for li in range(n_layers):
        out[li] = torch.stack(rows_by_layer[li], dim=0)
    return out



def _save_chunk(
    class_name: str,
    chunk_idx: int,
    texts: list[str],
    model,
    tokenizer,
    out_dir: Path,
    token_max_length: int,
    save_dtype: str,
) -> Path:
    act = _extract_means_for_texts(
        texts,
        model,
        tokenizer,
        token_max_length=token_max_length,
        save_dtype=save_dtype,
    )
    chunk_path = out_dir / "activations" / "chunks" / class_name / f"chunk_{chunk_idx:04d}.pt"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(act, chunk_path)
    return chunk_path



def _merge_chunks_to_final(
    chunk_paths: list[Path],
    final_path: Path,
    save_dtype: str,
) -> dict[str, Any]:
    if not chunk_paths:
        raise RuntimeError(f"No chunk files to merge for {final_path}")

    first = torch.load(chunk_paths[0], map_location="cpu", weights_only=False)
    layer_ids = sorted(int(k) for k in first.keys())

    merged: dict[int, torch.Tensor] = {}
    for li in layer_ids:
        parts = []
        for p in chunk_paths:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            parts.append(obj[li])
        cat = torch.cat(parts, dim=0)
        merged[li] = _cast_tensor_dtype(cat, save_dtype)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, final_path)

    n_examples = int(merged[layer_ids[0]].shape[0])
    hidden_dim = int(merged[layer_ids[0]].shape[1])
    n_layers = len(layer_ids)
    return {
        "n_examples": n_examples,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dtype": save_dtype,
        "path": str(final_path),
    }



def _select_texts_by_label(
    target_per_class: int,
    text_max_chars: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

    negative_texts: list[str] = []
    positive_texts: list[str] = []
    scanned = 0

    for ex in ds:
        scanned += 1
        label = int(ex["label"])
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        text = (text or "").strip()
        if len(text) < 5:
            continue
        if text_max_chars > 0:
            text = text[:text_max_chars]

        if label == 0 and len(negative_texts) < target_per_class:
            negative_texts.append(text)
        elif label == 1 and len(positive_texts) < target_per_class:
            positive_texts.append(text)

        if len(negative_texts) >= target_per_class and len(positive_texts) >= target_per_class:
            break

    meta = {
        "dataset": "fancyzhx/amazon_polarity",
        "split": "train",
        "streaming": True,
        "scanned_examples": scanned,
        "target_per_class": target_per_class,
        "selected_negative": len(negative_texts),
        "selected_positive": len(positive_texts),
    }
    return negative_texts, positive_texts, meta



def _mean_negative_score(texts: list[str], device: str) -> float | None:
    if not texts:
        return None
    scores = compute_sentiment_metrics(texts, device=device)
    return float(scores["mean_negative"])



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/speed_adapt/instruct_real_sentiment_activations"),
    )
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--token-max-length", type=int, default=TOKEN_MAX_LENGTH)
    parser.add_argument("--text-max-chars", type=int, default=TEXT_MAX_CHARS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--save-dtype", type=str, default=SAVE_DTYPE, choices=["float16", "float32"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-target-per-class", type=int, default=SMOKE_TARGET_PER_CLASS)
    parser.set_defaults(smoke_test=SMOKE_TEST)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    _seed_everything(SEED)

    out_dir = args.out_dir / ("smoke_test" if args.smoke_test else "full_run")
    (out_dir / "texts").mkdir(parents=True, exist_ok=True)
    (out_dir / "activations").mkdir(parents=True, exist_ok=True)

    target_per_class = args.smoke_target_per_class if args.smoke_test else args.target_per_class

    print(f"[data] selecting texts per class={target_per_class} (label-based)")
    negative_texts, positive_texts, sel_meta = _select_texts_by_label(
        target_per_class=target_per_class,
        text_max_chars=args.text_max_chars,
    )

    _write_json(out_dir / "texts" / "real_negative.json", {"texts": negative_texts, "label": 0})
    _write_json(out_dir / "texts" / "real_positive.json", {"texts": positive_texts, "label": 1})

    print(f"[model] loading {args.model_name} on {DEVICE}")
    model = load_model(model_name=args.model_name, device=DEVICE)
    tokenizer = load_tokenizer(model_name=args.model_name)

    chunk_paths_neg: list[Path] = []
    chunk_paths_pos: list[Path] = []

    print("[act] extracting real_negative chunks")
    for i in range(0, len(negative_texts), args.chunk_size):
        chunk = negative_texts[i : i + args.chunk_size]
        chunk_idx = i // args.chunk_size
        p = _save_chunk(
            class_name="real_negative",
            chunk_idx=chunk_idx,
            texts=chunk,
            model=model,
            tokenizer=tokenizer,
            out_dir=out_dir,
            token_max_length=args.token_max_length,
            save_dtype=args.save_dtype,
        )
        chunk_paths_neg.append(p)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[act] extracting real_positive chunks")
    for i in range(0, len(positive_texts), args.chunk_size):
        chunk = positive_texts[i : i + args.chunk_size]
        chunk_idx = i // args.chunk_size
        p = _save_chunk(
            class_name="real_positive",
            chunk_idx=chunk_idx,
            texts=chunk,
            model=model,
            tokenizer=tokenizer,
            out_dir=out_dir,
            token_max_length=args.token_max_length,
            save_dtype=args.save_dtype,
        )
        chunk_paths_pos.append(p)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[act] merging chunks -> final class files")
    neg_info = _merge_chunks_to_final(
        chunk_paths=chunk_paths_neg,
        final_path=out_dir / "activations" / "real_negative.pt",
        save_dtype=args.save_dtype,
    )
    pos_info = _merge_chunks_to_final(
        chunk_paths=chunk_paths_pos,
        final_path=out_dir / "activations" / "real_positive.pt",
        save_dtype=args.save_dtype,
    )

    # Sentiment audit (scores are P(NEGATIVE), so negative class should be higher).
    audit_n = min(200, len(negative_texts), len(positive_texts))
    neg_sent = _mean_negative_score(negative_texts[:audit_n], device=SENTIMENT_DEVICE)
    pos_sent = _mean_negative_score(positive_texts[:audit_n], device=SENTIMENT_DEVICE)

    summary = {
        "model_name": args.model_name,
        "device": DEVICE,
        "sentiment_device": SENTIMENT_DEVICE,
        "smoke_test": bool(args.smoke_test),
        "target_per_class": target_per_class,
        "token_max_length": args.token_max_length,
        "text_max_chars": args.text_max_chars,
        "chunk_size": args.chunk_size,
        "save_dtype": args.save_dtype,
        "selection": sel_meta,
        "negative_activation": neg_info,
        "positive_activation": pos_info,
        "sentiment_audit": {
            "n_samples": audit_n,
            "mean_p_negative_real_negative": neg_sent,
            "mean_p_negative_real_positive": pos_sent,
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
