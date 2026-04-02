"""
get_masked_activations.py – Extract masked-token activations for LiSeCo training.

Goal
----
Take the last N examples from the Amazon Polarity train split, create corrupted
versions by masking portions of the continuation region at different noise levels,
reconstruct each corrupted sequence with LLaDA using fixed-length in-place denoising,
filter invalid / degenerate reconstructions, and compute mean-pooled hidden
activations strictly over the *originally masked* positions.

Targets
-------
The probe target is the negativity score of the reconstructed text produced by LLaDA
from the noised input, not the original clean sentence score.

Filtering
---------
Filtering is done *per individual noise sample*.
For a given source text, if one noise level produces a bad reconstruction, only that
(text, noise_level) row is discarded; valid reconstructions at other noise levels for
the same source text are kept.

Outputs
-------
Saved under:
    data/steering_masked_activations/

Files:
    texts/
      tail_all.json
    activations/
      tail_all_masked_part000.pt      -> { layer_idx: Tensor[n_rows_chunk, D] }
      tail_all_masked_part000_meta.json
      ...
      manifest.json                   -> global info about all chunks

Usage
-----
    python get_masked_activations.py
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import _add_gumbel_noise, _get_num_transfer_tokens
from eval.sentiment_metrics import compute_sentiment_metrics


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
MASK_ID = 126336  # LLaDA mask token id

SMOKE_TEST = False
N_EXAMPLES = 5000
N_SMOKE = 5
N_WORDS = 50
PROMPT_WORDS = 20

NOISE_LEVELS = [0.15, 0.35, 0.55, 0.75, 0.90]
CHUNK_SIZE = 250

OUT_DIR = Path("data/steering_masked_activations")

RECON_PARAMS = dict(
    temperature=0.0,
    steps=30,
    fill_strategy="low_confidence",  # or "random"
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


# ── Data loading ──────────────────────────────────────────────────────────────

def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()


def load_amazon_tail_examples(n: int, n_words: int) -> list[str]:
    print(f"[data] Loading LAST {n} examples from Amazon Polarity ({n_words} words each) …")
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

    tail = deque(maxlen=n)

    for ex in tqdm(ds, desc="[data] scanning dataset tail", disable=SMOKE_TEST):
        text = ex.get("content") or ex.get("text") or (
            f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        )
        truncated = _truncate(text, n_words)
        if len(truncated.split()) >= (PROMPT_WORDS + 5):
            tail.append(truncated)

    tail_list = list(tail)
    print(f"[data] Loaded {len(tail_list)} tail examples.")
    return tail_list


# ── Filtering ─────────────────────────────────────────────────────────────────

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


# ── Reconstruction ────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_from_corrupted(
    batch_input_ids: torch.Tensor,
    tokenizer,
    model,
) -> list[str]:
    """
    Fixed-length in-place denoising of masked sequences.

    Input:
        batch_input_ids : [B, L] with MASK_ID already inserted inside the sequence

    Output:
        list[str] of length B, where each string is the reconstructed full sequence

    Important:
        This does NOT append new tokens. It only fills masked positions already
        present in the input sequence.
    """
    x = batch_input_ids.clone()
    device = x.device
    attention_mask = torch.ones_like(x, device=device)

    initial_mask_index = (x == MASK_ID)
    if not initial_mask_index.any():
        decoded = tokenizer.batch_decode(x, skip_special_tokens=True)
        return [t.strip() for t in decoded]

    num_transfer = _get_num_transfer_tokens(initial_mask_index, RECON_PARAMS["steps"])

    for step_i in range(RECON_PARAMS["steps"]):
        mask_index = (x == MASK_ID)
        if not mask_index.any():
            break

        logits = model(x, attention_mask=attention_mask).logits
        logits_noisy = _add_gumbel_noise(logits, temperature=RECON_PARAMS["temperature"])
        x0 = logits_noisy.argmax(dim=-1)

        if RECON_PARAMS["fill_strategy"] == "low_confidence":
            p = F.softmax(logits.float(), dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        elif RECON_PARAMS["fill_strategy"] == "random":
            x0_p = torch.rand(x0.shape, device=device)
        else:
            raise ValueError(f"Unknown fill_strategy: {RECON_PARAMS['fill_strategy']!r}")

        x0 = torch.where(mask_index, x0, x)
        conf_for_unmask = torch.where(
            mask_index,
            x0_p,
            torch.full_like(x0_p, -float("inf")),
        )

        transfer_index = torch.zeros_like(x, dtype=torch.bool)
        for b in range(x.shape[0]):
            k = min(int(num_transfer[b, step_i].item()), int(mask_index[b].sum().item()))
            if k > 0:
                _, sel = torch.topk(conf_for_unmask[b], k=k)
                transfer_index[b, sel] = True

        x[transfer_index] = x0[transfer_index]

    decoded = tokenizer.batch_decode(x, skip_special_tokens=True)
    return [t.strip() for t in decoded]


# ── Activation extraction ─────────────────────────────────────────────────────

def _get_transformer_layers(model) -> list:
    if hasattr(model, "model") and hasattr(model.model, "transformer") and "blocks" in model.model.transformer:
        return list(model.model.transformer["blocks"])
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError("Cannot locate transformer layers on model.")


@torch.no_grad()
def extract_masked_activations_chunk(
    texts: list[str],
    global_text_offset: int,
    model,
    tokenizer,
    device: str,
):
    """
    Extract activations for one chunk of texts only.

    Returns:
        acts_chunk: {layer_idx: Tensor[n_rows_chunk, D]}
        metadata_chunk: includes owner_text_idx, mask_ratio, target_negative,
                        reconstruction, source_text
    """
    layers = _get_transformer_layers(model)
    n_layers = len(layers)

    all_full: dict[int, list[np.ndarray]] = {li: [] for li in range(n_layers)}

    owner_text_idx: list[int] = []
    mask_ratio_list: list[float] = []
    target_negative_list: list[float] = []
    reconstruction_list: list[str] = []
    source_text_list: list[str] = []

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()
        return hook

    hooks = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]

    model.eval()
    try:
        for local_ti, text in enumerate(tqdm(texts, desc="[act] Extracting masked activations", leave=False)):
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

            batch_input_ids = torch.stack(batch_input_ids).to(device)  # [n_noise, L]

            # 1) Reconstruct each individual noised sample in-place
            recon_texts = reconstruct_from_corrupted(batch_input_ids, tokenizer, model)

            # 2) Filter per individual noise sample
            valid_indices = [i for i, recon in enumerate(recon_texts) if is_valid_completion(recon)]
            if not valid_indices:
                continue

            valid_recons = [recon_texts[i] for i in valid_indices]

            # 3) Score negativity on the actual reconstruction
            sent = compute_sentiment_metrics(valid_recons, device=device)
            neg_scores = sent["scores"].tolist()

            # 4) Capture activations only for the valid corrupted inputs
            valid_batch_input_ids = batch_input_ids[valid_indices]
            _ = model(valid_batch_input_ids)

            for li in range(n_layers):
                hs_batch = captured[li]  # [n_valid, L, D]

                for out_i, orig_i in enumerate(valid_indices):
                    mask_pos = batch_mask_pos_list[orig_i]
                    hs_masked = hs_batch[out_i, mask_pos]  # [n_mask, D]
                    hs_mean = hs_masked.mean(dim=0)        # [D]
                    all_full[li].append(hs_mean.numpy())

                    if li == 0:
                        owner_text_idx.append(global_ti)
                        mask_ratio_list.append(batch_ratio_list[orig_i])
                        target_negative_list.append(float(neg_scores[out_i]))
                        reconstruction_list.append(valid_recons[out_i])
                        source_text_list.append(text)

            captured.clear()

    finally:
        for h in hooks:
            h.remove()
        captured.clear()

    nonempty_layers = [li for li in range(n_layers) if len(all_full[li]) > 0]

    if not nonempty_layers:
        out_acts = {li: torch.empty((0, 0), dtype=torch.float32) for li in range(n_layers)}
    else:
        out_acts = {
            li: torch.tensor(np.stack(all_full[li]), dtype=torch.float32)
            for li in range(n_layers)
        }

    metadata = {
        "owner_text_idx": owner_text_idx,
        "mask_ratio": mask_ratio_list,
        "target_negative": target_negative_list,
        "reconstruction": reconstruction_list,
        "source_text": source_text_list,
    }

    return out_acts, metadata


def iter_chunks(items: list[str], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        yield start, end, items[start:end]


def save_chunk_outputs(
    out_act: Path,
    chunk_idx: int,
    acts: dict[int, torch.Tensor],
    metadata: dict,
    start_idx: int,
    end_idx: int,
):
    acts_path = out_act / f"tail_all_masked_part{chunk_idx:03d}.pt"
    meta_path = out_act / f"tail_all_masked_part{chunk_idx:03d}_meta.json"

    torch.save(acts, acts_path)

    chunk_meta = {
        "chunk_idx": chunk_idx,
        "text_start_idx": start_idx,
        "text_end_idx_exclusive": end_idx,
        "n_input_texts": end_idx - start_idx,
        "n_rows": len(metadata["owner_text_idx"]),
        "owner_text_idx": metadata["owner_text_idx"],
        "mask_ratio": metadata["mask_ratio"],
        "target_negative": metadata["target_negative"],
        "reconstruction": metadata["reconstruction"],
        "source_text": metadata["source_text"],
    }
    with open(meta_path, "w") as f:
        json.dump(chunk_meta, f, indent=2)

    return acts_path.name, meta_path.name, chunk_meta["n_rows"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    n_use = N_SMOKE if SMOKE_TEST else N_EXAMPLES

    out_texts = OUT_DIR / "texts"
    out_act = OUT_DIR / "activations"

    for d in [out_texts, out_act]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STEP 1 — Loading tail examples (N={n_use}, SMOKE_TEST={SMOKE_TEST})")
    print(f"{'='*60}")
    texts = load_amazon_tail_examples(n_use, n_words=N_WORDS)

    with open(out_texts / "tail_all.json", "w") as f:
        json.dump({"texts": texts, "n_words": N_WORDS, "tail": True}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STEP 2 — Loading model: {MODEL_NAME}")
    print(f"{'='*60}")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    print(f"\n{'='*60}")
    print("STEP 3 — Extracting MASKED-TOKEN activations (chunked)")
    print(f"{'='*60}")
    print(f"Chunk size: {CHUNK_SIZE} texts")

    manifest = {
        "smoke_test": SMOKE_TEST,
        "n_texts_total": len(texts),
        "n_words": N_WORDS,
        "prompt_words": PROMPT_WORDS,
        "noise_levels": NOISE_LEVELS,
        "chunk_size": CHUNK_SIZE,
        "model_name": MODEL_NAME,
        "reconstruction": {
            "mode": "fixed_length_in_place_denoising",
            "steps": RECON_PARAMS["steps"],
            "temperature": RECON_PARAMS["temperature"],
            "fill_strategy": RECON_PARAMS["fill_strategy"],
        },
        "target_definition": "negativity score of LLaDA reconstruction from corrupted input",
        "filtering": {
            "bad_patterns": BAD_PATTERNS,
            "repetition_loop_filter": True,
            "filter_unit": "individual_noise_sample",
        },
        "files": [],
    }

    total_rows = 0
    n_layers = None
    hidden_dim = None

    chunk_iter = list(iter_chunks(texts, CHUNK_SIZE))
    for chunk_idx, (start_idx, end_idx, text_chunk) in enumerate(
        tqdm(chunk_iter, desc="[chunk] Processing chunks")
    ):
        print(f"\n[chunk] {chunk_idx:03d} | texts {start_idx}:{end_idx} ({end_idx - start_idx} texts)")

        acts, metadata = extract_masked_activations_chunk(
            texts=text_chunk,
            global_text_offset=start_idx,
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
        )

        if n_layers is None:
            first_nonempty = None
            for li, tensor in acts.items():
                if tensor.numel() > 0:
                    first_nonempty = li
                    break
            if first_nonempty is not None:
                n_layers = len(acts)
                hidden_dim = acts[first_nonempty].shape[1]

        acts_file, meta_file, n_rows = save_chunk_outputs(
            out_act=out_act,
            chunk_idx=chunk_idx,
            acts=acts,
            metadata=metadata,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        manifest["files"].append({
            "chunk_idx": chunk_idx,
            "acts_file": acts_file,
            "meta_file": meta_file,
            "text_start_idx": start_idx,
            "text_end_idx_exclusive": end_idx,
            "n_rows": n_rows,
        })

        total_rows += n_rows

        print(f"[chunk] saved {acts_file} + {meta_file} | rows={n_rows}")

        del acts
        del metadata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["total_rows"] = total_rows
    manifest["n_layers"] = n_layers
    manifest["hidden_dim"] = hidden_dim

    with open(out_act / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print("MASKED-TOKEN ACTIVATION EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Smoke test: {SMOKE_TEST}")
    print(f"Sentences processed: {n_use}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total valid rows collected: {total_rows}")
    print(f"Layers extracted: {n_layers}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Chunks saved: {len(manifest['files'])}")
    print(f"Outputs saved under: {OUT_DIR.resolve()}/")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()