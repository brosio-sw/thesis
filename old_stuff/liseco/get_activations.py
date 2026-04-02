"""
get_activations.py – Activation extraction pipeline for mean-steering analysis.

Builds 4 text groups and extracts per-layer mean hidden activations from
GSAI-ML/LLaDA-8B-Base for later mean-steering (representation engineering)
experiments.

Text groups
-----------
1. generated_negative : base model generates continuations from negative
                        review prompts (label=0, first N_PROMPT_WORDS words)
2. generated_positive : base model generates continuations from positive
                        review prompts (label=1, first N_PROMPT_WORDS words)
3. real_negative      : real negative Amazon reviews (label=0, first N_REAL_WORDS words)
4. real_positive      : real positive Amazon reviews (label=1, first N_REAL_WORDS words)

Activations saved per group
---------------------------
generated_* groups (2 versions):
  generated_<polarity>_full.pt    – mean over ALL tokens (prompt + answer)
  generated_<polarity>_answer.pt  – mean over ANSWER tokens only
real_* groups (1 version):
  real_<polarity>.pt              – mean over all tokens

All .pt files have structure:
  { layer_idx (int): torch.Tensor of shape [N_examples, hidden_dim] }

Scores saved per generated group (JSON):
  sentiment (answer-only and combined) + perplexity (answer-only and combined)

Outputs under: data/steering_test/
  texts/         – input/generated texts as .json
  activations/   – per-layer mean activations as .pt
  scores/        – sentiment + perplexity scores as .json

Scale control
-------------
N_SMALL = 5 for a smoke-test run.
Change N_SMALL = 500 (and re-run) for the full-scale experiment.

Usage
-----
  python steering_test.py

To reload activations later:
  act = torch.load("data/steering_test/activations/generated_negative_full.pt")
  layer_0_vecs = act[0]   # Tensor of shape [N, hidden_dim]
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"

N_SMALL = 400           # test size; change to 500 for full-scale run

N_PROMPT_WORDS = 20   # words taken from each review for use as generation prompt
N_REAL_WORDS   = 50   # words taken from each review for real-review groups

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",
)

OUT_DIR = Path("data/steering_test")


# ── Data loading ──────────────────────────────────────────────────────────────

def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()


def load_amazon_prompts(n: int, label: int, n_words: int) -> list[str]:
    """
    Load n truncated prompts from Amazon Polarity (streaming).

    label=0 → negative reviews; label=1 → positive reviews.
    Each prompt is the first n_words words of the review content.
    """
    sentiment = "negative" if label == 0 else "positive"
    print(f"[data] Loading {n} {sentiment} prompts (label={label}, {n_words} words) …")
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

    prompts: list[str] = []
    for ex in ds:
        if ex["label"] != label:
            continue
        text = ex.get("content") or ex.get("text") or (
            f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        )
        truncated = _truncate(text, n_words)
        if len(truncated.split()) >= 5:
            prompts.append(truncated)
        if len(prompts) >= n:
            break

    print(f"[data] Loaded {len(prompts)} {sentiment} prompts.")
    return prompts


# ── Generation ────────────────────────────────────────────────────────────────

def generate_continuations(
    model,
    tokenizer,
    prompts: list[str],
    desc: str = "Generating",
) -> list[str]:
    """
    Generate one continuation per prompt using GEN_PARAMS (base model, no remasking).
    Returns list of decoded answer strings (prompt excluded).
    """
    answers: list[str] = []
    for prompt in tqdm(prompts, desc=desc):
        enc = tokenizer(
            [prompt],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(DEVICE)

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
            show_progress=False,
        )

        generated_ids = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers.extend(decoded)

    return answers


# ── Activation extraction ─────────────────────────────────────────────────────

def _get_transformer_layers(model) -> list:
    """
    Return the list of transformer block modules for a LLaDA model.

    Tries several known attribute paths in order:
      1. model.model.transformer['blocks']  – LLaDA-8B-Base/Instruct (MPT-style)
      2. model.model.layers                 – standard LLaMA hierarchy
      3. model.layers                       – flat hierarchy
    """
    # LLaDA-8B-Base / Instruct: ModuleDict with 'blocks' key
    if (
        hasattr(model, "model")
        and hasattr(model.model, "transformer")
        and "blocks" in model.model.transformer
    ):
        return list(model.model.transformer["blocks"])
    # Standard LLaMA-style
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
    prompt_lens_tokens: list[int] | None = None,
) -> dict:
    """
    Forward-pass each text through the model and compute per-layer mean
    hidden activations (mean-pooled over sequence positions).

    Forward hooks are registered on every transformer block so this works
    regardless of whether the model exposes output_hidden_states.

    Args:
        texts:              List of raw text strings (one per example).
        model:              LLaDA model (eval mode, any device mapping).
        tokenizer:          Matching tokenizer.
        device:             Device for tokenisation / forward pass.
        prompt_lens_tokens: If provided, also compute mean over ANSWER tokens
                            only (positions >= prompt_lens_tokens[i]).
                            Must have the same length as `texts`.

    Returns:
        {
          "full":   {layer_idx: Tensor[N, hidden_dim]},
          "answer": {layer_idx: Tensor[N, hidden_dim]},  # present only when
                                                          # prompt_lens_tokens given
        }
    """
    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    n_texts  = len(texts)
    do_answer = prompt_lens_tokens is not None

    # Per-text, per-layer accumulators  →  [n_texts][n_layers] = 1D array [D]
    all_full:   list[list[np.ndarray]] = []
    all_answer: list[list[np.ndarray]] = ([] if do_answer else None)  # type: ignore[assignment]

    # ── Register forward hooks on every transformer block ────────────────────
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, output):
            # LLaMA blocks return a tuple; first element is the hidden state.
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach().float().cpu()
        return hook

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(_make_hook(i)))

    model.eval()
    try:
        for text_i, text in enumerate(
            tqdm(texts, desc="[act] Extracting activations")
        ):
            enc = tokenizer(
                [text],
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
            ).to(device)

            input_ids = enc["input_ids"]                        # [1, L]
            attn_mask  = enc.get("attention_mask")              # [1, L] or None

            # Forward pass – hooks populate `captured`
            _ = model(input_ids, attention_mask=attn_mask)

            seq_len = input_ids.shape[1]
            attn_float = (
                attn_mask.float().cpu()
                if attn_mask is not None
                else torch.ones(1, seq_len, dtype=torch.float32)
            )  # [1, L]

            per_layer_full   = []
            per_layer_answer = [] if do_answer else None

            for li in range(n_layers):
                hs   = captured[li]                             # [1, L, D]
                mask = attn_float.unsqueeze(-1)                 # [1, L, 1]

                # Full mean (all tokens)
                pooled_full = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                per_layer_full.append(pooled_full.squeeze(0).numpy())  # [D]

                # Answer-only mean (tokens after the prompt)
                if do_answer:
                    p_len    = prompt_lens_tokens[text_i]
                    ans_mask = attn_float.clone()
                    ans_mask[0, :p_len] = 0.0               # zero-out prompt positions
                    ans_mask = ans_mask.unsqueeze(-1)        # [1, L, 1]
                    denom    = ans_mask.sum(dim=1).clamp(min=1)
                    pooled_ans = (hs * ans_mask).sum(dim=1) / denom
                    per_layer_answer.append(pooled_ans.squeeze(0).numpy())  # [D]

            captured.clear()
            all_full.append(per_layer_full)
            if do_answer:
                all_answer.append(per_layer_answer)  # type: ignore[union-attr]

    finally:
        for h in hooks:
            h.remove()
        captured.clear()

    # ── Stack into {layer_idx: Tensor[N, D]} ──────────────────────────────────
    result: dict[str, dict[int, torch.Tensor]] = {
        "full": {
            li: torch.tensor(
                np.stack([all_full[ti][li] for ti in range(n_texts)]),
                dtype=torch.float32,
            )
            for li in range(n_layers)
        }
    }

    if do_answer:
        result["answer"] = {
            li: torch.tensor(
                np.stack([all_answer[ti][li] for ti in range(n_texts)]),  # type: ignore[index]
                dtype=torch.float32,
            )
            for li in range(n_layers)
        }

    return result


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_generated(
    prompts: list[str],
    answers: list[str],
) -> dict:
    """Compute sentiment + perplexity for generated (prompt, answer) pairs."""
    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]

    print("[scores] Sentiment (answer only) …")
    sent_ans  = compute_sentiment_metrics(answers,  device=DEVICE)
    print("[scores] Sentiment (combined) …")
    sent_comb = compute_sentiment_metrics(combined, device=DEVICE)

    print("[scores] Perplexity (answer only) …")
    ppl_ans  = compute_perplexity(answers,  device="cpu")
    print("[scores] Perplexity (combined) …")
    ppl_comb = compute_perplexity(combined, device="cpu")

    return {
        "sentiment_answer": {
            "mean_negative":      sent_ans["mean_negative"],
            "negative_fraction":  sent_ans["negative_fraction"],
            "per_text":           sent_ans["scores"].tolist(),
        },
        "sentiment_combined": {
            "mean_negative":      sent_comb["mean_negative"],
            "negative_fraction":  sent_comb["negative_fraction"],
            "per_text":           sent_comb["scores"].tolist(),
        },
        "perplexity_answer": {
            "mean_ppl":  ppl_ans["mean_ppl"],
            "per_text":  ppl_ans["per_text_ppl"],
        },
        "perplexity_combined": {
            "mean_ppl":  ppl_comb["mean_ppl"],
            "per_text":  ppl_comb["per_text_ppl"],
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_prompt_token_lens(prompts: list[str], tokenizer) -> list[int]:
    """Return the tokenised length (no special tokens) for each prompt string."""
    return [
        tokenizer(p, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
        for p in prompts
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    out_texts  = OUT_DIR / "texts"
    out_act    = OUT_DIR / "activations"
    out_scores = OUT_DIR / "scores"
    for d in [out_texts, out_act, out_scores]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load input data ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 1 — Loading prompts  (N={N_SMALL})")
    print(f"{'='*60}")

    neg_prompts = load_amazon_prompts(N_SMALL, label=0, n_words=N_PROMPT_WORDS)
    pos_prompts = load_amazon_prompts(N_SMALL, label=1, n_words=N_PROMPT_WORDS)
    real_neg    = load_amazon_prompts(N_SMALL, label=0, n_words=N_REAL_WORDS)
    real_pos    = load_amazon_prompts(N_SMALL, label=1, n_words=N_REAL_WORDS)

    # Persist real reviews immediately
    for name, texts in [("real_negative", real_neg), ("real_positive", real_pos)]:
        with open(out_texts / f"{name}.json", "w") as f:
            json.dump({"texts": texts, "n_words": N_REAL_WORDS, "label": 0 if "neg" in name else 1}, f, indent=2)
    print(f"[texts] real_negative.json and real_positive.json saved.")

    # ── Step 2: Load model ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 2 — Loading model: {MODEL_NAME}")
    print(f"{'='*60}")
    model     = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    # ── Step 3: Generate continuations ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 3 — Generating continuations")
    print(f"{'='*60}")
    print(f"[gen] Params: {GEN_PARAMS}")

    neg_answers = generate_continuations(model, tokenizer, neg_prompts, "Gen-negative")
    pos_answers = generate_continuations(model, tokenizer, pos_prompts, "Gen-positive")

    for name, prompts, answers in [
        ("generated_negative", neg_prompts, neg_answers),
        ("generated_positive", pos_prompts, pos_answers),
    ]:
        pairs = [
            {"prompt": p, "answer": a, "combined": f"{p} {a}".strip()}
            for p, a in zip(prompts, answers)
        ]
        with open(out_texts / f"{name}.json", "w") as f:
            json.dump({"pairs": pairs, "gen_params": GEN_PARAMS, "model": MODEL_NAME}, f, indent=2)
    print(f"[texts] generated_negative.json and generated_positive.json saved.")

    # ── Step 4: Score generated texts ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 4 — Scoring generated texts")
    print(f"{'='*60}")

    neg_scores = score_generated(neg_prompts, neg_answers)
    pos_scores = score_generated(pos_prompts, pos_answers)

    for name, scores in [
        ("generated_negative_scores", neg_scores),
        ("generated_positive_scores", pos_scores),
    ]:
        with open(out_scores / f"{name}.json", "w") as f:
            json.dump(scores, f, indent=2)
    print(f"[scores] Saved to {out_scores}/")

    # ── Step 5: Extract activations ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 5 — Extracting per-layer mean activations")
    print(f"{'='*60}")

    neg_prompt_lens = get_prompt_token_lens(neg_prompts, tokenizer)
    pos_prompt_lens = get_prompt_token_lens(pos_prompts, tokenizer)

    # ── Group 1: generated_negative ──────────────────────────────────────────
    print(f"\n[act] Group 1/4: generated_negative  ({N_SMALL} examples)")
    neg_combined = [f"{p} {a}".strip() for p, a in zip(neg_prompts, neg_answers)]
    neg_act = extract_mean_layer_activations(
        neg_combined, model, tokenizer, DEVICE,
        prompt_lens_tokens=neg_prompt_lens,
    )
    torch.save(neg_act["full"],   out_act / "generated_negative_full.pt")
    torch.save(neg_act["answer"], out_act / "generated_negative_answer.pt")
    print(f"[act] → generated_negative_full.pt, generated_negative_answer.pt")

    # ── Group 2: generated_positive ──────────────────────────────────────────
    print(f"\n[act] Group 2/4: generated_positive  ({N_SMALL} examples)")
    pos_combined = [f"{p} {a}".strip() for p, a in zip(pos_prompts, pos_answers)]
    pos_act = extract_mean_layer_activations(
        pos_combined, model, tokenizer, DEVICE,
        prompt_lens_tokens=pos_prompt_lens,
    )
    torch.save(pos_act["full"],   out_act / "generated_positive_full.pt")
    torch.save(pos_act["answer"], out_act / "generated_positive_answer.pt")
    print(f"[act] → generated_positive_full.pt, generated_positive_answer.pt")

    # ── Group 3: real_negative ────────────────────────────────────────────────
    print(f"\n[act] Group 3/4: real_negative  ({N_SMALL} examples)")
    real_neg_act = extract_mean_layer_activations(
        real_neg, model, tokenizer, DEVICE,
    )
    torch.save(real_neg_act["full"], out_act / "real_negative.pt")
    print(f"[act] → real_negative.pt")

    # ── Group 4: real_positive ────────────────────────────────────────────────
    print(f"\n[act] Group 4/4: real_positive  ({N_SMALL} examples)")
    real_pos_act = extract_mean_layer_activations(
        real_pos, model, tokenizer, DEVICE,
    )
    torch.save(real_pos_act["full"], out_act / "real_positive.pt")
    print(f"[act] → real_positive.pt")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_layers  = len(neg_act["full"])
    ex_shape  = tuple(neg_act["full"][0].shape)   # (N, hidden_dim)

    print(f"\n{'='*60}")
    print(f"STEERING TEST — {N_SMALL}-EXAMPLE RUN COMPLETE")
    print(f"{'='*60}")
    print(f"\nActivation info:")
    print(f"  Layers extracted : {n_layers}")
    print(f"  Shape per layer  : {ex_shape}  (N_examples × hidden_dim)")
    print(f"\nSentiment scores (mean P(NEGATIVE) on answer-only):")
    print(f"  generated_negative : {neg_scores['sentiment_answer']['mean_negative']:.4f}")
    print(f"  generated_positive : {pos_scores['sentiment_answer']['mean_negative']:.4f}")
    print(f"\nPerplexity scores (mean PPL on answer-only, via Qwen2.5-3B):")
    print(f"  generated_negative : {neg_scores['perplexity_answer']['mean_ppl']:.1f}")
    print(f"  generated_positive : {pos_scores['perplexity_answer']['mean_ppl']:.1f}")

    print(f"\nAll outputs saved under: {OUT_DIR.resolve()}/")
    print(f"""
  texts/
    generated_negative.json   — prompts + generated answers
    generated_positive.json   — prompts + generated answers
    real_negative.json        — real negative reviews ({N_REAL_WORDS} words)
    real_positive.json        — real positive reviews ({N_REAL_WORDS} words)

  activations/
    generated_negative_full.pt    — shape {ex_shape}  (mean over prompt+answer tokens)
    generated_negative_answer.pt  — shape {ex_shape}  (mean over answer tokens only)
    generated_positive_full.pt    — shape {ex_shape}
    generated_positive_answer.pt  — shape {ex_shape}
    real_negative.pt              — shape {ex_shape}
    real_positive.pt              — shape {ex_shape}

  scores/
    generated_negative_scores.json  — sentiment + perplexity
    generated_positive_scores.json  — sentiment + perplexity
""")
    print("To reload activations:")
    print("  import torch")
    print("  act = torch.load('data/steering_test/activations/generated_negative_full.pt')")
    print("  layer_0_vecs = act[0]   # Tensor shape [N, hidden_dim]")
    print("\nTo scale to N=500: set N_SMALL = 500 at the top of this file and re-run.")


if __name__ == "__main__":
    main()
