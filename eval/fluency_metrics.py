"""
Fluency & quality metrics for generated text.

Metrics computed:
  * **Perplexity** under Qwen2.5-3B — standard proxy for fluency / coherence.
  * **Repetition rate** — fraction of unique n-grams (lower = more repetition).
  * **Mean generation length** (in tokens / characters).

Reference
---------
Holtzman et al. (2020) – "The Curious Case of Neural Text Degeneration".
https://arxiv.org/abs/1904.09751
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B"
_qwen_model = None
_qwen_tokenizer = None


def _get_qwen(device: str = "cpu"):
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        _qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        if _qwen_tokenizer.pad_token is None:
            _qwen_tokenizer.pad_token = _qwen_tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _qwen_model = (
            AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_NAME,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
    return _qwen_model, _qwen_tokenizer


@torch.no_grad()
def compute_perplexity(
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 256,
    device: str = "cpu",
) -> dict:
    """
    Compute mean perplexity of `texts` under Qwen2.5-3B.

    Args:
        texts:      List of strings.
        batch_size: Inference batch size.
        max_length: Truncate texts to this many tokens.
        device:     Torch device.

    Returns:
        dict with ``mean_ppl`` and ``per_text_ppl`` (list of floats).
    """
    model, tokenizer = _get_qwen(device)
    ppls = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        logits = model(input_ids, attention_mask=attn_mask).logits   # [B, L, V]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attn_mask[:, 1:].float()

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape)

        # Masked mean per sequence
        seq_loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        ppls.extend(seq_loss.exp().cpu().tolist())

    return {
        "mean_ppl": float(np.mean(ppls)),
        "per_text_ppl": ppls,
    }


def compute_repetition_rate(
    texts: List[str],
    n: int = 4,
    tokenize: bool = False,
) -> dict:
    """
    Measure repetition as fraction of *distinct* n-grams.

    distinct-n = |unique n-grams| / |total n-grams|.
    A value of 1 = no repetition; values close to 0 = highly repetitive.

    Args:
        texts:    List of strings.
        n:        N-gram order.
        tokenize: If True, tokenise on whitespace; otherwise use characters.

    Returns:
        dict with ``mean_distinct_n`` and ``per_text_distinct_n``.
    """
    rates = []
    for text in texts:
        tokens = text.split() if tokenize else list(text)
        if len(tokens) <= n:
            rates.append(1.0)
            continue
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        rates.append(len(set(ngrams)) / len(ngrams))
    return {
        "mean_distinct_n": float(np.mean(rates)),
        "per_text_distinct_n": rates,
    }


def compute_fluency_metrics(
    texts: List[str],
    batch_size: int = 16,
    device: str = "cpu",
) -> dict:
    """
    Convenience wrapper: compute all fluency metrics at once.

    Returns a flat dict with all keys from `compute_perplexity` and
    `compute_repetition_rate`.
    """
    metrics = {}
    metrics.update(compute_perplexity(texts, batch_size=batch_size, device=device))
    metrics.update(compute_repetition_rate(texts))
    metrics["mean_length_words"] = float(
        np.mean([len(t.split()) for t in texts])
    )
    return metrics
