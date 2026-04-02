"""
sentiment_metrics.py – Sentiment scoring via distilbert-base-uncased-finetuned-sst-2-english.

API mirrors toxicity_metrics.py: compute_sentiment_metrics(texts, device)
returns a dict with mean_negative, negative_fraction, max_negative, scores
(where scores = P(NEGATIVE) ∈ [0, 1], higher = more negative).
"""
from __future__ import annotations

import numpy as np
from transformers import pipeline as hf_pipeline

_SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
_classifier = None
_classifier_device: str | None = None


def _get_classifier(device: str):
    global _classifier, _classifier_device
    if _classifier is None or _classifier_device != device:
        dev_arg = 0 if device == "cuda" else -1
        _classifier = hf_pipeline(
            "text-classification",
            model=_SENTIMENT_MODEL,
            device=dev_arg,
            top_k=None,   # return both POSITIVE and NEGATIVE scores
        )
        _classifier_device = device
    return _classifier


def compute_sentiment_metrics(texts: list[str], device: str = "cpu") -> dict:
    """
    Score each text with P(NEGATIVE).  Higher = more negative sentiment.

    Returns:
        mean_negative     – mean P(NEGATIVE) across texts
        negative_fraction – fraction where P(NEGATIVE) ≥ 0.5
        max_negative      – highest P(NEGATIVE) in the batch
        scores            – np.float32 array of P(NEGATIVE), one per text
    """
    clf = _get_classifier(device)
    safe = [t.strip() or "." for t in texts]
    results = clf(safe, truncation=True, max_length=512)

    scores = np.array(
        [
            next((d["score"] for d in item if d["label"] == "NEGATIVE"), 0.0)
            for item in results
        ],
        dtype=np.float32,
    )

    return {
        "mean_negative":     float(scores.mean()),
        "negative_fraction": float((scores >= 0.5).mean()),
        "max_negative":      float(scores.max()),
        "scores":            scores,
    }
