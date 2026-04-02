"""
H3 fill-stage sentiment-gradient scoring.

This scorer mirrors the H2 fill-scoring interface but replaces the auxiliary
score with a classifier-attribution signal:

    g_j = || dS(x) / de_j ||_2

where S(x) is the NEGATIVE sentiment logit from DistilBERT SST-2 and e_j is the
input embedding at classifier token j.

At each denoising step:
1) Build a fully filled provisional sequence by replacing masked positions in
   x_curr with x0.
2) Decode that sequence to text.
3) Run DistilBERT on that text and compute per-token gradient norms.
4) Approximate-map classifier token scores back to eligible LLaDA candidate
   positions using overlapping character spans.
5) Min-max normalize mapped scores over eligible candidates only.
6) Mix with baseline commit score:

    mixed_fill_score = alpha_mix * baseline_fill_score
                     + (1 - alpha_mix) * grad_fill_score

Higher mapped gradient score means higher commit desirability.
"""

from __future__ import annotations

from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fill_scoring.base import BaseFillScorer


_SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class SentimentGradientFillScorer(BaseFillScorer):
    """Classifier-attribution fill scorer with optional per-step debug records."""

    def __init__(
        self,
        llada_tokenizer,
        alpha_mix: float = 0.8,
        sentiment_model_name: str = _SENTIMENT_MODEL,
        sentiment_device: str = "cpu",
        sentiment_tokenizer=None,
        sentiment_model=None,
        enable_debug: bool = False,
        max_debug_classifier_tokens: int = 96,
    ):
        self.llada_tokenizer = llada_tokenizer
        self.alpha_mix = float(alpha_mix)
        self.enable_debug = enable_debug
        self.max_debug_classifier_tokens = int(max_debug_classifier_tokens)

        self.sentiment_device = torch.device(sentiment_device)
        self.sentiment_tokenizer = (
            sentiment_tokenizer
            if sentiment_tokenizer is not None
            else AutoTokenizer.from_pretrained(sentiment_model_name, use_fast=True)
        )
        self.sentiment_model = (
            sentiment_model
            if sentiment_model is not None
            else AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        )
        self.sentiment_model.to(self.sentiment_device)
        self.sentiment_model.eval()

        label2id = {str(k).upper(): int(v) for k, v in getattr(self.sentiment_model.config, "label2id", {}).items()}
        self.neg_label_id = int(label2id.get("NEGATIVE", 0))

        self.debug_records: list[dict] = []
        self._pending: dict[tuple[int, int, int], int] = {}

    def reset(self) -> None:
        self.debug_records = []
        self._pending = {}

    @staticmethod
    def _normalize_candidate_scores(raw_scores: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        vals = raw_scores[cand_mask]
        s_min = vals.min()
        s_max = vals.max()
        if (s_max - s_min).abs() < 1e-8:
            return torch.full_like(raw_scores, 0.5)
        out = (raw_scores - s_min) / (s_max - s_min + 1e-8)
        return out.clamp_(0.0, 1.0)

    def _decode_llada(self, token_ids: list[int]) -> str:
        return self.llada_tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _candidate_spans(self, token_ids: list[int], candidate_pos: list[int]) -> dict[int, tuple[int, int]]:
        """
        Approximate LLaDA char spans with prefix decoding for candidate positions.

        For position p:
          span = [len(decode(ids[:p])), len(decode(ids[:p+1])))
        """
        spans: dict[int, tuple[int, int]] = {}
        for p in candidate_pos:
            if p < 0 or p >= len(token_ids):
                spans[p] = (0, 0)
                continue
            left = self._decode_llada(token_ids[:p])
            right = self._decode_llada(token_ids[: p + 1])
            spans[p] = (len(left), len(right))
        return spans

    def _classifier_token_grad_norms(self, text: str) -> tuple[list[tuple[int, int]], list[float], list[str], float]:
        """
        Return classifier token offsets, grad-norm attribution, token strings,
        and the NEGATIVE logit scalar used for attribution.
        """
        safe_text = text if text.strip() else "."

        with torch.enable_grad():
            enc = self.sentiment_tokenizer(
                safe_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            input_ids = enc["input_ids"].to(self.sentiment_device)
            attention_mask = enc["attention_mask"].to(self.sentiment_device)

            emb_layer = self.sentiment_model.get_input_embeddings()
            embeds = emb_layer(input_ids).detach()
            embeds.requires_grad_(True)

            self.sentiment_model.zero_grad(set_to_none=True)
            logits = self.sentiment_model(inputs_embeds=embeds, attention_mask=attention_mask).logits
            neg_logit = logits[0, self.neg_label_id]
            neg_logit.backward()

            grad_norm = embeds.grad[0].norm(dim=-1).detach().cpu().tolist()
            toks = self.sentiment_tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())

        return offsets, grad_norm, toks, float(neg_logit.detach().cpu().item())

    @staticmethod
    def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return min(a[1], b[1]) > max(a[0], b[0])

    def build_fill_scores(
        self,
        x_curr: torch.LongTensor,
        logits: torch.FloatTensor,
        x0: torch.LongTensor,
        baseline_scores: torch.FloatTensor,
        mask_index: torch.BoolTensor,
        prompt_len: int,
        block_start: int,
        block_end: int,
        block_idx: int,
        step_i: int,
    ) -> torch.FloatTensor:
        bsz, seq_len = x_curr.shape
        device = x_curr.device

        out_scores = torch.full_like(baseline_scores, float("-inf"))

        candidates = mask_index.clone()
        candidates[:, :prompt_len] = False
        candidates[:, block_end:] = False

        for b in range(bsz):
            cand = candidates[b]
            if not cand.any():
                continue

            baseline_b = baseline_scores[b]

            provisional_ids = torch.where(mask_index[b], x0[b], x_curr[b]).detach().cpu().tolist()
            provisional_text = self._decode_llada(provisional_ids)
            cand_pos = cand.nonzero(as_tuple=True)[0].detach().cpu().tolist()

            offsets, cls_grad_norms, cls_tokens, neg_logit = self._classifier_token_grad_norms(provisional_text)
            llada_spans = self._candidate_spans(provisional_ids, cand_pos)

            raw_mapped = torch.zeros(seq_len, dtype=torch.float32, device=device)
            mapped_before_norm_for_debug: list[float] = []

            for p in cand_pos:
                span = llada_spans.get(p, (0, 0))
                if span[1] <= span[0]:
                    raw = 0.0
                else:
                    overlap_vals: list[float] = []
                    for (off, g) in zip(offsets, cls_grad_norms):
                        cls_span = (int(off[0]), int(off[1]))
                        if cls_span[1] <= cls_span[0]:
                            continue
                        if self._spans_overlap(span, cls_span):
                            overlap_vals.append(float(g))
                    raw = float(sum(overlap_vals) / len(overlap_vals)) if overlap_vals else 0.0
                raw_mapped[p] = raw
                mapped_before_norm_for_debug.append(raw)

            grad_fill_score = self._normalize_candidate_scores(raw_mapped, cand)
            mixed = self.alpha_mix * baseline_b + (1.0 - self.alpha_mix) * grad_fill_score
            out_scores[b, cand] = mixed[cand]

            if self.enable_debug:
                n_tok = min(self.max_debug_classifier_tokens, len(cls_grad_norms))
                rec = {
                    "alpha_mix": self.alpha_mix,
                    "block_idx": block_idx,
                    "step_i": step_i,
                    "batch_idx": b,
                    "candidate_pos": cand_pos,
                    "baseline_fill_score": baseline_b[cand].detach().cpu().tolist(),
                    "mapped_grad_score_before_norm": mapped_before_norm_for_debug,
                    "grad_fill_score": grad_fill_score[cand].detach().cpu().tolist(),
                    "mixed_fill_score": mixed[cand].detach().cpu().tolist(),
                    "raw_grad_norm_by_classifier_token": {
                        "tokens": cls_tokens[:n_tok],
                        "offsets": offsets[:n_tok],
                        "grad_norm": cls_grad_norms[:n_tok],
                        "truncated": len(cls_grad_norms) > n_tok,
                    },
                    "classifier_negative_logit": neg_logit,
                    "provisional_text": provisional_text[:600],
                    "chosen_commit_pos": [],
                }
                self._pending[(block_idx, step_i, b)] = len(self.debug_records)
                self.debug_records.append(rec)

        return out_scores

    def record_selection(self, block_idx: int, step_i: int, batch_idx: int, selected_pos: torch.LongTensor) -> None:
        if not self.enable_debug:
            return
        key = (block_idx, step_i, batch_idx)
        if key not in self._pending:
            return
        idx = self._pending[key]
        self.debug_records[idx]["chosen_commit_pos"] = selected_pos.detach().cpu().tolist()
