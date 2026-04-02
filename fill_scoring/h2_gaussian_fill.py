"""
H2 fill-stage Gaussian-aware scoring.

This scorer mirrors the structure of H2SAEAwareFillScorer but replaces the SAE
residual score with a Gaussian in-distribution desirability score.

Direction convention:
    direction = mu_pos_train - mu_neg_train

Target Gaussian selection:
    - steering toward positive: score with the positive Gaussian
    - steering toward negative: score with the negative Gaussian

Raw score per token is negative diagonal Mahalanobis distance:
    raw = -sum((h - mu)^2 / var)

Normalization to [0, 1] is done per layer and per step over current candidates.
"""

from __future__ import annotations

from typing import Dict, Literal

import torch

from activations_modeling.gaussian.gaussian_models import LayerGaussianPair
from fill_scoring.base import BaseFillScorer
from steering.precomputed_steering import SteeringHiddenStateBuffer


class H2GaussianFillScorer(BaseFillScorer):
    """Gaussian-aware fill scorer with optional per-step debug records."""

    def __init__(
        self,
        buffer: SteeringHiddenStateBuffer,
        gaussian_models_by_layer: dict[int, LayerGaussianPair],
        monitored_layers: list[int],
        steer_target_class: Literal["positive", "negative"],
        alpha_mix: float = 0.5,
        enable_debug: bool = False,
    ):
        self.buffer = buffer
        self.gaussian_models_by_layer = gaussian_models_by_layer
        self.monitored_layers = list(monitored_layers)
        if steer_target_class not in ("positive", "negative"):
            raise ValueError("steer_target_class must be 'positive' or 'negative'")
        self.steer_target_class = steer_target_class
        self.alpha_mix = float(alpha_mix)
        self.enable_debug = enable_debug

        self.debug_records: list[dict] = []
        self._pending: dict[tuple[int, int, int], int] = {}

    def reset(self) -> None:
        self.debug_records = []
        self._pending = {}

    @staticmethod
    def _neg_md2_diag(h: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        diff = h - mean.view(1, 1, -1)
        md2 = ((diff * diff) / var.view(1, 1, -1)).sum(dim=-1)
        return -md2

    def _normalize_candidate_scores(self, raw_scores: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        vals = raw_scores[cand_mask]
        s_min = vals.min()
        s_max = vals.max()
        if (s_max - s_min).abs() < 1e-8:
            out = torch.full_like(raw_scores, 0.5)
            return out
        out = (raw_scores - s_min) / (s_max - s_min + 1e-8)
        return out.clamp_(0.0, 1.0)

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
        bsz, _ = x_curr.shape
        device = x_curr.device

        out_scores = torch.full_like(baseline_scores, float("-inf"))

        candidates = mask_index.clone()
        candidates[:, :prompt_len] = False
        candidates[:, block_end:] = False

        raw_by_layer: Dict[int, torch.Tensor] = {}
        for layer in self.monitored_layers:
            hs_cpu = self.buffer.last_hidden_by_layer.get(layer)
            if hs_cpu is None:
                continue
            model = self.gaussian_models_by_layer.get(layer)
            if model is None:
                continue
            hs = hs_cpu.to(device=device, dtype=torch.float32)
            target = model.positive if self.steer_target_class == "positive" else model.negative
            mean = target.mean.to(device=device, dtype=torch.float32)
            var = target.var.to(device=device, dtype=torch.float32)
            raw_by_layer[layer] = self._neg_md2_diag(hs, mean=mean, var=var)

        for b in range(bsz):
            cand = candidates[b]
            if not cand.any():
                continue

            baseline_b = baseline_scores[b]
            layer_scores: list[torch.Tensor] = []
            debug_raw: dict[str, list[float]] = {}
            debug_norm: dict[str, list[float]] = {}

            for layer, raw in raw_by_layer.items():
                raw_b = raw[b]
                norm_b = self._normalize_candidate_scores(raw_b, cand)
                layer_scores.append(norm_b)

                if self.enable_debug:
                    debug_raw[str(layer)] = raw_b[cand].detach().cpu().tolist()
                    debug_norm[str(layer)] = norm_b[cand].detach().cpu().tolist()

            if layer_scores:
                gaussian_fill_score = torch.stack(layer_scores, dim=0).mean(dim=0)
            else:
                gaussian_fill_score = baseline_b

            mixed = self.alpha_mix * baseline_b + (1.0 - self.alpha_mix) * gaussian_fill_score
            out_scores[b, cand] = mixed[cand]

            if self.enable_debug:
                cand_pos = cand.nonzero(as_tuple=True)[0].detach().cpu().tolist()
                rec = {
                    "alpha_mix": self.alpha_mix,
                    "block_idx": block_idx,
                    "step_i": step_i,
                    "batch_idx": b,
                    "candidate_pos": cand_pos,
                    "steer_target_class": self.steer_target_class,
                    "gaussian_score_definition": "negative_diagonal_mahalanobis",
                    "gaussian_normalization": "per-step per-layer minmax over candidates to [0,1]",
                    "baseline_fill_score": baseline_b[cand].detach().cpu().tolist(),
                    "raw_gaussian_by_layer": debug_raw,
                    "norm_gaussian_by_layer": debug_norm,
                    "gaussian_fill_score": gaussian_fill_score[cand].detach().cpu().tolist(),
                    "mixed_fill_score": mixed[cand].detach().cpu().tolist(),
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
