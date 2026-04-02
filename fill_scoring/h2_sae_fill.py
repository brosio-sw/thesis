"""
H2 fill-stage SAE-aware scoring.

This module mixes a baseline fill score with an SAE-derived manifold score at
commit/unmask selection time (top-k over currently masked generation tokens).

    mixed_fill_score = alpha_mix * baseline_fill_score
                     + (1 - alpha_mix) * sae_fill_score

SAE computation:
  - use post-steering hidden states from monitored layers
  - residual norm per token: ||h - h_recon||
  - layerwise min-max normalization over candidate tokens only
  - aggregate by averaging across monitored layers

Score direction assumption:
  residual is an OOD score (higher = more off-manifold), while fill ranking
  expects higher = better to commit now. Therefore we convert to a commit score:

    sae_fill_score = 1 - normalized_residual
"""

from __future__ import annotations

from typing import Dict

import torch

from fill_scoring.base import BaseFillScorer
from remasking.h2_sae_conf import LayerwiseTopKSAEReconstructor
from steering.precomputed_steering import SteeringHiddenStateBuffer


class H2SAEAwareFillScorer(BaseFillScorer):
    """SAE-aware fill scorer with optional per-step debug records."""

    def __init__(
        self,
        buffer: SteeringHiddenStateBuffer,
        sae_reconstructor: LayerwiseTopKSAEReconstructor,
        monitored_layers: list[int],
        alpha_mix: float = 0.5,
        enable_debug: bool = False,
    ):
        self.buffer = buffer
        self.sae_reconstructor = sae_reconstructor
        self.monitored_layers = list(monitored_layers)
        self.alpha_mix = float(alpha_mix)
        self.enable_debug = enable_debug
        self.debug_records: list[dict] = []
        self._pending: dict[tuple[int, int, int], int] = {}

    def reset(self) -> None:
        self.debug_records = []
        self._pending = {}

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
        B, L = x_curr.shape
        device = x_curr.device

        out_scores = torch.full_like(baseline_scores, float("-inf"))

        # Fill candidates: currently masked generation tokens in the current block.
        candidates = mask_index.clone()
        candidates[:, :prompt_len] = False
        candidates[:, block_end:] = False

        raw_sae_by_layer: Dict[int, torch.Tensor] = {}
        for layer in self.monitored_layers:
            hs_cpu = self.buffer.last_hidden_by_layer.get(layer)
            if hs_cpu is None:
                continue
            hs = hs_cpu.to(device=device, dtype=torch.float32)
            raw_sae_by_layer[layer] = self.sae_reconstructor.residual_norm(layer, hs)

        for b in range(B):
            cand = candidates[b]
            if not cand.any():
                continue

            baseline_b = baseline_scores[b]
            layer_fill_scores: list[torch.Tensor] = []
            debug_raw: dict[str, list[float]] = {}
            debug_norm_resid: dict[str, list[float]] = {}

            for layer, raw in raw_sae_by_layer.items():
                layer_raw = raw[b]
                vals = layer_raw[cand]
                r_min = vals.min()
                r_max = vals.max()
                norm_resid = (layer_raw - r_min) / (r_max - r_min + 1e-8)
                # Convert OOD residual into commit desirability score.
                layer_fill = 1.0 - norm_resid
                layer_fill_scores.append(layer_fill)

                if self.enable_debug:
                    debug_raw[str(layer)] = vals.detach().cpu().tolist()
                    debug_norm_resid[str(layer)] = norm_resid[cand].detach().cpu().tolist()

            if layer_fill_scores:
                sae_fill_score = torch.stack(layer_fill_scores, dim=0).mean(dim=0)
            else:
                # If SAE scores are unavailable, preserve baseline behavior.
                sae_fill_score = baseline_b

            mixed = self.alpha_mix * baseline_b + (1.0 - self.alpha_mix) * sae_fill_score
            out_scores[b, cand] = mixed[cand]

            if self.enable_debug:
                cand_pos = cand.nonzero(as_tuple=True)[0].detach().cpu().tolist()
                rec = {
                    "alpha_mix": self.alpha_mix,
                    "block_idx": block_idx,
                    "step_i": step_i,
                    "batch_idx": b,
                    "candidate_pos": cand_pos,
                    "baseline_fill_score": baseline_b[cand].detach().cpu().tolist(),
                    "raw_sae_by_layer": debug_raw,
                    "norm_residual_by_layer": debug_norm_resid,
                    "sae_fill_score": sae_fill_score[cand].detach().cpu().tolist(),
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
