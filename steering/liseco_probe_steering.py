"""
LiSeCo-style probe steering for LLaDA diffusion generation.

Implements per-layer, per-token hidden-state projection using trained linear
probe parameters:
    f_t(x) = sigmoid(w_t^T x + b_t)

For each controlled token position at each selected layer, we enforce:
    alpha_min <= f_t(x) <= alpha_max

Equivalent logit-space bounds:
    z = w_t^T x + b_t
    z_min = logit(alpha_min)
    z_max = logit(alpha_max)

If z is outside [z_min, z_max], apply minimum-norm correction along w_t:
    if z < z_min:
        delta = ((z_min - z) / (||w_t||^2 + 1e-8)) * w_t
    elif z > z_max:
        delta = ((z_max - z) / (||w_t||^2 + 1e-8)) * w_t

Then x <- x + delta.

Scope of control:
- Applied ONLY at token positions that are [MASK] when passed into the model
  at the current denoising forward pass (captured via a model pre-hook).
- Applied at EVERY denoising step and EVERY selected layer, because hooks run
  on every forward call inside llada/generate.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseSteering
from .precomputed_steering import _get_llada_layers, LLADA_MASK_ID


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    p = float(max(eps, min(1.0 - eps, p)))
    return float(torch.logit(torch.tensor(p, dtype=torch.float32)).item())


@dataclass
class ProbeParams:
    weight: torch.Tensor  # [D], float32 cpu
    bias: torch.Tensor    # [1], float32 cpu
    norm_sq: float


class LiSeCoProbeSteering(BaseSteering):
    """
    LiSeCo-style interval projection steering using trained linear probes.

    Parameters
    ----------
    probe_by_layer:
        Dict layer_idx -> ProbeParams for selected layers.
    layer_ids:
        Which layers to control.
    alpha_min, alpha_max:
        Allowed interval for probe output f_t(x)=sigmoid(w^T x + b).
    mask_id:
        LLaDA mask token id. Control applies only where input_ids == mask_id.
    """

    def __init__(
        self,
        probe_by_layer: Dict[int, ProbeParams],
        layer_ids: List[int],
        alpha_min: float,
        alpha_max: float,
        mask_id: int = LLADA_MASK_ID,
        legacy_mode: bool = True,
    ):
        super().__init__(alpha=1.0)
        if not (0.0 <= alpha_min < alpha_max <= 1.0):
            raise ValueError(f"Invalid interval [{alpha_min}, {alpha_max}]")

        self.probe_by_layer = probe_by_layer
        self.layer_ids = layer_ids
        self.mask_id = mask_id
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.legacy_mode = bool(legacy_mode)

        self.z_min = _safe_logit(alpha_min)
        self.z_max = _safe_logit(alpha_max)

        # updated on every forward pre-hook: [B, L] bool cpu
        self._current_mask_index: Optional[torch.Tensor] = None

        # run-level debug counters
        self.forward_calls = 0
        self.layer_hook_calls = 0
        self.total_masked_positions_seen = 0
        self.total_projected_positions = 0
        self.sum_delta_norm = 0.0
        self.max_delta_norm = 0.0

        # Optional diagnostics rows (used only when legacy_mode=False).
        self.diagnostic_rows: list[dict[str, Any]] = []
        self._diag_context: dict[str, Any] = {}

    def set_diagnostics_context(self, **kwargs: Any) -> None:
        """Attach per-example context to diagnostic rows."""
        self._diag_context = dict(kwargs)

    def pop_diagnostics_rows(self) -> list[dict[str, Any]]:
        rows = self.diagnostic_rows
        self.diagnostic_rows = []
        return rows

    def register_hooks(self, model: nn.Module) -> None:
        self._current_mask_index = None

        def _pre_hook(module, args):
            self.forward_calls += 1
            if isinstance(args, (tuple, list)) and len(args) > 0:
                inp = args[0]
                if isinstance(inp, torch.Tensor) and inp.dtype in (torch.long, torch.int32):
                    self._current_mask_index = (inp == self.mask_id).cpu()

        self._hooks.append(model.register_forward_pre_hook(_pre_hook))

        layers = _get_llada_layers(model)

        def _make_layer_hook(layer_idx: int):
            probe = self.probe_by_layer[layer_idx]
            w_cpu = probe.weight.float().cpu()  # [D]
            b_cpu = probe.bias.float().cpu()    # [1]
            norm_sq = float(probe.norm_sq)

            def hook(module, inp, output):
                self.layer_hook_calls += 1

                hs = output[0] if isinstance(output, tuple) else output
                # hs: [B, L, D]
                B, L, D = hs.shape

                if (
                    self._current_mask_index is not None
                    and self._current_mask_index.shape[-1] == L
                ):
                    mask = self._current_mask_index.to(device=hs.device)  # [B, L] bool
                else:
                    mask = torch.ones(B, L, dtype=torch.bool, device=hs.device)

                self.total_masked_positions_seen += int(mask.sum().item())

                # Flatten for row-wise control on controlled positions only
                flat_hs = hs.reshape(B * L, D)
                flat_mask = mask.reshape(B * L)
                idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # [M]

                if idx.numel() > 0:
                    x = flat_hs[idx]  # [M, D]
                    w = w_cpu.to(device=hs.device, dtype=hs.dtype).view(D, 1)  # [D,1]
                    b = b_cpu.to(device=hs.device, dtype=hs.dtype).view(1)      # [1]

                    # z = w^T x + b, vectorized over rows
                    z_pre = (x @ w).squeeze(-1) + b   # [M]

                    # compute correction coefficient per row
                    coef = torch.zeros_like(z_pre)
                    low = z_pre < self.z_min
                    high = z_pre > self.z_max
                    if low.any():
                        coef[low] = (self.z_min - z_pre[low]) / (norm_sq + 1e-8)
                    if high.any():
                        coef[high] = (self.z_max - z_pre[high]) / (norm_sq + 1e-8)

                    corrected = (low | high)
                    n_corr = int(corrected.sum().item())
                    self.total_projected_positions += n_corr

                    # delta[row, :] = coef[row] * w
                    delta = coef.view(-1, 1) * w.view(1, -1)  # [M, D]

                    if n_corr > 0:
                        x = x + delta
                        flat_hs[idx] = x
                        hs = flat_hs.view(B, L, D)

                        # Update delta norm stats
                        with torch.no_grad():
                            delta_norms = torch.linalg.norm(delta[corrected], dim=-1)
                            self.sum_delta_norm += float(delta_norms.sum().item())
                            self.max_delta_norm = max(self.max_delta_norm, float(delta_norms.max().item()))

                    if not self.legacy_mode:
                        z_post = z_pre + coef * norm_sq
                        score_pre = torch.sigmoid(z_pre)
                        score_post = torch.sigmoid(z_post)
                        delta_norm = torch.linalg.norm(delta, dim=-1)
                        pos = idx.detach().cpu()
                        row_idx = torch.div(pos, L, rounding_mode="floor")
                        token_pos = torch.remainder(pos, L)

                        z_pre_cpu = z_pre.detach().cpu()
                        z_post_cpu = z_post.detach().cpu()
                        score_pre_cpu = score_pre.detach().cpu()
                        score_post_cpu = score_post.detach().cpu()
                        corrected_cpu = corrected.detach().cpu()
                        delta_norm_cpu = delta_norm.detach().cpu()

                        for i in range(pos.numel()):
                            row = {
                                "forward_call_idx": int(self.forward_calls),
                                "layer": int(layer_idx),
                                "batch_idx": int(row_idx[i].item()),
                                "token_pos": int(token_pos[i].item()),
                                "z_pre": float(z_pre_cpu[i].item()),
                                "z_post": float(z_post_cpu[i].item()),
                                "score_pre": float(score_pre_cpu[i].item()),
                                "score_post": float(score_post_cpu[i].item()),
                                "delta_z": float((z_post_cpu[i] - z_pre_cpu[i]).item()),
                                "corrected": bool(corrected_cpu[i].item()),
                                "delta_norm": float(delta_norm_cpu[i].item()),
                                "alpha_min": float(self.alpha_min),
                                "alpha_max": float(self.alpha_max),
                                "z_min": float(self.z_min),
                                "z_max": float(self.z_max),
                            }
                            row.update(self._diag_context)
                            self.diagnostic_rows.append(row)

                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs

            return hook

        for layer_idx in self.layer_ids:
            self._hooks.append(layers[layer_idx].register_forward_hook(_make_layer_hook(layer_idx)))

        print(
            f"[LiSeCoProbeSteering] Hooks on layers={self.layer_ids} "
            f"interval=[{self.alpha_min:.2f},{self.alpha_max:.2f}]"
        )


def load_probe_params(
    probes_root: Path,
    family: str,
    layer_ids: List[int],
) -> Dict[int, ProbeParams]:
    """
    Load probe weight/bias artifacts produced by train_probe_llada.py.

    Expected file per layer:
      {probes_root}/{family}/layer_XX/probe_weight_bias.pt

    family examples:
      - merged_probes
      - sanity_probes
    """
    probe_by_layer: Dict[int, ProbeParams] = {}

    for layer in layer_ids:
        p = probes_root / family / f"layer_{layer:02d}" / "probe_weight_bias.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"Probe artifact missing for layer {layer}: {p}"
            )
        obj = torch.load(p, map_location="cpu", weights_only=False)
        w = obj["weight"].float().cpu().view(-1)
        b = obj["bias"].float().cpu().view(-1)
        norm_sq = float((w * w).sum().item())
        probe_by_layer[layer] = ProbeParams(weight=w, bias=b, norm_sq=norm_sq)

    return probe_by_layer
