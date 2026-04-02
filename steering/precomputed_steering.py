"""
PrecomputedLayerSteering – additive mean-activation steering from pre-saved vectors.

Applies steering vectors (loaded from .pt files produced by steering_test.py) at
selected transformer layers during LLaDA generation, using the existing hook API
in llada/generate.py.

Design
------
* Direction: mean_positive_layer - mean_negative_layer,  L2-normalised per layer.
* Hook action: hidden_state += alpha * v_layer  (additive → toward positive)
* Scope: only positions that were [MASK] tokens when input to the model at that
  denoising step.  This is resolved via a forward_pre_hook that captures the
  input_ids just before each forward pass, so prompt tokens are left unsteered.

LLaDA layer path
----------------
LLaDA-8B-Base/Instruct uses model.model.transformer['blocks'] (ModuleList),
not model.model.layers as in a standard LLaMA.  `_get_llada_layers` handles this
with fallbacks for portability.

Usage
-----
    from steering.precomputed_steering import PrecomputedLayerSteering
    steerer = PrecomputedLayerSteering(
        vectors={23: torch.zeros(4096)},   # normalised direction per layer
        layer_ids=[23],
        alpha=2.0,
        mask_id=126336,
    )
    # pass to llada_generate(..., steering=steerer)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseSteering


LLADA_MASK_ID = 126336  # [MASK] token id in LLaDA vocabulary


def _get_llada_layers(model: nn.Module) -> list:
    """
    Return the list of transformer block modules for a LLaDA model.

    Priority order:
      1. model.model.transformer['blocks']  – LLaDA-8B-Base/Instruct (MPT-style)
      2. model.model.layers                 – standard LLaMA hierarchy
      3. model.layers                       – flat hierarchy
    """
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
        "Cannot locate transformer layers. "
        "Tried model.model.transformer['blocks'], model.model.layers, model.layers."
    )


class PrecomputedLayerSteering(BaseSteering):
    """
    Steering from precomputed per-layer direction vectors (loaded from .pt files).

    The class registers two types of hooks:
      1. A forward_pre_hook on the top-level model to capture which positions
         are [MASK] tokens at the start of each forward pass.
      2. Per-layer forward_hooks that add alpha * v_layer to the hidden states
         at masked positions only.

    Parameters
    ----------
    vectors:
        Dict mapping layer_idx -> 1-D tensor of shape [hidden_dim].
        Should already be L2-normalised (see compute_steering_vectors helper).
    layer_ids:
        List of layer indices to steer.  Must be keys of `vectors`.
    alpha:
        Steering strength.  Positive alpha steers toward the direction encoded
        in `vectors` (i.e., toward positive sentiment if vectors = pos - neg).
    mask_id:
        Token id used for [MASK] positions in LLaDA (default 126336).
    """

    def __init__(
        self,
        vectors: Dict[int, torch.Tensor],
        layer_ids: List[int],
        alpha: float = 1.0,
        mask_id: int = LLADA_MASK_ID,
    ):
        super().__init__(alpha=alpha)
        self.vectors   = vectors
        self.layer_ids = layer_ids
        self.mask_id   = mask_id
        # Mutable state: updated by the pre-hook before every forward pass.
        # Shape [B, L] bool or None if not yet set.
        self._current_mask_index: Optional[torch.Tensor] = None

    def register_hooks(self, model: nn.Module) -> None:
        """
        Attach hooks to `model`.  Called once by llada_generate before the loop.
        """
        self._current_mask_index = None

        # ── Pre-hook: capture which positions are [MASK] before each fwd pass ──
        def _pre_hook(module, args):
            # args[0] is input_ids for the top-level model call
            # model(x, attention_mask=...) → args = (x,)
            if isinstance(args, (tuple, list)) and len(args) > 0:
                inp = args[0]
                if isinstance(inp, torch.Tensor) and inp.dtype in (torch.long, torch.int32):
                    # Store on CPU to avoid holding extra GPU memory;
                    # layer hooks will move it to the correct device.
                    self._current_mask_index = (inp == self.mask_id).cpu()

        pre_h = model.register_forward_pre_hook(_pre_hook)
        self._hooks.append(pre_h)

        # ── Per-layer forward hooks: apply steering at masked positions ─────────
        layers = _get_llada_layers(model)

        # Determine a concrete device for the vectors (avoid 'meta' parameters).
        vec_device = torch.device("cpu")
        for p in model.parameters():
            if p.device.type not in ("meta", "cpu"):
                vec_device = p.device
                break

        def _make_layer_hook(layer_idx: int):
            # Pre-move the vector to a reasonable device; the hook will also
            # move it to match the actual hidden-state device lazily.
            vec_cpu = self.vectors[layer_idx].float().cpu()  # [D]

            def hook(module, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                # hs: [B, L, D]

                # Build the positional mask: [B, L, 1] float
                if self._current_mask_index is not None and (
                    self._current_mask_index.shape[-1] == hs.shape[1]
                ):
                    pos_mask = (
                        self._current_mask_index
                        .to(device=hs.device, dtype=hs.dtype)
                        .unsqueeze(-1)                        # [B, L, 1]
                    )
                else:
                    # Fallback: apply to all positions (safe default)
                    pos_mask = torch.ones(
                        hs.shape[0], hs.shape[1], 1,
                        device=hs.device, dtype=hs.dtype,
                    )

                # Steering: add alpha * v at masked positions only
                vec = vec_cpu.to(device=hs.device, dtype=hs.dtype)   # [D]
                delta = self.alpha * vec.unsqueeze(0).unsqueeze(0)    # [1, 1, D]
                hs = hs + pos_mask * delta                             # broadcast

                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs

            return hook

        for ℓ in self.layer_ids:
            h = layers[ℓ].register_forward_hook(_make_layer_hook(ℓ))
            self._hooks.append(h)

        print(
            f"[PrecomputedSteering] Registered hooks on layers {self.layer_ids} "
            f"with α={self.alpha:.3f}  (masked-positions only)"
        )

    # remove_hooks() is inherited from BaseSteering


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude-capturing extension for H1 remasking
# ─────────────────────────────────────────────────────────────────────────────

class SteeringMagnitudeBuffer:
    """
    Shared mutable buffer written by MagnitudeCaptureSteering during each
    forward pass and read by H1SteeringConfRemasking during the remasking step.

    Timing guarantee (from generate.py):
        forward pass → layer hooks write magnitudes → remasking reads magnitudes

    Shape of `last_magnitudes`:
        [B, L] float32, on CPU.
        last_magnitudes[b, i]  =  sum over steered layers l of
                                  ||applied_delta_{b,i,l}||_2
        where applied_delta = alpha * v_l  if position i was [MASK] at input,
                            = 0           otherwise.
    """

    def __init__(self):
        self.last_magnitudes: Optional[torch.Tensor] = None

    def reset(self):
        self.last_magnitudes = None


class MagnitudeCaptureSteering(PrecomputedLayerSteering):
    """
    Extension of PrecomputedLayerSteering that additionally accumulates the
    per-token L2 norm of the applied steering delta into a SteeringMagnitudeBuffer.

    Each layer hook:
      1.  Applies the additive steering (identical to the parent class).
      2.  Computes ||alpha * v_l * pos_mask||_2  per token and accumulates it
          into `_magnitude_accum`.  After the last steered layer, the buffer
          holds the full sum across layers.

    Parameters
    ----------
    magnitude_buffer:
        A SteeringMagnitudeBuffer instance shared with the H1 remasking class.
        If None, a fresh buffer is created (useful when steering alone is enough).
    All other parameters are identical to PrecomputedLayerSteering.
    """

    def __init__(
        self,
        vectors: Dict[int, torch.Tensor],
        layer_ids: List[int],
        alpha: float = 1.0,
        mask_id: int = LLADA_MASK_ID,
        magnitude_buffer: Optional["SteeringMagnitudeBuffer"] = None,
    ):
        super().__init__(vectors=vectors, layer_ids=layer_ids, alpha=alpha, mask_id=mask_id)
        self.magnitude_buffer: SteeringMagnitudeBuffer = (
            magnitude_buffer if magnitude_buffer is not None else SteeringMagnitudeBuffer()
        )
        self._magnitude_accum: Optional[torch.Tensor] = None  # [B, L] on CPU

    def register_hooks(self, model: nn.Module) -> None:
        """Register hooks for steering AND magnitude capture."""
        self._current_mask_index = None
        self._magnitude_accum = None
        self.magnitude_buffer.reset()

        # ── Pre-hook: capture [MASK] positions and reset accumulator ─────────
        def _pre_hook(module, args):
            if isinstance(args, (tuple, list)) and len(args) > 0:
                inp = args[0]
                if isinstance(inp, torch.Tensor) and inp.dtype in (torch.long, torch.int32):
                    self._current_mask_index = (inp == self.mask_id).cpu()
                    self._magnitude_accum = None   # will be initialised on first layer hook

        pre_h = model.register_forward_pre_hook(_pre_hook)
        self._hooks.append(pre_h)

        # ── Per-layer hooks: steer + capture magnitude ────────────────────────
        layers = _get_llada_layers(model)

        def _make_layer_hook(layer_idx: int):
            vec_cpu = self.vectors[layer_idx].float().cpu()   # [D]

            def hook(module, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                B, L, D = hs.shape

                # --- positional mask (same logic as parent) ---
                if (
                    self._current_mask_index is not None
                    and self._current_mask_index.shape[-1] == L
                ):
                    pos_mask = (
                        self._current_mask_index
                        .to(device=hs.device, dtype=hs.dtype)
                        .unsqueeze(-1)           # [B, L, 1]
                    )
                else:
                    pos_mask = torch.ones(B, L, 1, device=hs.device, dtype=hs.dtype)

                # --- apply steering ---
                vec = vec_cpu.to(device=hs.device, dtype=hs.dtype)   # [D]
                # delta[b, i, :] = alpha * v_l   if position i was [MASK], else 0
                delta = self.alpha * vec.view(1, 1, D) * pos_mask     # [B, L, D]
                hs = hs + delta

                # --- accumulate L2 norm of applied delta per position ---
                # delta_norm[b, i] = ||delta[b, i, :]||_2  (in float32, on CPU)
                delta_norm = delta.detach().float().norm(dim=-1).cpu()  # [B, L]
                if self._magnitude_accum is None:
                    self._magnitude_accum = delta_norm
                else:
                    self._magnitude_accum = self._magnitude_accum + delta_norm

                # Keep buffer updated so the last layer leaves the final sum
                self.magnitude_buffer.last_magnitudes = self._magnitude_accum

                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs

            return hook

        for ℓ in self.layer_ids:
            h = layers[ℓ].register_forward_hook(_make_layer_hook(ℓ))
            self._hooks.append(h)

        print(
            f"[MagnitudeCaptureSteering] Registered hooks on layers "
            f"{self.layer_ids} with α={self.alpha:.3f}  "
            f"(steers masked positions, captures magnitudes)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state capture extension for H2b SAE remasking
# ─────────────────────────────────────────────────────────────────────────────

class SteeringHiddenStateBuffer:
    """
    Shared mutable buffer written by HiddenStateCaptureSteering and read by
    H2SAEConfRemasking.

    Shape of `last_hidden_by_layer[layer]`:
        [B, L, D] float32, on CPU, containing post-steering hidden states at
        the monitored layer for the most recent forward pass.
    """

    def __init__(self):
        self.last_hidden_by_layer: dict[int, torch.Tensor] = {}

    def reset(self):
        self.last_hidden_by_layer = {}


class HiddenStateCaptureSteering(PrecomputedLayerSteering):
    """
    Extension of PrecomputedLayerSteering that captures post-steering hidden
    states at selected layers for SAE residual scoring.

    The steering behavior is identical to PrecomputedLayerSteering:
      hidden_state += alpha * v_layer at positions that were [MASK] in the
      current model input.

    Additionally, for each layer in `capture_layer_ids`, this class stores the
    post-steering hidden state tensor in a shared SteeringHiddenStateBuffer.
    """

    def __init__(
        self,
        vectors: Dict[int, torch.Tensor],
        layer_ids: List[int],
        alpha: float = 1.0,
        mask_id: int = LLADA_MASK_ID,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_state_buffer: Optional["SteeringHiddenStateBuffer"] = None,
    ):
        super().__init__(vectors=vectors, layer_ids=layer_ids, alpha=alpha, mask_id=mask_id)
        self.capture_layer_ids = set(capture_layer_ids or [])
        self.hidden_state_buffer: SteeringHiddenStateBuffer = (
            hidden_state_buffer
            if hidden_state_buffer is not None
            else SteeringHiddenStateBuffer()
        )

    def register_hooks(self, model: nn.Module) -> None:
        """Register hooks for steering and layerwise hidden-state capture."""
        self._current_mask_index = None
        self.hidden_state_buffer.reset()

        def _pre_hook(module, args):
            if isinstance(args, (tuple, list)) and len(args) > 0:
                inp = args[0]
                if isinstance(inp, torch.Tensor) and inp.dtype in (torch.long, torch.int32):
                    self._current_mask_index = (inp == self.mask_id).cpu()
                    self.hidden_state_buffer.reset()

        pre_h = model.register_forward_pre_hook(_pre_hook)
        self._hooks.append(pre_h)

        layers = _get_llada_layers(model)

        def _make_layer_hook(layer_idx: int):
            vec_cpu = self.vectors[layer_idx].float().cpu()  # [D]

            def hook(module, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                B, L, D = hs.shape

                if (
                    self._current_mask_index is not None
                    and self._current_mask_index.shape[-1] == L
                ):
                    pos_mask = (
                        self._current_mask_index
                        .to(device=hs.device, dtype=hs.dtype)
                        .unsqueeze(-1)
                    )
                else:
                    pos_mask = torch.ones(B, L, 1, device=hs.device, dtype=hs.dtype)

                vec = vec_cpu.to(device=hs.device, dtype=hs.dtype)
                delta = self.alpha * vec.view(1, 1, D) * pos_mask
                hs = hs + delta

                if layer_idx in self.capture_layer_ids:
                    self.hidden_state_buffer.last_hidden_by_layer[layer_idx] = (
                        hs.detach().float().cpu()
                    )

                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs

            return hook

        for ℓ in self.layer_ids:
            h = layers[ℓ].register_forward_hook(_make_layer_hook(ℓ))
            self._hooks.append(h)

        print(
            f"[HiddenStateCaptureSteering] Registered hooks on layers "
            f"{self.layer_ids} with α={self.alpha:.3f}; "
            f"capturing layers={sorted(self.capture_layer_ids)}"
        )


# ── Convenience helpers (used by mean_steering_test.py) ──────────────────────

def compute_steering_vectors(
    neg_act: Dict[int, torch.Tensor],
    pos_act: Dict[int, torch.Tensor],
    layer_ids: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Compute L2-normalised mean-diff steering vectors for `layer_ids`.

    Direction: mean(positive) - mean(negative) → adding alpha * v pushes
    the model toward the positive sentiment manifold.

    Args:
        neg_act:   {layer_idx: Tensor[N, D]}  negative-sentiment activations
        pos_act:   {layer_idx: Tensor[N, D]}  positive-sentiment activations
        layer_ids: which layers to compute vectors for

    Returns:
        {layer_idx: Tensor[D]}  unit-normalised direction vectors
    """
    vectors: Dict[int, torch.Tensor] = {}
    for ℓ in layer_ids:
        mean_neg = neg_act[ℓ].float().mean(dim=0)   # [D]
        mean_pos = pos_act[ℓ].float().mean(dim=0)   # [D]
        direction = mean_pos - mean_neg              # [D]
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm
        vectors[ℓ] = direction
    return vectors
