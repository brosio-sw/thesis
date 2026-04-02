"""
Mean Activation Steering  — Steering Method S1.

Implements representation-engineering style steering (Turner et al. 2023,
"Activation Addition"; Zou et al. 2023, "Representation Engineering").

Idea
----
1.  Collect two contrastive sets of texts:
      - `positive` (toxic / high-toxicity continuations)
      - `negative` (non-toxic / neutral continuations)
2.  Pass both through the frozen LLaDA backbone and record the per-layer
    hidden states averaged over sequence positions and examples.
3.  The *steering vector* for layer ℓ is:
        v_ℓ = mean_positive_ℓ - mean_negative_ℓ
4.  During generation, a forward hook subtracts  α · v_ℓ  from the hidden
    states of layer ℓ at every denoising step.  Subtracting the toxic
    direction pushes the model toward less toxic representations.

Reference
---------
Turner et al. (2023) – "Activation Addition: Steering Language Models
Without Optimization".  https://arxiv.org/abs/2308.10248

Zou et al. (2023) – "Representation Engineering: A Top-Down Approach to
AI Transparency".  https://arxiv.org/abs/2310.01405
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from .base import BaseSteering


class MeanActivationSteering(BaseSteering):
    """
    Compute a per-layer steering vector as the mean-activation difference
    between *positive* (toxic) and *negative* (non-toxic) text sets, then
    hook into the model during generation to subtract  α · v_ℓ.

    Parameters
    ----------
    layer_ids:
        Which transformer block indices to steer.  If None, steer ALL
        hidden layers.
    alpha:
        Steering strength (subtracted coefficient).
    token_average:
        If True (default), average hidden states over sequence positions.
        If False, use only the [MASK] token positions.
    """

    def __init__(
        self,
        layer_ids: Optional[List[int]] = None,
        alpha: float = 15.0,
        token_average: bool = True,
    ):
        super().__init__(alpha=alpha)
        self.layer_ids = layer_ids          # set after build_steering_vector
        self.token_average = token_average
        # dict  layer_idx → steering vector  (shape: [hidden_dim])
        self.vectors: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Vector construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_steering_vector(
        self,
        model: nn.Module,
        tokenizer,
        positive_texts: List[str],
        negative_texts: List[str],
        batch_size: int = 8,
        layer_ids: Optional[List[int]] = None,
        cache_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Compute and store per-layer steering vectors from contrastive texts.

        All texts are mean-pooled (over non-padding positions) to produce
        one vector per text per layer; the steering vector is then the
        difference of their means.

        Args:
            model:          The LLaDA model (or any HuggingFace transformer
                            with `.config.num_hidden_layers`).
            tokenizer:      Matching tokenizer.
            positive_texts: Texts representing the direction to *steer away
                            from* (e.g. toxic sentences).
            negative_texts: Texts representing the *target* direction
                            (e.g. non-toxic sentences).
            batch_size:     Number of texts to process per forward pass.
            layer_ids:      Override `self.layer_ids`.
            cache_path:     If given, save / load computed vectors here so
                            the expensive forward pass only runs once.
            device:         Torch device string.  Defaults to model's device.
        """
        if cache_path is not None and Path(cache_path).exists():
            self.vectors = torch.load(cache_path, map_location="cpu")
            # infer layer_ids from stored vectors
            self.layer_ids = sorted(self.vectors.keys())
            print(f"[MeanSteering] Loaded steering vectors from {cache_path}")
            return

        device = device or next(model.parameters()).device
        n_layers = model.config.num_hidden_layers
        if layer_ids is not None:
            self.layer_ids = layer_ids
        elif self.layer_ids is None:
            self.layer_ids = list(range(n_layers))

        def _collect_means(texts: List[str]) -> dict[int, torch.Tensor]:
            """
            Returns {layer_idx: mean_hidden_state}  shape [hidden_dim] each.
            """
            # accumulators
            sums = {ℓ: torch.zeros(model.config.hidden_size) for ℓ in self.layer_ids}
            counts = {ℓ: 0 for ℓ in self.layer_ids}

            captured: dict[int, torch.Tensor] = {}

            def _make_hook(layer_idx: int):
                def hook(module, input, output):
                    # output is usually a tuple; first element = hidden state
                    hs = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = hs.detach().float().cpu()
                return hook

            # Register hooks for desired layers
            hooks = []
            for ℓ in self.layer_ids:
                h = model.encoder.layer[ℓ].register_forward_hook(_make_hook(ℓ))
                hooks.append(h)

            model.eval()
            for start in tqdm(
                range(0, len(texts), batch_size),
                desc=f"[MeanSteering] Collecting activations",
                leave=False,
            ):
                batch = texts[start : start + batch_size]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(device)

                with torch.no_grad():
                    _ = model(**enc, output_hidden_states=False)

                attn = enc["attention_mask"].cpu().float()  # [B, L]
                for ℓ, hs in captured.items():
                    # hs: [B, L, D]
                    mask = attn.unsqueeze(-1)          # [B, L, 1]
                    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    sums[ℓ] += pooled.sum(dim=0)
                    counts[ℓ] += pooled.shape[0]
                captured.clear()

            for h in hooks:
                h.remove()

            return {ℓ: sums[ℓ] / counts[ℓ] for ℓ in self.layer_ids}

        print("[MeanSteering] Computing mean activations for positive (toxic) set …")
        pos_means = _collect_means(positive_texts)
        print("[MeanSteering] Computing mean activations for negative (safe) set …")
        neg_means = _collect_means(negative_texts)

        self.vectors = {
            ℓ: (pos_means[ℓ] - neg_means[ℓ]).to(device)
            for ℓ in self.layer_ids
        }

        if cache_path is not None:
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            torch.save(
                {k: v.cpu() for k, v in self.vectors.items()},
                cache_path,
            )
            print(f"[MeanSteering] Saved steering vectors to {cache_path}")

    # ------------------------------------------------------------------
    # Hook registration (called inside llada/generate.py)
    # ------------------------------------------------------------------

    def register_hooks(self, model: nn.Module) -> None:
        """
        Attach forward hooks that subtract  α · v_ℓ  from the hidden states
        of each targeted transformer layer.

        Must be called *after* `build_steering_vector`.
        """
        if not self.vectors:
            raise RuntimeError(
                "Call build_steering_vector() before register_hooks()."
            )

        device = next(model.parameters()).device

        def _make_hook(layer_idx: int):
            vec = self.vectors[layer_idx].to(device)  # [D]

            def hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                # Subtract the toxic direction  → steers toward non-toxic
                hs = hs - self.alpha * vec.unsqueeze(0).unsqueeze(0)
                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs

            return hook

        for ℓ in self.layer_ids:
            h = model.encoder.layer[ℓ].register_forward_hook(_make_hook(ℓ))
            self._hooks.append(h)

        print(
            f"[MeanSteering] Registered hooks on layers {self.layer_ids} "
            f"with α={self.alpha}"
        )
