"""
ReMDM-conf remasking strategy.

Maintains a running *confidence score* for each generated token across
denoising steps and uses it to decide which tokens to remask.

When a token is first unmasked at step t, its confidence score is set to
the *negative* probability of the chosen token under the model (so lower
model confidence → higher score → more likely to be remasked later).
At each subsequent step, the confidence score is updated: if a token is
remasked again its score is reset to −∞; otherwise it stays.

The strategy then remasks the `num_to_remask` tokens with the *highest*
score — i.e. those that were unmasked with the least confidence.

This is the "ReMDM-conf" variant from Kuleshov-group / ReMDM, adapted for
the LLaDA API.

Reference
---------
Wang et al. (2025) – "Remasking Discrete Diffusion Models with
Inference-Time Scaling".  arXiv:2503.00307.
https://github.com/kuleshov-group/remdm  – diffusion.py, `_ddpm_caching_update`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseRemasking


class ReMDMConfRemasking(BaseRemasking):
    """
    Confidence-history based remasking (ReMDM-conf).

    Maintains a running ``conf_scores`` [B, L] tensor across denoising steps.
    This is semantically equivalent to ``conf_cache`` in the author's
    ``llada_remdm_sample`` (eval_llada.py), with an inverted sign convention:

    Correspondence with the author's ``conf_cache``
    ------------------------------------------------
    Author:  conf_cache = +inf  (not eligible / never unmasked)
             on unmask:  conf_cache[transfer_index] = softmax_prob  (positive)
             on remask:  conf_cache[remask_index]   = +inf  (reset)
             selection:  topk(largest=False) → smallest = least confident

    Here:    conf_scores = -inf  (not eligible / never unmasked)
             on unmask:  conf_scores[newly_unmasked] = -prob  (negative)
             on remask:  conf_scores[remask]         = -inf  (reset)
             selection:  topk(largest=True) → highest of negatives = least confident

    Both select the same positions.  The ``-inf`` sentinel is used instead of
    ``+inf`` to stay consistent with other components in this codebase.

    Pair with ``fill_strategy="low_confidence"`` and schedule controls
    ``remask_fixed_count`` / ``remask_start_frac`` in ``generate()`` for a
    configuration close to the author's ``llada_remdm_sample``.

    Usage
    -----
    Instantiate once per run.  Call ``reset()`` before each new prompt so
    per-sequence confidence history does not bleed across prompts.
    """

    def __init__(self):
        self.conf_scores: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear stored confidence scores (call before each new generation)."""
        self.conf_scores = None

    def select_tokens_to_remask(
        self,
        x_prev: torch.LongTensor,
        x_curr: torch.LongTensor,
        logits: torch.FloatTensor,
        x0: torch.LongTensor,
        mask_id: int,
        num_to_remask: int,
        prompt_len: int,
    ) -> torch.BoolTensor:
        B, L = x_curr.shape
        device = x_curr.device

        if self.conf_scores is None or self.conf_scores.shape != (B, L):
            self.conf_scores = torch.full(
                (B, L), float("-inf"), device=device, dtype=torch.float32
            )

        probs = F.softmax(logits.float(), dim=-1)  # [B, L, V]
        x0_prob = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]

        # Positions that were just unmasked this step
        newly_unmasked = (x_prev == mask_id) & (x_curr != mask_id)
        self.conf_scores[newly_unmasked] = (-x0_prob)[newly_unmasked]

        # Prompt tokens are never eligible
        self.conf_scores[:, :prompt_len] = float("-inf")

        remask = torch.zeros(B, L, dtype=torch.bool, device=device)
        if num_to_remask <= 0:
            return remask

        # Only currently unmasked generation tokens are eligible
        eligible = (x_curr != mask_id)
        eligible[:, :prompt_len] = False

        scores = self.conf_scores.clone()
        scores[~eligible] = float("-inf")

        for b in range(B):
            k = min(num_to_remask, int(eligible[b].sum().item()))
            if k <= 0:
                continue
            _, idx = torch.topk(scores[b], k=k, largest=True)
            remask[b, idx] = True

        # Clear history for tokens we remask now
        self.conf_scores[remask] = float("-inf")

        return remask
