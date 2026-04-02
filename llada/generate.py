"""
Generation utilities for LLaDA-style masked diffusion decoding.

This module keeps two concerns separate:

1. Commit / fill policy
   Decide which *currently masked* positions to commit at the current step.
   - ``fill_strategy='low_confidence'``: standard LLaDA behavior. In practice
     this commits the **highest-confidence** masked tokens first.
   - ``fill_strategy='random'``: random commit baseline.
   - ``fill_scorer`` can optionally modify or replace those baseline commit
     scores before the top-k commit step.

2. Optional remasking / revision
   Reopen already committed generation tokens by setting them back to
   ``[MASK]``. This is disabled by default. If provided, ``remasking`` only
   needs to expose a ``select_tokens_to_remask(...)`` method; no formal base
   remasking class is required.

Recommended usage
-----------------
Plain base LLaDA:
    generate(..., fill_strategy="low_confidence", remasking=None)

Random commit baseline:
    generate(..., fill_strategy="random", remasking=None)

Commit-side heuristic (e.g. H2):
    generate(..., fill_strategy="low_confidence", fill_scorer=h2_scorer,
             remasking=None)

Optional ReMDM-style revision:
    generate(..., fill_strategy="low_confidence",
             remasking=ReMDMConfRemasking(),
             remask_fixed_count=<k>, remask_start_frac=0.875)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:  # package usage: from llada.generate import generate
    from .fill_scoring.base import BaseFillScorer
    from .steering.base import BaseSteering
except ImportError:  # script / flat-layout fallback
    from fill_scoring.base import BaseFillScorer
    from steering.base import BaseSteering

MASK_ID = 126336
DEFAULT_FILL_STRATEGY = "low_confidence"


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply the Gumbel-max perturbation used by LLaDA-style decoding.

    When ``temperature == 0``, decoding is deterministic argmax.
    """
    if temperature == 0.0:
        return logits

    logits = logits.float()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise



def _get_num_transfer_tokens(
    mask_index: torch.BoolTensor,
    steps: int,
) -> torch.LongTensor:
    """
    Compute how many masked positions to commit at each denoising step.

    The total number of masked tokens is distributed as evenly as possible
    across the available steps, matching the standard LLaDA schedule.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer = torch.zeros(
        mask_num.size(0),
        steps,
        device=mask_index.device,
        dtype=torch.long,
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1

    return num_transfer



def _build_default_fill_scores(
    *,
    logits: torch.FloatTensor,
    x0: torch.LongTensor,
    fill_strategy: str,
    device: torch.device,
) -> torch.FloatTensor:
    """
    Build baseline commit scores for candidate tokens.

    Higher scores mean the position is more likely to be committed this step.
    """
    if fill_strategy == "low_confidence":
        probs = F.softmax(logits.float(), dim=-1)
        return probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    if fill_strategy == "random":
        return torch.rand(x0.shape, device=device)

    raise ValueError(
        f"Unknown fill_strategy {fill_strategy!r}. "
        "Expected 'low_confidence' or 'random'."
    )


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    fill_strategy: str = DEFAULT_FILL_STRATEGY,
    fill_scorer: Optional[BaseFillScorer] = None,
    remasking: Optional[Any] = None,
    remask_fraction: float = 0.0,
    remask_fixed_count: Optional[int] = None,
    remask_start_frac: float = 0.0,
    mask_id: int = MASK_ID,
    steering: Optional[BaseSteering] = None,
    show_progress: bool = True,
    show_remask_stats: bool = False,
) -> torch.LongTensor:
    """
    Generate tokens with LLaDA-style masked diffusion decoding.

    Args:
        fill_strategy:
            Baseline commit policy over currently masked positions.
            ``'low_confidence'`` is standard LLaDA and commits the highest-
            confidence masked tokens first.
        fill_scorer:
            Optional object with ``build_fill_scores(...)`` and optional
            ``record_selection(...)`` / ``reset()`` hooks. Use this for H2 or
            other commit-side heuristics.
        remasking:
            Optional object with ``select_tokens_to_remask(...)`` and optional
            ``reset()``. Leave as ``None`` for the plain baseline.
    """
    device = prompt.device
    batch_size, prompt_len = prompt.shape

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of blocks")
    steps_per_block = steps // num_blocks

    if fill_scorer is not None and hasattr(fill_scorer, "reset"):
        fill_scorer.reset()
    if remasking is not None and hasattr(remasking, "reset"):
        remasking.reset()

    x = torch.full(
        (batch_size, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, gen_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id

    if steering is not None:
        steering.register_hooks(model)

    try:
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length

            block_mask_index = x[:, block_start:block_end] == mask_id
            num_transfer = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            step_iter = range(steps_per_block)
            if show_progress:
                step_iter = tqdm(
                    step_iter,
                    desc=f"Block {block_idx + 1}/{num_blocks}",
                    leave=False,
                )

            for step_i in step_iter:
                mask_index = x == mask_id

                # 1) Forward pass (optionally CFG)
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_in = torch.cat([x, un_x], dim=0)
                    attn_in = (
                        torch.cat([attention_mask, attention_mask], dim=0)
                        if attention_mask is not None
                        else None
                    )
                    logits = model(x_in, attention_mask=attn_in).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits

                logits = logits.to(device)

                # 2) Predict candidate x0 at every position
                logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
                x0 = logits_noisy.argmax(dim=-1)

                # Keep prompt / already-unmasked tokens fixed
                x0 = torch.where(mask_index, x0, x)

                # 3) Build commit scores over currently masked positions
                baseline_scores = _build_default_fill_scores(
                    logits=logits,
                    x0=x0,
                    fill_strategy=fill_strategy,
                    device=device,
                )
                baseline_scores[:, block_end:] = float("-inf")
                baseline_scores = torch.where(
                    mask_index,
                    baseline_scores,
                    torch.full_like(baseline_scores, float("-inf")),
                )

                if fill_scorer is not None:
                    commit_scores = fill_scorer.build_fill_scores(
                        x_curr=x,
                        logits=logits,
                        x0=x0,
                        baseline_scores=baseline_scores,
                        mask_index=mask_index,
                        prompt_len=prompt_len,
                        block_start=block_start,
                        block_end=block_end,
                        block_idx=block_idx,
                        step_i=step_i,
                    )
                else:
                    commit_scores = baseline_scores

                # 4) Commit top-k masked positions for this step
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for b in range(batch_size):
                    k = int(num_transfer[b, step_i].item())
                    if k <= 0:
                        continue

                    _, selected_pos = torch.topk(commit_scores[b], k=k)
                    transfer_index[b, selected_pos] = True

                    if fill_scorer is not None and hasattr(fill_scorer, "record_selection"):
                        fill_scorer.record_selection(
                            block_idx=block_idx,
                            step_i=step_i,
                            batch_idx=b,
                            selected_pos=selected_pos,
                        )

                x_prev = x.clone()
                x[transfer_index] = x0[transfer_index]

                # 5) Optional remasking / revision stage
                if remasking is None:
                    continue

                step_frac = step_i / max(steps_per_block - 1, 1)
                if step_frac < remask_start_frac:
                    continue

                for b in range(batch_size):
                    if remask_fixed_count is not None:
                        num_to_remask = remask_fixed_count
                    elif remask_fraction > 0.0:
                        gen_unmasked = ((x[b] != mask_id) & ~prompt_index[b]).sum().item()
                        num_to_remask = max(1, int(gen_unmasked * remask_fraction))
                    else:
                        num_to_remask = 0

                    if num_to_remask <= 0:
                        continue

                    remask_bool = remasking.select_tokens_to_remask(
                        x_prev=x_prev[b : b + 1],
                        x_curr=x[b : b + 1],
                        logits=logits[b : b + 1],
                        x0=x0[b : b + 1],
                        mask_id=mask_id,
                        num_to_remask=num_to_remask,
                        prompt_len=prompt_len,
                    )

                    if show_remask_stats and remask_bool[0].any():
                        n = int(remask_bool[0].sum().item())
                        print(
                            f"  [remask] block={block_idx} "
                            f"step={step_i:02d} b={b}: {n} tokens re-masked"
                        )

                    x[b, remask_bool[0]] = mask_id

    finally:
        if steering is not None:
            steering.remove_hooks()

    return x