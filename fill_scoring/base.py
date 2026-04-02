"""
Abstract interface for fill-stage scoring.

A fill scorer takes baseline fill scores for currently masked tokens and may
return adjusted scores used by the commit/unmask top-k selection step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseFillScorer(ABC):
    """Interface for modular fill-stage score adjustments."""

    @abstractmethod
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
        """
        Return scores used by top-k commit selection. Higher is better.

        Returned tensor shape must be [B, L].
        """
        ...

    def record_selection(self, block_idx: int, step_i: int, batch_idx: int, selected_pos: torch.LongTensor) -> None:
        """Optional hook called by generate() after top-k commit positions are chosen."""
        return
