"""
Abstract base class for steering strategies.

Steering modifies LLaDA's generation process at each denoising step by
intervening on the model's internal representations (hidden states).
Each subclass implements a different steering strategy (e.g. mean
activation steering, classifier guidance, …).
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseSteering(ABC):
    """
    Base class for all steering strategies.

    A steering object is passed to `llada/generate.py`, which calls
    `register_hooks` before the forward pass and `remove_hooks` after it.
    The hooks modify hidden states in-place, biasing generation away from
    undesired content (e.g. toxicity).
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Steering strength. Larger values = stronger intervention.
        """
        self.alpha = alpha
        self._hooks: list = []

    @abstractmethod
    def register_hooks(self, model: torch.nn.Module) -> None:
        """
        Attach forward hooks to `model` that will modify hidden states.
        Hooks should be appended to `self._hooks` so they can be removed later.
        """
        ...

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()
