"""
Transformers compatibility shims for the upstream ``LLaDA-V/train/llava`` package.

Why
---
The cloned LLaDA-V repo bundles a copy of LLaVA's source tree which still
imports several private symbols from ``transformers.modeling_utils`` that
were removed when ``transformers`` 5.x reorganised them. The relevant ones:

  * ``apply_chunking_to_forward``      — moved to ``transformers.pytorch_utils``
  * ``prune_linear_layer``             — moved to ``transformers.pytorch_utils``
  * ``find_pruneable_heads_and_indices`` — removed; we install a stub.

These imports happen at module-load time of
``llava.model.multimodal_resampler.qformer``. ``Qformer`` itself is *not*
instantiated for ``GSAI-ML/LLaDA-V`` (the config sets ``mm_resampler_type=None``),
so the stub's only job is to satisfy the import statement; if any code path
ever calls the stub it will raise loudly.

Call ``ensure_compat()`` once before importing anything from ``llava.*``.
"""

from __future__ import annotations


_PATCH_APPLIED = False


def ensure_compat() -> None:
    """Install missing-symbol shims into ``transformers.modeling_utils``."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu

    if not hasattr(mu, "apply_chunking_to_forward"):
        mu.apply_chunking_to_forward = pu.apply_chunking_to_forward
    if not hasattr(mu, "prune_linear_layer"):
        mu.prune_linear_layer = pu.prune_linear_layer
    if not hasattr(mu, "find_pruneable_heads_and_indices"):
        def _stub_find_pruneable_heads_and_indices(*args, **kwargs):
            raise RuntimeError(
                "find_pruneable_heads_and_indices was removed in transformers 5.x. "
                "It is only required for Qformer code paths in llava, which "
                "GSAI-ML/LLaDA-V never uses (mm_resampler_type=None)."
            )
        mu.find_pruneable_heads_and_indices = _stub_find_pruneable_heads_and_indices

    _PATCH_APPLIED = True
