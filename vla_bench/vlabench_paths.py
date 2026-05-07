"""
Lightweight VLABench wiring.

The official VLABench imports (`VLABench.evaluation.evaluator.VLMEvaluator`) drag
in `mujoco`, `dm_control`, and the simulation stack via the base
``Evaluator`` class. For a *VLM-only* sanity check we don't need any of that —
we only need:

  * the official prompt file (``configs/prompt/eval_vlm_en.txt``)
  * the official chat-template builder (``get_ti_list`` in ``model.vlm.base``)
  * the official Qwen2-VL wrapper (``model.vlm.qwen_vl.Qwen2_VL``)
  * the official scoring routine (``evaluation.utils.get_final_score``)
  * the official ``seq_independent_task.json`` (drives Sequential vs
    Seq-independent dependency for scoring)

This module exposes the minimum side-effects to make those imports work:
  * adds the cloned ``VLABench`` repo to ``sys.path``;
  * sets ``VLABENCH_ROOT`` to the package directory.

If you ever need the full simulation stack for policy evaluation, switch to a
real install (``pip install -e .`` from the cloned repo) — but keep this module
in place because the VLM-only path is still the cheapest way to import the
relevant utilities.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VLABENCH_REPO_DIR = REPO_ROOT / "vla_bench" / "VLABench"
VLABENCH_PACKAGE_DIR = VLABENCH_REPO_DIR / "VLABench"
VLABENCH_DATASET_ROOT = REPO_ROOT / "vla_bench" / "data" / "vlm_evaluation_v1.0"


# Sub-packages we want Python to treat as already-imported namespace packages,
# so that pulling in e.g. ``VLABench.evaluation.utils`` does not transitively
# execute ``VLABench/evaluation/__init__.py``. The official __init__ does
# ``from VLABench.evaluation.evaluator import *``, which imports
# ``evaluator/base.py`` and pulls in mujoco / dm_control / mediapy — none of
# which we need for VLM-only evaluation.
_NAMESPACE_PACKAGES_TO_FAKE = (
    "VLABench",
    "VLABench.evaluation",
    "VLABench.evaluation.model",
    "VLABench.evaluation.model.vlm",
    "VLABench.evaluation.evaluator",
)


def _register_namespace_package(name: str, sub_path: str) -> None:
    """Pre-register ``name`` in ``sys.modules`` as an empty namespace package
    rooted at ``VLABENCH_PACKAGE_DIR/sub_path``. This bypasses the upstream
    ``__init__.py`` of that package so deeper modules can be imported in
    isolation.
    """
    if name in sys.modules:
        return
    pkg_path = VLABENCH_PACKAGE_DIR / sub_path
    if not pkg_path.exists():
        raise FileNotFoundError(
            f"Expected VLABench sub-package at {pkg_path} (for namespace fake of {name})"
        )
    mod = types.ModuleType(name)
    mod.__path__ = [str(pkg_path)]
    mod.__file__ = None
    sys.modules[name] = mod


def setup() -> None:
    """Make ``import VLABench.evaluation.model.vlm.base`` and similar work
    without installing the package or triggering its simulation imports.

    Preconditions:
      * ``VLABench`` cloned at ``vla_bench/VLABench`` (one-time).
    """
    if not VLABENCH_PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"VLABench clone not found at {VLABENCH_REPO_DIR}. "
            "Run: git clone https://github.com/OpenMOSS/VLABench "
            f"{VLABENCH_REPO_DIR}"
        )

    if str(VLABENCH_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(VLABENCH_REPO_DIR))

    os.environ.setdefault("VLABENCH_ROOT", str(VLABENCH_PACKAGE_DIR))

    # Map each fake namespace package to its on-disk subdirectory.
    name_to_sub = {
        "VLABench": "",
        "VLABench.evaluation": "evaluation",
        "VLABench.evaluation.model": "evaluation/model",
        "VLABench.evaluation.model.vlm": "evaluation/model/vlm",
        "VLABench.evaluation.evaluator": "evaluation/evaluator",
    }
    for name in _NAMESPACE_PACKAGES_TO_FAKE:
        _register_namespace_package(name, name_to_sub[name])


def dataset_dim_dir(eval_dim: str) -> Path:
    """
    Map a logical evaluation dimension (as named in the VLABench config) to the
    actual on-disk folder name in the HuggingFace dataset release.

    The HF dataset uses ``CommenSence/`` (typo) instead of ``CommonSense/``.
    """
    mapping = {
        "CommonSense": "CommenSence",
    }
    folder = mapping.get(eval_dim, eval_dim)
    return VLABENCH_DATASET_ROOT / folder


# Tasks listed in the user's sanity grid → (eval_dim, on-disk folder).
SANITY_TASKS: dict[str, str] = {
    "select_fruit": "M&T",
    "select_toy": "M&T",
    "select_fruit_common_sense": "CommonSense",
}
