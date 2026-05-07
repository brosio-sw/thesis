"""
Prompt builders for the VLABench VLM sanity grid.

We deliberately re-use the *official* VLABench primitives:

  * the prompt text from ``configs/prompt/eval_vlm_<lang>.txt``,
  * the official ``get_ti_list`` chat-template assembler from
    ``VLABench.evaluation.model.vlm.base``.

The only thing we add is an optional ``image_mode`` filter applied to the
ti_list *after* the official builder has produced it. See
``vla_bench/image_utils.py``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from vla_bench import vlabench_paths

vlabench_paths.setup()

from VLABench.evaluation.model.vlm.base import get_ti_list  # noqa: E402

from vla_bench.image_utils import filter_ti_list_by_image_mode  # noqa: E402


def load_pre_prompt(language: str = "en") -> str:
    """Read the official VLABench VLM prompt from its config file."""
    cfg = (
        vlabench_paths.VLABENCH_PACKAGE_DIR
        / "configs"
        / "prompt"
        / f"eval_vlm_{language}.txt"
    )
    return cfg.read_text(encoding="utf-8")


def load_seq_independent_tasks() -> set[str]:
    cfg = (
        vlabench_paths.VLABENCH_PACKAGE_DIR
        / "configs"
        / "evaluation"
        / "seq_independent_task.json"
    )
    return set(json.loads(cfg.read_text(encoding="utf-8")))


def _example_dir(data_dim_root: Path, task_name: str, example_num: int) -> Path:
    return data_dim_root / task_name / f"example{example_num}"


def _load_single_input(
    data_dim_root: Path, task_name: str, example_num: int
) -> tuple[str, str, str]:
    """Mirror VLMEvaluator.load_single_input — returns paths and instruction."""
    example_dir = _example_dir(data_dim_root, task_name, example_num)
    input_pic = str(example_dir / "input" / "input.png")
    input_pic_gt = str(example_dir / "input" / "input_mask.png")
    instruction = (
        (example_dir / "input" / "instruction.txt")
        .read_text(encoding="utf-8")
    )
    return input_pic, input_pic_gt, instruction


def _load_single_output(
    data_dim_root: Path, task_name: str, example_num: int
) -> dict[str, Any]:
    example_dir = _example_dir(data_dim_root, task_name, example_num)
    return json.loads(
        (example_dir / "output" / "operation_sequence.json").read_text(encoding="utf-8")
    )


def list_examples(data_dim_root: Path, task_name: str) -> list[int]:
    """Return sorted example indices available on disk for ``task_name``."""
    task_dir = data_dim_root / task_name
    out = []
    for entry in os.listdir(task_dir):
        if not entry.startswith("example"):
            continue
        try:
            out.append(int(entry[len("example"):]))
        except ValueError:
            continue
    return sorted(out)


def build_input(
    *,
    data_dim_root: Path,
    task_name: str,
    example_num: int,
    pre_prompt: str,
    few_shot_num: int,
    rng: random.Random,
    available_shot_tasks: list[str],
    available_shots_per_task: dict[str, list[int]],
) -> dict[str, Any]:
    """
    Mirror ``VLMEvaluator.build_input`` exactly, but parameterised so we don't
    have to subclass the heavy ``Evaluator`` base (which pulls in mujoco /
    dm_control via ``VLABench.envs``).

    Few-shot examples are sampled from ``available_shot_tasks`` (tasks present
    on disk in the same eval dim); the random source is `rng` so seeds can be
    fixed externally.
    """
    prepared: dict[str, Any] = {"pre_prompt": pre_prompt}

    if few_shot_num != 0:
        prepared["shot_input_pic"] = {}
        prepared["shot_input_pic_gt"] = {}
        prepared["shot_input_instruction"] = {}
        prepared["shot_output"] = {}

    for i in range(few_shot_num):
        shot_task_name = rng.choice(available_shot_tasks)
        shot_example_num = rng.choice(available_shots_per_task[shot_task_name])
        # Reroll if we picked the very test example we're scoring.
        while shot_task_name == task_name and shot_example_num == example_num:
            shot_task_name = rng.choice(available_shot_tasks)
            shot_example_num = rng.choice(available_shots_per_task[shot_task_name])

        shot_input_pic, shot_input_pic_gt, shot_instruction = _load_single_input(
            data_dim_root, shot_task_name, shot_example_num
        )
        shot_output = _load_single_output(data_dim_root, shot_task_name, shot_example_num)

        prepared["shot_input_pic"][str(i)] = shot_input_pic
        prepared["shot_input_pic_gt"][str(i)] = shot_input_pic_gt
        prepared["shot_input_instruction"][str(i)] = shot_instruction
        prepared["shot_output"][str(i)] = shot_output

    input_pic, input_pic_gt, instruction = _load_single_input(
        data_dim_root, task_name, example_num
    )
    prepared["input_pic"] = input_pic
    prepared["input_pic_gt"] = input_pic_gt
    prepared["input_instruction"] = instruction
    return prepared


def build_ti_list(
    *,
    input_dict: dict[str, Any],
    language: str,
    with_cot: bool,
    image_mode: str,
) -> list[list[str]]:
    """
    Build the official ti_list and apply the `image_mode` filter on top.
    """
    base_ti_list = get_ti_list(input_dict, language=language, with_CoT=with_cot)
    return filter_ti_list_by_image_mode(base_ti_list, image_mode=image_mode)


def gt_skill_sequence(
    *,
    data_dim_root: Path,
    task_name: str,
    example_num: int,
) -> list[dict[str, Any]]:
    out = _load_single_output(data_dim_root, task_name, example_num)
    return out["skill_sequence"]
