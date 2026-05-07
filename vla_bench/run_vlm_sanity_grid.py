"""
VLABench VLM sanity grid for Qwen2-VL.

Runs an isolated sanity check that exercises the *official* VLABench VLM
evaluation pipeline (prompt + chat-template + JSON parsing + skill-sequence
scoring) on three easy tasks:

  * select_fruit                       (M&T)
  * select_toy                         (M&T)
  * select_fruit_common_sense          (CommenSence/CommonSense)

Grid (full):
    3 tasks × 25 episodes × {CoT off, CoT on} × {original_plus_numbered,
    numbered_only}  ==  300 examples

Smoke (SMOKE_TEST=True):
    3 tasks × 2 episodes × {CoT off, CoT on} × {original_plus_numbered,
    numbered_only}  ==  24 examples

For each example we save a JSONL row with full diagnostics:
prompt construction, raw output, parser status, scoring breakdown, etc.

The scoring path is the *official* ``VLABench.evaluation.utils.get_final_score``,
so reported numbers are directly comparable to the published VLABench numbers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vla_bench import vlabench_paths

vlabench_paths.setup()

from VLABench.evaluation.utils import get_final_score  # noqa: E402

from vla_bench.image_utils import (  # noqa: E402
    collect_image_paths,
    humanize_image_paths,
)
from vla_bench.parsing import summarize_output  # noqa: E402
from vla_bench.prompts import (  # noqa: E402
    build_input,
    build_ti_list,
    gt_skill_sequence,
    list_examples,
    load_pre_prompt,
    load_seq_independent_tasks,
)


SMOKE_TEST = True

DEFAULT_TASKS = ("select_fruit", "select_toy", "select_fruit_common_sense")
DEFAULT_FEW_SHOT_NUM = 1
DEFAULT_LANGUAGE = "en"
DEFAULT_COT_VALUES = (False, True)
DEFAULT_IMAGE_MODES = ("original_plus_numbered", "numbered_only")

DEFAULT_FULL_EPISODES = 25
DEFAULT_SMOKE_EPISODES = 2

DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "vla_bench" / "qwen2vl_sanity"

DEFAULT_SEED = 42

# --- Per-model defaults --------------------------------------------------
# ``--model qwen2vl`` (default): the existing Qwen2-VL sanity grid.
# ``--model llada_v``           : the LLaDA-V baseline. Outputs land in a
#                                 separate directory and the default model
#                                 name + decoding kwargs change.
MODEL_KEY_QWEN2VL = "qwen2vl"
MODEL_KEY_LLADA_V = "llada_v"

MODEL_DEFAULTS = {
    MODEL_KEY_QWEN2VL: {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "out_subdir": "qwen2vl_sanity",
    },
    MODEL_KEY_LLADA_V: {
        "model_name": "GSAI-ML/LLaDA-V",
        "out_subdir": "llada_v_sanity",
    },
}


def cot_tag(cot: bool) -> str:
    return "cot_on" if cot else "cot_off"


def image_mode_tag(image_mode: str) -> str:
    return image_mode  # already short and slug-friendly


def evaluate_single_example(
    *,
    vlm,
    data_dim_root: Path,
    task_name: str,
    example_num: int,
    pre_prompt: str,
    few_shot_num: int,
    language: str,
    with_cot: bool,
    image_mode: str,
    rng: random.Random,
    available_shot_tasks: list[str],
    available_shots_per_task: dict[str, list[int]],
    seq_independent_tasks: set[str],
    dataset_root_str: str,
) -> dict[str, Any]:
    """Run one (task, episode, cot, image_mode) example and return a record."""
    record: dict[str, Any] = {
        "task": task_name,
        "episode_id": int(example_num),
        "few_shot": int(few_shot_num),
        "cot": bool(with_cot),
        "image_mode": image_mode,
        "language": language,
    }

    try:
        input_dict = build_input(
            data_dim_root=data_dim_root,
            task_name=task_name,
            example_num=example_num,
            pre_prompt=pre_prompt,
            few_shot_num=few_shot_num,
            rng=rng,
            available_shot_tasks=available_shot_tasks,
            available_shots_per_task=available_shots_per_task,
        )
    except Exception as e:
        record.update(
            prompt=None,
            raw_output=None,
            parsed_output=None,
            parse_success=False,
            score=None,
            success=False,
            error_type="build_input_error",
            error_message=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )
        return record

    try:
        ti_list = build_ti_list(
            input_dict=input_dict,
            language=language,
            with_cot=with_cot,
            image_mode=image_mode,
        )
    except Exception as e:
        record.update(
            prompt=None,
            raw_output=None,
            parsed_output=None,
            parse_success=False,
            score=None,
            success=False,
            error_type="ti_list_error",
            error_message=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )
        return record

    image_paths = collect_image_paths(ti_list)
    image_paths_short = humanize_image_paths(image_paths, dataset_root=dataset_root_str)

    record["instruction"] = input_dict["input_instruction"]
    record["image_paths"] = image_paths_short
    record["n_images_passed"] = len(image_paths)
    # The "prompt" we save is the textual side of the chat template
    # (image entries become "[image:...]" placeholders). This is what we
    # actually feed to the chat template; the model itself sees this text plus
    # the attached images.
    record["prompt"] = "\n".join(
        [
            (f"[image:{value}]" if kind == "image" else str(value))
            for kind, value in ti_list
        ]
    )

    output: dict[str, Any] | str | None = None
    crash_info: dict[str, Any] | None = None
    try:
        output = vlm.evaluate_with_ti_list(ti_list, with_CoT=with_cot)
    except Exception as e:
        crash_info = {
            "error_type": "model_crash",
            "error_message": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    if crash_info is not None:
        record.update(
            raw_output=None,
            parsed_output=None,
            parse_success=False,
            score=None,
            success=False,
        )
        record.update(crash_info)
        return record

    parse = summarize_output(output)
    record.update(parse)

    # Scoring: route through the *official* scorer in
    # VLABench.evaluation.utils.get_final_score.
    if not parse["parse_success"]:
        record["score"] = {
            "skill_match_score": 0,
            "entity_match_score": 0,
            "skill_with_entity_match_score": 0,
            "exact_match_score": 0,
            "total_score": 0,
        }
        record["success"] = False
        record.setdefault("error_type", "format_error")
        return record

    try:
        gt = gt_skill_sequence(
            data_dim_root=data_dim_root,
            task_name=task_name,
            example_num=example_num,
        )
        dependency = (
            "Sequential" if task_name not in seq_independent_tasks else "Seq-independent"
        )
        score = get_final_score(
            gt,
            parse["parsed_output"]["skill_sequence"],
            dependency=dependency,
        )
        record["score"] = score
        record["success"] = bool(score.get("exact_match_score", 0.0) >= 100.0 - 1e-9)
        record["error_type"] = None
        record["error_message"] = None
    except Exception as e:
        record["score"] = {
            "skill_match_score": 0,
            "entity_match_score": 0,
            "skill_with_entity_match_score": 0,
            "exact_match_score": 0,
            "total_score": 0,
        }
        record["success"] = False
        record["error_type"] = "scoring_error"
        record["error_message"] = f"{type(e).__name__}: {e}"
        record["scoring_traceback"] = traceback.format_exc()

    return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the summary JSON over all evaluated rows."""
    n = len(records)
    n_crashes = sum(1 for r in records if r.get("error_type") == "model_crash")
    parsed = [r for r in records if r.get("parse_success")]
    n_parsed = len(parsed)
    parse_rate = n_parsed / n if n else 0.0

    n_scored = sum(1 for r in records if isinstance(r.get("score"), dict))
    score_rate = n_scored / n if n else 0.0

    def _mean_score_field(rs: list[dict[str, Any]], key: str) -> float | None:
        vals = [
            r["score"][key]
            for r in rs
            if isinstance(r.get("score"), dict) and r["score"].get(key) is not None
        ]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    by_task: dict[str, dict[str, Any]] = defaultdict(dict)
    by_cot: dict[str, dict[str, Any]] = defaultdict(dict)
    by_image_mode: dict[str, dict[str, Any]] = defaultdict(dict)
    by_task_cot_mode: dict[str, dict[str, Any]] = defaultdict(dict)

    for partition_name, partitioner in [
        ("task", lambda r: r["task"]),
        ("cot", lambda r: cot_tag(bool(r["cot"]))),
        ("image_mode", lambda r: r["image_mode"]),
        ("task__cot__image_mode",
         lambda r: f"{r['task']}__{cot_tag(bool(r['cot']))}__{r['image_mode']}"),
    ]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            groups[partitioner(r)].append(r)

        target = {
            "task": by_task,
            "cot": by_cot,
            "image_mode": by_image_mode,
            "task__cot__image_mode": by_task_cot_mode,
        }[partition_name]

        for key, rs in groups.items():
            n_g = len(rs)
            target[key] = {
                "n_total": n_g,
                "n_parsed": sum(1 for r in rs if r.get("parse_success")),
                "n_crashes": sum(1 for r in rs if r.get("error_type") == "model_crash"),
                "n_success": sum(1 for r in rs if r.get("success")),
                "parse_rate": (
                    sum(1 for r in rs if r.get("parse_success")) / n_g if n_g else 0.0
                ),
                "success_rate": (
                    sum(1 for r in rs if r.get("success")) / n_g if n_g else 0.0
                ),
                "mean_total_score": _mean_score_field(rs, "total_score"),
                "mean_skill_match_score": _mean_score_field(rs, "skill_match_score"),
                "mean_entity_match_score": _mean_score_field(rs, "entity_match_score"),
                "mean_exact_match_score": _mean_score_field(rs, "exact_match_score"),
            }

    # Representative example outputs.
    succ_examples = [r for r in records if r.get("parse_success")][:5]
    fail_examples = [r for r in records if not r.get("parse_success")][:5]

    return {
        "n_examples": n,
        "n_crashes": n_crashes,
        "parse_success_rate": parse_rate,
        "scoring_success_rate": score_rate,
        "mean_total_score": _mean_score_field(records, "total_score"),
        "mean_skill_match_score": _mean_score_field(records, "skill_match_score"),
        "mean_entity_match_score": _mean_score_field(records, "entity_match_score"),
        "mean_exact_match_score": _mean_score_field(records, "exact_match_score"),
        "by_task": dict(by_task),
        "by_cot": dict(by_cot),
        "by_image_mode": dict(by_image_mode),
        "by_task__cot__image_mode": dict(by_task_cot_mode),
        "representative_failures": [
            {
                "task": r.get("task"),
                "episode_id": r.get("episode_id"),
                "cot": r.get("cot"),
                "image_mode": r.get("image_mode"),
                "error_type": r.get("error_type"),
                "raw_output": (
                    str(r.get("raw_output"))[:1500]
                    if r.get("raw_output") is not None
                    else None
                ),
            }
            for r in fail_examples
        ],
        "representative_successes": [
            {
                "task": r.get("task"),
                "episode_id": r.get("episode_id"),
                "cot": r.get("cot"),
                "image_mode": r.get("image_mode"),
                "raw_output": (
                    str(r.get("raw_output"))[:1500]
                    if r.get("raw_output") is not None
                    else None
                ),
                "score": r.get("score"),
            }
            for r in succ_examples
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Override the SMOKE_TEST module flag.")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="Force a full run regardless of the SMOKE_TEST flag.")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_KEY_QWEN2VL,
        choices=[MODEL_KEY_QWEN2VL, MODEL_KEY_LLADA_V],
        help="Which VLM wrapper to use. qwen2vl=Qwen2-VL (default), "
             "llada_v=LLaDA-V baseline (no steering).",
    )
    parser.add_argument("--tasks", nargs="+", type=str, default=list(DEFAULT_TASKS))
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Episodes per task (default depends on smoke flag).")
    parser.add_argument("--cot", nargs="+", type=str, default=None,
                        choices=["on", "off"],
                        help="Which CoT settings to run; default = both.")
    parser.add_argument("--image-modes", nargs="+", type=str,
                        default=list(DEFAULT_IMAGE_MODES),
                        choices=["original_plus_numbered", "numbered_only"])
    parser.add_argument("--few-shot-num", type=int, default=DEFAULT_FEW_SHOT_NUM)
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HF model name. Defaults to the standard checkpoint for --model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Qwen2-VL generation budget (ignored by LLaDA-V; see --gen-length).",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=256,
        help="LLaDA-V gen_length (number of generation positions).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="LLaDA-V denoising steps (must divide num_blocks).",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=128,
        help="LLaDA-V block length (gen_length must be divisible by it).",
    )
    parser.add_argument(
        "--llada-v-max-gpu-gib",
        type=float,
        default=10.0,
        help="GPU weight budget passed to accelerate's max_memory for LLaDA-V "
             "(rest spills to CPU). Only used if --llada-v-device-map auto.",
    )
    parser.add_argument(
        "--llada-v-device-map",
        type=str,
        default="cpu",
        choices=["cpu", "auto"],
        help="Where to put the LLaDA-V weights. cpu (default) is the "
             "reliable path on a 12 GB GPU; auto attempts an accelerate-"
             "based GPU/CPU split (currently flaky under transformers 5.x "
             "with the bundled LLaDA-V SigLIP code).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_smoke(args: argparse.Namespace) -> bool:
    if args.smoke_test:
        return True
    if args.no_smoke_test:
        return False
    return SMOKE_TEST


def main() -> None:
    args = parse_args()
    smoke = resolve_smoke(args)

    n_episodes = args.n_episodes
    if n_episodes is None:
        n_episodes = DEFAULT_SMOKE_EPISODES if smoke else DEFAULT_FULL_EPISODES

    if args.cot is None:
        cot_values = list(DEFAULT_COT_VALUES)
    else:
        cot_values = sorted({c == "on" for c in args.cot})
    image_modes = list(args.image_modes)

    # Resolve per-model defaults.
    model_defaults = MODEL_DEFAULTS[args.model]
    model_name = args.model_name or model_defaults["model_name"]
    if args.out_dir is None:
        out_dir = REPO_ROOT / "results" / "vla_bench" / model_defaults["out_subdir"]
    else:
        out_dir = args.out_dir

    out_root = out_dir / ("smoke_test" if smoke else "full_run")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[setup] model_key={args.model}")
    print(f"[setup] smoke_test={smoke}")
    print(f"[setup] tasks={args.tasks}")
    print(f"[setup] n_episodes={n_episodes}")
    print(f"[setup] cot={cot_values}, image_modes={image_modes}")
    print(f"[setup] model_name={model_name}")
    print(f"[setup] out_root={out_root}")

    # Pre-load shared resources.
    pre_prompt = load_pre_prompt(language=args.language)
    seq_independent_tasks = load_seq_independent_tasks()

    # Group requested tasks by dim, since few-shot sampling is dim-scoped.
    tasks_by_dim: dict[str, list[str]] = defaultdict(list)
    for task_name in args.tasks:
        if task_name not in vlabench_paths.SANITY_TASKS:
            raise ValueError(
                f"Task {task_name!r} not in the sanity grid. "
                f"Allowed: {sorted(vlabench_paths.SANITY_TASKS)}"
            )
        tasks_by_dim[vlabench_paths.SANITY_TASKS[task_name]].append(task_name)

    # Build the list of (task, episode_num) work items, per dim.
    work_items_by_dim: dict[str, list[tuple[str, int]]] = {}
    available_shots_by_dim: dict[str, dict[str, list[int]]] = {}
    available_shot_tasks_by_dim: dict[str, list[str]] = {}

    for dim, dim_tasks in tasks_by_dim.items():
        data_dim_root = vlabench_paths.dataset_dim_dir(dim)
        if not data_dim_root.exists():
            raise FileNotFoundError(
                f"VLABench dataset dim {dim} not found at {data_dim_root}. "
                "Run the snapshot_download command from the README first."
            )
        # Available shot tasks: every task on disk under this dim.
        on_disk_tasks = sorted(
            t for t in os.listdir(data_dim_root)
            if (data_dim_root / t).is_dir()
        )
        shots_per_task = {
            t: list_examples(data_dim_root, t) for t in on_disk_tasks
        }
        available_shot_tasks_by_dim[dim] = on_disk_tasks
        available_shots_by_dim[dim] = shots_per_task

        items: list[tuple[str, int]] = []
        for task_name in dim_tasks:
            episodes_on_disk = list_examples(data_dim_root, task_name)
            chosen = episodes_on_disk[:n_episodes]
            for ep in chosen:
                items.append((task_name, ep))
        work_items_by_dim[dim] = items

    total_examples = sum(
        len(items) * len(cot_values) * len(image_modes)
        for items in work_items_by_dim.values()
    )
    print(f"[setup] total examples to evaluate: {total_examples}")

    # Load the VLM.
    print(f"[model] loading {model_name} (this may take a while)…")
    if args.model == MODEL_KEY_QWEN2VL:
        from vla_bench.model_loader import Qwen2_VL_Local
        vlm = Qwen2_VL_Local(
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == MODEL_KEY_LLADA_V:
        from vla_bench.llada_v_loader import Llada_V_Local
        # LLaDA-V is ~16 GB in fp16. Multiple attempts to split across the
        # 12 GB GPU (device_map="auto", bnb 4/8-bit, manual dispatch) all
        # tripped transformers 5.x's meta-tensor handling around the
        # patched SigLIP submodule. The smoke test therefore runs on CPU
        # by default; flip --llada-v-device-map to "auto" if you have a
        # GPU big enough to hold the full fp16 model.
        if args.llada_v_device_map == "cpu":
            llada_v_device_map = None
        elif args.llada_v_device_map == "auto":
            llada_v_device_map = "auto"
        else:
            llada_v_device_map = args.llada_v_device_map

        if llada_v_device_map == "auto":
            gib = max(1.0, float(args.llada_v_max_gpu_gib))
            max_memory = {0: f"{gib:.1f}GiB", "cpu": "60GiB"}
        else:
            max_memory = None

        vlm = Llada_V_Local(
            model_name=model_name,
            attn_implementation="sdpa",
            device_map=llada_v_device_map,
            max_memory=max_memory,
            torch_dtype="float16",
            generation_kwargs={
                "steps": int(args.steps),
                "gen_length": int(args.gen_length),
                "block_length": int(args.block_length),
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
            },
            stopping_criteria=("<|eot_id|>",),
            use_fast_dllm=False,
            use_dllm_cache=False,
        )
    else:  # pragma: no cover — argparse choices guard this.
        raise ValueError(f"Unknown --model {args.model!r}")
    print(f"[model] loaded; name={vlm.name}")

    rng = random.Random(args.seed)

    # Run.
    all_records: list[dict[str, Any]] = []
    jsonl_path = out_root / "examples.jsonl"
    started_at = time.time()

    with jsonl_path.open("w", encoding="utf-8") as fout:
        idx = 0
        for dim, items in work_items_by_dim.items():
            data_dim_root = vlabench_paths.dataset_dim_dir(dim)
            available_shot_tasks = available_shot_tasks_by_dim[dim]
            available_shots_per_task = available_shots_by_dim[dim]
            dataset_root_str = str(vlabench_paths.VLABENCH_DATASET_ROOT)

            for with_cot in cot_values:
                for image_mode in image_modes:
                    for task_name, episode_num in items:
                        idx += 1
                        # Re-seed per example so few-shot sampling is reproducible.
                        ex_rng = random.Random(
                            (args.seed, task_name, episode_num,
                             bool(with_cot), image_mode)
                        )
                        t0 = time.time()
                        record = evaluate_single_example(
                            vlm=vlm,
                            data_dim_root=data_dim_root,
                            task_name=task_name,
                            example_num=episode_num,
                            pre_prompt=pre_prompt,
                            few_shot_num=args.few_shot_num,
                            language=args.language,
                            with_cot=with_cot,
                            image_mode=image_mode,
                            rng=ex_rng,
                            available_shot_tasks=available_shot_tasks,
                            available_shots_per_task=available_shots_per_task,
                            seq_independent_tasks=seq_independent_tasks,
                            dataset_root_str=dataset_root_str,
                        )
                        record["wall_time_s"] = round(time.time() - t0, 3)
                        all_records.append(record)
                        fout.write(json.dumps(record) + "\n")
                        fout.flush()

                        msg = (
                            f"[{idx:03d}/{total_examples}] "
                            f"task={task_name} ep={episode_num} "
                            f"cot={with_cot} img={image_mode} "
                            f"parse={record.get('parse_success')} "
                            f"err={record.get('error_type')} "
                            f"score={(record.get('score') or {}).get('total_score')}"
                        )
                        print(msg, flush=True)

    summary = aggregate(all_records)
    summary["smoke_test"] = bool(smoke)
    summary["n_episodes_per_task"] = int(n_episodes)
    summary["tasks"] = args.tasks
    summary["cot_values"] = [bool(c) for c in cot_values]
    summary["image_modes"] = image_modes
    summary["few_shot_num"] = args.few_shot_num
    summary["language"] = args.language
    summary["model_key"] = args.model
    summary["model_name"] = model_name
    if args.model == MODEL_KEY_LLADA_V:
        summary["llada_v_generation_kwargs"] = {
            "steps": int(args.steps),
            "gen_length": int(args.gen_length),
            "block_length": int(args.block_length),
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
            "stopping_criteria": ["<|eot_id|>"],
        }
        summary["llada_v_max_gpu_gib"] = float(args.llada_v_max_gpu_gib)
    summary["seed"] = args.seed
    summary["out_dir"] = str(out_root)
    summary["wall_time_s"] = round(time.time() - started_at, 3)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"[done] examples.jsonl  -> {jsonl_path}")
    print(f"[done] summary.json    -> {out_root / 'summary.json'}")
    print(
        f"[done] parse_success_rate={summary['parse_success_rate']:.3f}, "
        f"scoring_success_rate={summary['scoring_success_rate']:.3f}, "
        f"mean_total_score={summary['mean_total_score']}, "
        f"n_crashes={summary['n_crashes']}"
    )

    del vlm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
