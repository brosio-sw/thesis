# VLABench VLM sanity grid

A small isolated experiment runner that confirms the VLABench VLM
evaluation pipeline works end-to-end on this machine before we touch LLaDA-V.

## What it tests

* VLABench dataset loads correctly (3 easy tasks).
* Images load correctly.
* The official prompt is constructed correctly.
* `Qwen2-VL` runs without crashing.
* JSON / skill-sequence parsing works (or fails diagnosably).
* The official VLABench scoring runs.

## Pipeline reuse

We use the *official* VLABench primitives unchanged:

| Concern        | Source                                                                  |
|----------------|-------------------------------------------------------------------------|
| Pre-prompt     | `VLABench/configs/prompt/eval_vlm_en.txt`                               |
| Chat template  | `VLABench.evaluation.model.vlm.base.get_ti_list`                        |
| VLM wrapper    | `VLABench.evaluation.model.vlm.qwen_vl.Qwen2_VL` (subclassed; see below)|
| JSON parsing   | The wrapper's own `output_text.split("```json")...` (unchanged)         |
| Scoring        | `VLABench.evaluation.utils.get_final_score`                             |
| Seq dependency | `VLABench/configs/evaluation/seq_independent_task.json`                 |

### Two deliberate, isolated additions

1. `vla_bench/model_loader.py::Qwen2_VL_Local` subclasses the official
   `Qwen2_VL` only to replace `__init__`. The official wrapper hardcodes
   `qwen/Qwen2-VL-7B-Instruct` via `modelscope` and forces
   `flash_attention_2`, neither of which works on this 12 GB GPU. We load
   from HuggingFace, default to `Qwen/Qwen2-VL-2B-Instruct` (fits in 12 GB),
   and use `attn_implementation="sdpa"`. The `evaluate(...)` body is
   functionally identical to the official one (same prompt, same parser).

2. `vla_bench/image_utils.py::filter_ti_list_by_image_mode` post-processes
   the official `ti_list` to support a `numbered_only` ablation in addition
   to the official `original_plus_numbered` mode. It only drops image
   entries (and the matching `"Input picture"` / `"Example N input picture:"`
   label entries); it never edits the rest of the prompt. The default mode
   is identical to the official path.

We do **not** subclass / use `VLMEvaluator` because importing it pulls in
`VLABench.envs` → `mujoco`, `dm_control`, `mediapy` etc. — all the simulation
deps we don't need for VLM evaluation. Instead we replicate the few lines
of `VLMEvaluator.build_input` (pure path joins + json reads) inside
`vla_bench/prompts.py` and route results through the official scorer.

## Layout

```text
vla_bench/
  __init__.py
  vlabench_paths.py        # adds VLABench clone to sys.path, sets VLABENCH_ROOT
  prompts.py               # build_input + ti_list (uses official prompt + get_ti_list)
  parsing.py               # diagnostics around the wrapper's parsed output
  image_utils.py           # image_mode filter applied AFTER get_ti_list
  model_loader.py          # Qwen2-VL subclass that loads from HF on local GPU
  run_vlm_sanity_grid.py   # main runner
  README.md                # this file
  VLABench/                # cloned upstream repo (https://github.com/OpenMOSS/VLABench)
  data/vlm_evaluation_v1.0/
        M&T/select_fruit/...
        M&T/select_toy/...
        CommenSence/select_fruit_common_sense/...
```

## Setup

Activate the project conda env:

```bash
conda activate thesis
```

Install the minimal extra deps (one-time):

```bash
pip install qwen-vl-utils colorama networkx
```

Clone VLABench (one-time):

```bash
git clone https://github.com/OpenMOSS/VLABench.git vla_bench/VLABench
```

Download the three tasks from HuggingFace (one-time, ~500 MB):

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="VLABench/vlm_evaluation_v1.0",
    repo_type="dataset",
    local_dir="vla_bench/data/vlm_evaluation_v1.0",
    allow_patterns=[
        "M&T/select_fruit/**",
        "M&T/select_toy/**",
        "CommenSence/select_fruit_common_sense/**",
    ],
)
```

The HF release uses the typo folder name `CommenSence` instead of
`CommonSense`; `vla_bench/vlabench_paths.py::dataset_dim_dir` handles
the mapping.

## Running

```bash
# Smoke test (3 tasks × 2 episodes × 2 CoT × 2 image modes = 24 examples)
python -m vla_bench.run_vlm_sanity_grid

# Full grid (3 tasks × 25 episodes × 2 CoT × 2 image modes = 300 examples)
python -m vla_bench.run_vlm_sanity_grid --no-smoke-test
```

Toggle the smoke flag in code:

```python
# vla_bench/run_vlm_sanity_grid.py
SMOKE_TEST = True   # or False
```

CLI overrides:

```bash
python -m vla_bench.run_vlm_sanity_grid \
    --tasks select_fruit select_toy select_fruit_common_sense \
    --image-modes original_plus_numbered numbered_only \
    --cot on off \
    --few-shot-num 1 \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --n-episodes 25 \
    --no-smoke-test
```

Outputs land under:

```text
results/vla_bench/qwen2vl_sanity/{smoke_test,full_run}/
    examples.jsonl   # one row per evaluated example with full diagnostics
    summary.json     # aggregate metrics (parse rate, scoring rate, breakdowns)
```

Each `examples.jsonl` row contains:

```json
{
  "task": "...",
  "episode_id": 0,
  "few_shot": 1,
  "cot": false,
  "image_mode": "original_plus_numbered",
  "language": "en",
  "instruction": "...",
  "image_paths": ["M&T/select_fruit/example0/input/input.png", "..."],
  "n_images_passed": 4,
  "prompt": "...full chat-template text with [image:...] placeholders...",
  "raw_output": "...the model's verbatim output...",
  "parsed_output": {"skill_sequence": [...]},
  "parse_success": true,
  "score": {"skill_match_score": ..., "total_score": ...},
  "success": false,
  "error_type": null,
  "error_message": null,
  "wall_time_s": 12.345
}
```

## Few-shot reproducibility

The official `VLMEvaluator.build_input` samples few-shot examples with
`random.choice(...)` from a *process-global* `random` state. Our runner
gives every example its own `random.Random` seeded by
`(args.seed, task, episode, cot, image_mode)` so a re-run with the same
`--seed` chooses the same few-shot demonstration per example.

## Known constraints

* The 7B variant of Qwen2-VL does not fit on a 12 GB RTX 4070 Ti in bf16,
  so the runner defaults to `Qwen/Qwen2-VL-2B-Instruct`. Override with
  `--model-name` if you have more VRAM.

## Future extensions

* Add a LLaDA-V wrapper that exposes the same surface as `Qwen2_VL_Local`
  (same `evaluate_with_ti_list(ti_list, with_CoT=...) -> dict[str, Any]`).
  The runner is model-agnostic above that interface.
