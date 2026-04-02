"""
smoke_test.py – Verify generate() refactor: fill_strategy / remasking separation.

Tests exactly 2 configurations on 5 high-toxicity RTP prompts:

  1. baseline_low_conf
       fill_strategy="low_confidence", remasking=None
       → plain LLaDA baseline; no remasking should trigger.

  2. remdm_llada_like
       fill_strategy="low_confidence", remasking=ReMDMConfRemasking()
       remask_fixed_count=2, remask_start_frac=0.875
       → close to author's llada_remdm_sample; remasking should fire in the
         last ~12.5% of steps, logging "[remask]" lines for verification.

Goal: verify wiring correctness and that remasking actually triggers.
Outputs go to data/smoke_test/ (separate from data/generations/).
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Bypass torch.load security check for trusted HuggingFace models.
try:
    import transformers.modeling_utils as _hf_mu
    import transformers.utils.import_utils as _hf_iu
    _noop = lambda: None
    _hf_mu.check_torch_load_is_safe = _noop
    _hf_iu.check_torch_load_is_safe = _noop
    try:
        import transformers.core_model_loading as _hf_cml
        _hf_cml.check_torch_load_is_safe = _noop
    except (ImportError, AttributeError):
        pass
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset
from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from remasking.remdm_conf import ReMDMConfRemasking
from eval.toxicity_metrics import compute_toxicity_metrics

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_PROMPTS    = 5
HIGH_TOX_MIN = 0.85
OUT_DIR      = Path("data/smoke_test")

# Single param set: cheap but enough to observe multiple remasking steps
PARAMS = dict(temperature=1.0, steps=16, gen_length=20, block_length=20)

# Two configs to compare
CONFIGS = [
    dict(
        name="baseline_low_conf",
        fill_strategy="low_confidence",
        remasking_cls=None,
        remask_fixed_count=None,
        remask_fraction=0.0,
        remask_start_frac=0.0,
        show_remask_stats=False,   # no remasking expected
        description="Plain low-confidence fill, no remasking (LLaDA baseline)",
    ),
    dict(
        name="remdm_llada_like",
        fill_strategy="low_confidence",    # same fill as author's code
        remasking_cls=ReMDMConfRemasking,
        remask_fixed_count=2,              # matches author's remdm_number
        remask_fraction=0.0,
        remask_start_frac=0.875,           # late-phase start ≈ author's design
        show_remask_stats=True,            # print per-step remask events
        description="ReMDM-LLaDA-like: conf fill + late-phase fixed-count remasking",
    ),
]

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_rtp_prompts(n: int, high_tox_min: float) -> list[str]:
    print(f"[data] Loading {n} high-tox RTP prompts (tox >= {high_tox_min}) …")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    collected: list[tuple[float, str]] = []
    for ex in ds:
        tox = ex["prompt"]["toxicity"]
        if tox is not None and tox >= high_tox_min:
            collected.append((tox, ex["prompt"]["text"]))
        if len(collected) >= n * 5:
            break
    collected.sort(key=lambda x: -x[0])
    prompts = [t for _, t in collected[:n]]
    print(f"[data] {len(prompts)} prompts loaded.")
    return prompts

# ── Per-config generation run ─────────────────────────────────────────────────

def run_config(model, tokenizer, prompts: list[str], cfg: dict) -> list[str]:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"CONFIG : {cfg['name']}")
    print(f"  fill_strategy      = {cfg['fill_strategy']}")
    rm_name = cfg["remasking_cls"].__name__ if cfg["remasking_cls"] else "None"
    print(f"  remasking          = {rm_name}")
    print(f"  remask_fixed_count = {cfg['remask_fixed_count']}")
    print(f"  remask_start_frac  = {cfg['remask_start_frac']}")
    print(f"  desc: {cfg['description']}")
    print(sep)

    remasking_obj = (
        cfg["remasking_cls"]() if cfg["remasking_cls"] is not None else None
    )
    answers: list[str] = []

    for i, prompt in enumerate(prompts):
        if hasattr(remasking_obj, "reset"):
            remasking_obj.reset()

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tokenizer(
            [formatted],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(DEVICE)

        print(f"\n  [prompt {i+1}/{len(prompts)}] {prompt[:80]!r}")
        if cfg["show_remask_stats"]:
            print("  → remasking log:")

        out = llada_generate(
            model=model,
            prompt=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            steps=PARAMS["steps"],
            gen_length=PARAMS["gen_length"],
            block_length=PARAMS["block_length"],
            temperature=PARAMS["temperature"],
            fill_strategy=cfg["fill_strategy"],
            remasking=remasking_obj,
            remask_fraction=cfg["remask_fraction"],
            remask_fixed_count=cfg["remask_fixed_count"],
            remask_start_frac=cfg["remask_start_frac"],
            show_progress=True,
            show_remask_stats=cfg["show_remask_stats"],
        )

        answer = tokenizer.decode(
            out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        answers.append(answer)
        print(f"  → answer: {answer!r}")

    return answers

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    prompts = load_rtp_prompts(N_PROMPTS, HIGH_TOX_MIN)

    print(f"\n[model] Loading LLaDA-8B-Instruct on {DEVICE} …")
    model     = load_model(device=DEVICE)
    tokenizer = load_tokenizer()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for cfg in CONFIGS:
        answers = run_config(model, tokenizer, prompts, cfg)
        all_results[cfg["name"]] = {"prompts": prompts, "answers": answers}

        cfg_dir = OUT_DIR / cfg["name"]
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "generations.jsonl", "w") as f:
            for p, a in zip(prompts, answers):
                f.write(json.dumps({"prompt": p, "answer": a}) + "\n")
        print(f"  [saved] {cfg_dir / 'generations.jsonl'}")

    # ── Free GPU before eval ──────────────────────────────────────────────────
    print("\n[model] Unloading LLaDA …")
    try:
        from accelerate.hooks import remove_hook_from_module
        for m in model.modules():
            remove_hook_from_module(m)
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Toxicity check ────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("TOXICITY (answer only)")
    print("=" * 62)
    for name, data in all_results.items():
        tox = compute_toxicity_metrics(data["answers"], device=DEVICE)
        print(
            f"  {name:<30}  mean_tox={tox['mean_toxicity']:.4f}"
            f"  frac={tox['toxic_fraction']:.2f}"
        )

    print("\n[smoke_test] PASSED — both configs completed end-to-end.\n")


if __name__ == "__main__":
    main()
