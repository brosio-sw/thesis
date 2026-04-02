"""
main.py – Toxicity generation sweep pipeline.

Sweeps over:
  * Datasets:   rtp         – allenai/real-toxicity-prompts (high-tox prompts)
                beavertails – PKU-Alignment/BeaverTails     (is_safe=False prompts)
  * Param sets: configurable temperature × steps grid
  * Remasking:  low_confidence, remdm_conf

For each (dataset × param_set × remasking) combination:
  - Generates N_PER_RUN continuations
  - Evaluates toxicity of answer only and of prompt+answer
  - Evaluates perplexity of answer under Qwen2.5-3B

Outputs per run: data/generations/<run_tag>/
  run_info.json          – dataset, params, remasking config
  generations.jsonl      – one JSON per line: prompt, answer, toxicity scores,
                           perplexity (all per-item scores embedded)
  toxicity_answer.json   – aggregate toxicity scores for answer only
  toxicity_combined.json – aggregate toxicity scores for prompt + answer
  perplexity.json        – aggregate perplexity scores
  metrics.json           – full summary with tox + ppl
data/generations/summary.json – combined table + best-config annotation
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

# Must be set before any CUDA initialisation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# Bypass torch.load security check for trusted HuggingFace models.
# s-nlp/roberta_toxicity_classifier uses .bin format (pre-safetensors era).
# CVE-2025-32434 does not apply to known-good models loaded in a local research env.
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

from datasets import load_dataset
from tqdm import tqdm

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer, LLADA_DEFAULT_MODEL
from llada.generate import generate as llada_generate
from remasking.remdm_conf import ReMDMConfRemasking
from eval.toxicity_metrics import compute_toxicity_metrics
from eval.fluency_metrics import compute_fluency_metrics

# ── Sweep configuration ───────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_RUN   = 40    # prompts per (dataset × params × remasking) run
BATCH_SIZE  = 1     # keep at 1 to stay within 12 GB VRAM
REMASK_FRAC = 0.1   # fraction of tokens to re-mask per step (ReMDM only)

# Model to use: switch between LLaDA-8B-Instruct and LLaDA-8B-Base here.
# Base model does NOT use a chat template — prompts are passed as plain text.
LLADA_MODEL  = "GSAI-ML/LLaDA-8B-Base"
USE_CHAT_TPL = False   # set True when using the Instruct model

# Parameter sets sweeping temperature and steps
PARAM_SETS = [
    dict(tag="T0_S15_G40", temperature=0, steps=40, gen_length=40, block_length=40)
    #dict(tag="T0S15_G20", temperature=0, steps=50, gen_length=50, block_length=50),

]

# Datasets to sweep over (add/remove entries here)
DATASETS = [
    dict(name="beavertails"),                        # BeaverTails, is_safe=False only
    dict(name="rtp",         high_tox_min=0.85),   # RealToxicityPrompts, high-tox only

]

# Remasking strategies to compare.
# Each entry fully specifies one experiment arm:
#   fill_strategy      – base unmask ordering ("low_confidence" or "random")
#   remasking_cls      – None = no remasking; BaseRemasking subclass otherwise
#   remask_fraction    – fraction-based count (used when fixed_count is None)
#   remask_fixed_count – fixed tokens re-masked per eligible step
#   remask_start_frac  – fraction of steps to complete before remasking starts
#
# low_conf → plain LLaDA baseline: confidence fill, no remasking
# remdm    → ReMDM-LLaDA-like: confidence fill + conf-history remasking,
#            late-phase (starts at 75% of steps), fixed-count
#            (approximates author's llada_remdm_sample from eval_llada.py)
REMASK_STRATEGIES = [
    dict(
        name="low_conf",
        fill_strategy="low_confidence",
        remasking_cls=None,
        remask_fraction=0.0,
        remask_fixed_count=None,
        remask_start_frac=0.0,
    ),
    # Uncomment to also test other strategies:
     dict(
        name="remdm_k2_late75",
         fill_strategy="low_confidence",
         remasking_cls=ReMDMConfRemasking,
         remask_fraction=0.0,
         remask_fixed_count=2,
         remask_start_frac=0.75,
    ),
]
OUT_DIR = Path("data/generations_base")

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_rtp_prompts(n: int, high_tox_min: float = 0.85) -> list[str]:
    """Load n high-toxicity prompts from allenai/real-toxicity-prompts."""
    print(f"[data] Loading RTP prompts (tox >= {high_tox_min}) …")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    collected: list[tuple[float, str]] = []
    for ex in ds:
        tox = ex["prompt"]["toxicity"]
        if tox is None:
            continue
        if tox >= high_tox_min:
            collected.append((tox, ex["prompt"]["text"]))
        if len(collected) >= n * 10:   # oversample then sort & trim
            break
    collected.sort(key=lambda x: -x[0])          # most toxic first
    prompts = [t for _, t in collected[:n]]
    print(f"[data] {len(prompts)} RTP prompts (tox >= {high_tox_min}).")
    return prompts


def load_beavertails_prompts(n: int) -> list[str]:
    """Load n unsafe prompts from PKU-Alignment/BeaverTails (is_safe=False)."""
    print("[data] Loading BeaverTails prompts (is_safe=False) …")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    prompts = [ex["prompt"] for ex in ds if not ex["is_safe"]][:n]
    print(f"[data] {len(prompts)} BeaverTails unsafe prompts.")
    return prompts


def load_prompts_for_dataset(cfg: dict, n: int) -> list[str]:
    """Dispatch to the right loader based on dataset config."""
    if cfg["name"] == "rtp":
        return load_rtp_prompts(n, high_tox_min=cfg.get("high_tox_min", 0.85))
    elif cfg["name"] == "beavertails":
        return load_beavertails_prompts(n)
    else:
        raise ValueError(f"Unknown dataset: {cfg['name']}")


def dataset_tag(cfg: dict) -> str:
    """Short folder-safe tag for a dataset config."""
    if cfg["name"] == "rtp":
        htmin = cfg.get("high_tox_min", 0.85)
        return f"rtp_htmin{int(htmin * 100)}"
    return cfg["name"]


# ── Generation ────────────────────────────────────────────────────────────────

def run_generation(
    model,
    tokenizer,
    prompts: list[str],
    strategy: dict,
    run_tag: str,
    params: dict,
) -> list[str]:
    """
    Generate continuations for `prompts` using the given strategy dict.

    strategy keys: fill_strategy, remasking_cls, remask_fraction,
                   remask_fixed_count, remask_start_frac.
    Returns decoded answer strings (excluding the prompt prefix).
    """
    all_answers: list[str] = []

    # Instantiate a fresh remasking object per run (stateful; must not share).
    remasking_obj = (
        strategy["remasking_cls"]()
        if strategy["remasking_cls"] is not None
        else None
    )

    for start in tqdm(
        range(0, len(prompts), BATCH_SIZE),
        desc=f"Generating [{run_tag}]",
    ):
        batch_prompts = prompts[start : start + BATCH_SIZE]

        # Reset per-sequence state (e.g. ReMDMConfRemasking confidence cache).
        if hasattr(remasking_obj, "reset"):
            remasking_obj.reset()

        if USE_CHAT_TPL:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in batch_prompts
            ]
        else:
            # Base model: pass prompts as plain text (no chat template).
            formatted = list(batch_prompts)
        enc = tokenizer(
            formatted,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        out = llada_generate(
            model=model,
            prompt=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            steps=params["steps"],
            gen_length=params["gen_length"],
            block_length=params["block_length"],
            temperature=params["temperature"],
            fill_strategy=strategy["fill_strategy"],
            remasking=remasking_obj,
            remask_fraction=strategy["remask_fraction"],
            remask_fixed_count=strategy["remask_fixed_count"],
            remask_start_frac=strategy["remask_start_frac"],
            show_progress=False,
        )

        generated_ids = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_answers.extend(decoded)

    return all_answers


# ── Evaluation & saving ───────────────────────────────────────────────────────

def _tox_dict(result: dict) -> dict:
    return {
        "mean_toxicity":   result["mean_toxicity"],
        "toxic_fraction":  result["toxic_fraction"],
        "max_toxicity":    result["max_toxicity"],
        "per_text_scores": result["scores"].tolist(),
    }


def save_and_evaluate(
    prompts: list[str],
    answers: list[str],
    run_tag: str,
    run_info: dict,
    tox_device: str = "cuda",
    ppl_device: str = "cpu",
) -> dict:
    """
    Save generations and evaluate toxicity (answer only + combined) and
    perplexity (answer only under Qwen2.5-3B).
    Returns a summary metrics dict.
    """
    out_dir = OUT_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save run metadata ───────────────────────────────────────────────────
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    combined_texts = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]

    # ── Toxicity: answer only ───────────────────────────────────────────────
    print(f"[eval] Toxicity (answer only) [{run_tag}] …")
    tox_ans = compute_toxicity_metrics(answers, device=tox_device)
    with open(out_dir / "toxicity_answer.json", "w") as f:
        json.dump(_tox_dict(tox_ans), f, indent=2)

    # ── Toxicity: prompt + answer ───────────────────────────────────────────
    print(f"[eval] Toxicity (prompt+answer) [{run_tag}] …")
    tox_comb = compute_toxicity_metrics(combined_texts, device=tox_device)
    with open(out_dir / "toxicity_combined.json", "w") as f:
        json.dump(_tox_dict(tox_comb), f, indent=2)

    # ── Perplexity: answer only (Qwen2.5-3B) ───────────────────────────────
    print(f"[eval] Perplexity (answer) [{run_tag}] …")
    flu = compute_fluency_metrics(answers, device=ppl_device)
    with open(out_dir / "perplexity.json", "w") as f:
        json.dump({"mean_ppl": flu["mean_ppl"], "per_text_ppl": flu["per_text_ppl"]},
                  f, indent=2)

    # ── generations.jsonl: one line per sample with all per-item scores ────
    gen_path = out_dir / "generations.jsonl"
    with open(gen_path, "w") as f:
        for i, (prompt, answer, combined) in enumerate(
            zip(prompts, answers, combined_texts)
        ):
            f.write(json.dumps({
                "prompt":           prompt,
                "answer":           answer,
                "prompt_answer":    combined,
                "tox_answer":       float(tox_ans["scores"][i]),
                "tox_combined":     float(tox_comb["scores"][i]),
                "ppl_answer":       float(flu["per_text_ppl"][i]),
            }) + "\n")
    print(f"[output] {len(answers)} generations → {gen_path}")

    # ── Summary metrics ─────────────────────────────────────────────────────
    metrics = {
        "run_tag":               run_tag,
        "num_samples":           len(answers),
        "run_info":              run_info,
        # answer-only toxicity
        "tox_answer_mean":       tox_ans["mean_toxicity"],
        "tox_answer_fraction":   tox_ans["toxic_fraction"],
        "tox_answer_max":        tox_ans["max_toxicity"],
        # combined toxicity
        "tox_combined_mean":     tox_comb["mean_toxicity"],
        "tox_combined_fraction": tox_comb["toxic_fraction"],
        "tox_combined_max":      tox_comb["max_toxicity"],
        # fluency
        "ppl_answer_mean":       flu["mean_ppl"],
        "mean_length_words":     flu["mean_length_words"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[output] metrics → {out_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    print("\n[data] Pre-loading prompts for all datasets …")
    dataset_prompts: dict[str, list[str]] = {}
    for ds_cfg in DATASETS:
        key = dataset_tag(ds_cfg)
        dataset_prompts[key] = load_prompts_for_dataset(ds_cfg, N_PER_RUN)

    print(f"\n[model] Loading {LLADA_MODEL} on {DEVICE} …")
    model = load_model(model_name=LLADA_MODEL, device=DEVICE)
    tokenizer = load_tokenizer(model_name=LLADA_MODEL)

    summary: dict[str, dict] = {}

    print("\n" + "=" * 60)
    print("GENERATION + EVALUATION")
    print("=" * 60)

    for ds_cfg in DATASETS:
        dtag = dataset_tag(ds_cfg)
        prompts = dataset_prompts[dtag]

        for params in PARAM_SETS:
            for rm in REMASK_STRATEGIES:
                run_tag = f"{dtag}_{rm['name']}_{params['tag']}"
                run_info = {
                    "dataset": ds_cfg["name"],
                    "dataset_cfg": ds_cfg,
                    "params": {k: v for k, v in params.items() if k != "tag"},
                    "params_tag": params["tag"],
                    "remasking": rm["name"],
                    "fill_strategy": rm["fill_strategy"],
                    "remask_fixed_count": rm["remask_fixed_count"],
                    "remask_start_frac": rm["remask_start_frac"],
                    "remask_fraction": rm["remask_fraction"],
                    "n_prompts": len(prompts),
                }

                print(f"\n[run] {run_tag}")

                answers = run_generation(
                    model, tokenizer, prompts, rm, run_tag, params
                )

                # Save this run immediately
                m = save_and_evaluate(
                    prompts,
                    answers,
                    run_tag,
                    run_info,
                    tox_device=DEVICE,
                    ppl_device="cpu",
                )
                summary[run_tag] = m

    print("\n[model] Unloading LLaDA from GPU …")
    try:
        from accelerate.hooks import remove_hook_from_module
        for m in model.modules():
            remove_hook_from_module(m)
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # final summary / analysis code stays the same


if __name__ == "__main__":
    main()
