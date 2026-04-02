"""
main_sentiment.py – Sentiment generation sweep pipeline.

Goal: test whether LLaDA models can be steered toward generating
negative-sentiment continuations when prompted with already-negative
Amazon review prefixes.

Mirrors main.py in structure; replaces toxicity with sentiment:
  * Dataset:   fancyzhx/amazon_polarity – negative reviews (label=0),
               first N_PROMPT_WORDS words used as the prompt
  * Prompt filter: keep only truncated prompts whose sentiment-model
                   negativity score is >= PROMPT_NEGATIVITY_PREFILTER
  * Models:    LLaDA-8B-Base     (plain-text prompts)
               LLaDA-8B-Instruct (chat-template prompts)
               → both tested in the same sweep loop
  * Param sets + Remasking strategies: configurable grid (see below)

For each (model × param_set × remasking):
  - Generate N_PER_RUN continuations of prefiltered negative prompts
  - Evaluate P(NEGATIVE) of answer only and of prompt + answer
    using distilbert-base-uncased-finetuned-sst-2-english
  - Evaluate perplexity of answer under Qwen2.5-3B

Outputs per run: data/generations_sentiment/<run_tag>/
  run_info.json           – model, params, remasking config
  generations.jsonl       – one JSON per line with all per-item scores
  sentiment_answer.json   – aggregate P(NEGATIVE) for answer only
  sentiment_combined.json – aggregate P(NEGATIVE) for prompt + answer
  perplexity.json         – aggregate perplexity scores
  metrics.json            – full summary

data/generations_sentiment/summary.json – combined table + analysis
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

# Must be set before any CUDA initialisation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Bypass torch.load security check for trusted HuggingFace models.
# distilbert classifiers may use .bin format.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from remasking.remdm_conf import ReMDMConfRemasking
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_fluency_metrics


# ── Sweep configuration ───────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_RUN = 40
BATCH_SIZE = 1
N_PROMPT_WORDS = 20

# Filter the BASE PROMPTS before generation.
PROMPT_NEGATIVITY_PREFILTER = 0.8

# Load more raw candidates than needed, because many may fail the prompt filter.
PROMPT_CANDIDATE_MULTIPLIER = 5

MODELS = [
    dict(name="GSAI-ML/LLaDA-8B-Base", tag="base", use_chat_tpl=False),
    dict(name="GSAI-ML/LLaDA-8B-Instruct", tag="instruct", use_chat_tpl=True),
]

PARAM_SETS = [
    dict(tag="T0.0_S40_G40", temperature=0.0, steps=30, gen_length=30, block_length=30),
    dict(tag="T0.2_S40_G40", temperature=0.0, steps=20, gen_length=30, block_length=30),
    dict(tag="T0.2_S40_G40", temperature=0.2, steps=50, gen_length=50, block_length=50),
    dict(tag="T0.2_S40_G40", temperature=0.8, steps=25, gen_length=50, block_length=50)
]

DATASETS = [
    dict(name="amazon_polarity"),
]

REMASK_STRATEGIES = [
    dict(
        name="low_conf",
        fill_strategy="low_confidence",
        remasking_cls=None,
        remask_fraction=0.0,
        remask_fixed_count=None,
        remask_start_frac=0.0,
    ),
    dict(
        name="remdm_k2_late75",
        fill_strategy="low_confidence",
        remasking_cls=ReMDMConfRemasking,
        remask_fraction=0.0,
        remask_fixed_count=2,
        remask_start_frac=0.75,
    ),
]

OUT_DIR = Path("data/generations_sentiment")


# ── Data loading / prompt filtering ───────────────────────────────────────────

def _truncate(text: str, n_words: int) -> str:
    """Return the first n_words space-separated tokens."""
    return " ".join(text.split()[:n_words]).strip()


def load_amazon_polarity_negative_prompts(n_candidates: int) -> list[str]:
    """
    Load candidate prompts from NEGATIVE Amazon Polarity reviews (label=0).
    Each prompt is the first N_PROMPT_WORDS words of the review content.
    """
    print("[data] Loading Amazon Polarity candidate prompts (label=0, negative) …")
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

    prompts: list[str] = []
    for ex in ds:
        if ex["label"] != 0:
            continue

        text = ex.get("content") or ex.get("text") or (
            f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        )
        truncated = _truncate(text, N_PROMPT_WORDS)

        if len(truncated.split()) >= 5:
            prompts.append(truncated)

        if len(prompts) >= n_candidates:
            break

    print(
        f"[data] Loaded {len(prompts)} negative candidate prompts "
        f"(first {N_PROMPT_WORDS} words of each review)."
    )
    return prompts


def filter_prompts_by_negativity(
    prompts: list[str],
    threshold: float,
    device: str = "cuda",
) -> tuple[list[str], dict]:
    """
    Keep only prompts whose negativity score is >= threshold.
    The sentiment score is computed on the BASE PROMPT text itself.
    """
    print(f"[filter] Scoring prompt negativity (threshold={threshold}) …")

    if not prompts:
        stats = {
            "threshold": threshold,
            "num_before": 0,
            "num_after": 0,
            "kept_fraction": 0.0,
            "kept_indices": [],
            "all_scores": [],
        }
        return [], stats

    sent = compute_sentiment_metrics(prompts, device=device)
    scores = sent["scores"]

    keep_idx = [i for i, s in enumerate(scores) if float(s) >= threshold]
    filtered_prompts = [prompts[i] for i in keep_idx]

    stats = {
        "threshold": threshold,
        "num_before": len(prompts),
        "num_after": len(filtered_prompts),
        "kept_fraction": len(filtered_prompts) / len(prompts),
    }

    print(
        f"[filter] kept {stats['num_after']}/{stats['num_before']} prompts "
        f"({stats['kept_fraction']:.2%}) with negativity >= {threshold}"
    )

    return filtered_prompts, stats


def load_prompts_for_dataset(cfg: dict, n_final: int, device: str) -> tuple[list[str], dict]:
    """
    Load a pool of candidate prompts, filter them by prompt negativity,
    and return up to n_final prompts plus filter stats.
    """
    if cfg["name"] != "amazon_polarity":
        raise ValueError(f"Unknown dataset: {cfg['name']}")

    n_candidates = max(n_final * PROMPT_CANDIDATE_MULTIPLIER, n_final)
    raw_prompts = load_amazon_polarity_negative_prompts(n_candidates=n_candidates)

    filtered_prompts, filter_stats = filter_prompts_by_negativity(
        raw_prompts,
        threshold=PROMPT_NEGATIVITY_PREFILTER,
        device=device,
    )

    final_prompts = filtered_prompts[:n_final]

    filter_stats = {
        **filter_stats,
        "num_used_for_generation": len(final_prompts),
        "candidate_multiplier": PROMPT_CANDIDATE_MULTIPLIER,
    }

    print(f"[data] Using {len(final_prompts)} prompts for generation.")
    return final_prompts, filter_stats


def dataset_tag(cfg: dict) -> str:
    return cfg["name"]


# ── Generation ────────────────────────────────────────────────────────────────

def run_generation(
    model,
    tokenizer,
    prompts: list[str],
    strategy: dict,
    run_tag: str,
    params: dict,
    use_chat_tpl: bool,
) -> list[str]:
    """
    Generate continuations for `prompts`.

    For the Instruct model, each prompt is wrapped in a chat message asking to
    continue the product review. For the Base model, the prompt is passed as-is.
    """
    all_answers: list[str] = []

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

        if hasattr(remasking_obj, "reset"):
            remasking_obj.reset()

        if use_chat_tpl:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"Continue the following product review: {p}"}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in batch_prompts
            ]
        else:
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


# ── Evaluation / saving ───────────────────────────────────────────────────────

def _sent_dict(result: dict) -> dict:
    return {
        "mean_negative": result["mean_negative"],
        "negative_fraction": result["negative_fraction"],
        "max_negative": result["max_negative"],
        "per_text_scores": result["scores"].tolist(),
    }


def save_and_evaluate(
    prompts: list[str],
    answers: list[str],
    run_tag: str,
    run_info: dict,
    sent_device: str = "cuda",
    ppl_device: str = "cpu",
) -> dict:
    """
    Save generations and evaluate:
      - sentiment of answer only
      - sentiment of prompt + answer
      - perplexity of answer only
    """
    out_dir = OUT_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    combined_texts = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]

    print(f"[eval] Sentiment (answer only) [{run_tag}] …")
    sent_ans = compute_sentiment_metrics(answers, device=sent_device)
    with open(out_dir / "sentiment_answer.json", "w") as f:
        json.dump(_sent_dict(sent_ans), f, indent=2)

    print(f"[eval] Sentiment (prompt+answer) [{run_tag}] …")
    sent_comb = compute_sentiment_metrics(combined_texts, device=sent_device)
    with open(out_dir / "sentiment_combined.json", "w") as f:
        json.dump(_sent_dict(sent_comb), f, indent=2)

    print(f"[eval] Perplexity (answer) [{run_tag}] …")
    flu = compute_fluency_metrics(answers, device=ppl_device)
    with open(out_dir / "perplexity.json", "w") as f:
        json.dump(
            {
                "mean_ppl": flu["mean_ppl"],
                "per_text_ppl": flu["per_text_ppl"],
            },
            f,
            indent=2,
        )

    gen_path = out_dir / "generations.jsonl"
    with open(gen_path, "w") as f:
        for i, (prompt, answer, combined) in enumerate(zip(prompts, answers, combined_texts)):
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "answer": answer,
                        "prompt_answer": combined,
                        "sent_answer": float(sent_ans["scores"][i]),
                        "sent_combined": float(sent_comb["scores"][i]),
                        "ppl_answer": float(flu["per_text_ppl"][i]),
                    }
                )
                + "\n"
            )
    print(f"[output] {len(answers)} generations → {gen_path}")

    metrics = {
        "run_tag": run_tag,
        "num_samples": len(answers),
        "run_info": run_info,
        "sent_answer_mean": sent_ans["mean_negative"],
        "sent_answer_fraction": sent_ans["negative_fraction"],
        "sent_answer_max": sent_ans["max_negative"],
        "sent_combined_mean": sent_comb["mean_negative"],
        "sent_combined_fraction": sent_comb["negative_fraction"],
        "sent_combined_max": sent_comb["max_negative"],
        "ppl_answer_mean": flu["mean_ppl"],
        "mean_length_words": flu["mean_length_words"],
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[output] metrics → {out_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    print("\n[data] Pre-loading and prefiltering prompts for all datasets …")
    dataset_prompts: dict[str, list[str]] = {}
    dataset_prompt_filter_stats: dict[str, dict] = {}

    for ds_cfg in DATASETS:
        dtag = dataset_tag(ds_cfg)
        prompts, filter_stats = load_prompts_for_dataset(
            ds_cfg,
            n_final=N_PER_RUN,
            device=DEVICE,
        )
        dataset_prompts[dtag] = prompts
        dataset_prompt_filter_stats[dtag] = filter_stats

    summary: dict[str, dict] = {}

    print("\n" + "=" * 60)
    print("GENERATION + EVALUATION (sentiment sweep)")
    print("=" * 60)

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        model_tag = model_cfg["tag"]
        use_chat_tpl = model_cfg["use_chat_tpl"]

        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 60}")

        model = load_model(model_name=model_name, device=DEVICE)
        tokenizer = load_tokenizer(model_name=model_name)

        for ds_cfg in DATASETS:
            dtag = dataset_tag(ds_cfg)
            prompts = dataset_prompts[dtag]

            if len(prompts) == 0:
                print(f"[run] No prompts available for dataset={dtag}; skipping.")
                continue

            for params in PARAM_SETS:
                for rm in REMASK_STRATEGIES:
                    run_tag = f"{model_tag}_{dtag}_{rm['name']}_{params['tag']}"

                    run_info = {
                        "model": model_name,
                        "model_tag": model_tag,
                        "use_chat_tpl": use_chat_tpl,
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
                        "prompt_negativity_prefilter": dataset_prompt_filter_stats[dtag],
                    }

                    print(f"\n[run] {run_tag}")

                    answers = run_generation(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        strategy=rm,
                        run_tag=run_tag,
                        params=params,
                        use_chat_tpl=use_chat_tpl,
                    )

                    m = save_and_evaluate(
                        prompts=prompts,
                        answers=answers,
                        run_tag=run_tag,
                        run_info=run_info,
                        sent_device=DEVICE,
                        ppl_device="cpu",
                    )
                    summary[run_tag] = m

        print(f"\n[model] Unloading {model_name} from GPU …")
        try:
            from accelerate.hooks import remove_hook_from_module
            for mod in model.modules():
                remove_hook_from_module(mod)
        except Exception:
            pass

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    header = (
        f"{'Run':<58} {'AnsS_mean':>9} {'CombS_mean':>10} "
        f"{'AnsS_frac':>9} {'PPL_mean':>8}"
    )
    print(header)
    print("-" * 98)

    for run_tag, m in summary.items():
        print(
            f"{run_tag:<58} "
            f"{m['sent_answer_mean']:>9.4f} "
            f"{m['sent_combined_mean']:>10.4f} "
            f"{m['sent_answer_fraction']:>9.4f} "
            f"{m['ppl_answer_mean']:>8.1f}"
        )

    print("\n[analysis] Best configs per (model × dataset):")
    groups: dict[str, dict[str, dict]] = {}
    for run_tag, m in summary.items():
        key = f"{m['run_info']['model_tag']}_{m['run_info']['dataset']}"
        groups.setdefault(key, {})[run_tag] = m

    for key, runs in sorted(groups.items()):
        most_neg = max(runs, key=lambda t: runs[t]["sent_answer_mean"])
        least_neg = min(runs, key=lambda t: runs[t]["sent_answer_mean"])
        best_ppl = min(runs, key=lambda t: runs[t]["ppl_answer_mean"])

        print(
            f"  {key}:  most_neg={most_neg} ({runs[most_neg]['sent_answer_mean']:.4f})"
            f"  least_neg={least_neg} ({runs[least_neg]['sent_answer_mean']:.4f})"
            f"  best_ppl={best_ppl} ({runs[best_ppl]['ppl_answer_mean']:.1f})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"runs": summary}, f, indent=2)

    print(f"\n[output] Combined summary → {summary_path}")


if __name__ == "__main__":
    main()