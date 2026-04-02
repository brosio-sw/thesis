"""
liseco_test.py – LiSeCo-style probe control experiment for LLaDA.

This version evaluates LiSeCo steering on a configurable list of layer groups.

Examples:
    LAYER_GROUP_SPECS = ["all", [23]]
    LAYER_GROUP_SPECS = [[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
    LAYER_GROUP_SPECS = ["all", [23], [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]

All selected settings use the same number of prompts in a given run mode.

Evaluation note
---------------
Degenerate generations are filtered before sentiment/perplexity evaluation.
We report the fraction of invalid generations separately, and compute
sentiment/perplexity only on valid generations.

A generation is marked invalid if it matches one of the known bad-template
patterns or contains obvious repetition loops.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity
from steering.liseco_probe_steering import LiSeCoProbeSteering, load_probe_params


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SMOKE_TEST: bool = False

MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROBES_ROOT_CANDIDATES = [
    Path("data/probes_llada_masked/full_run"),
    Path("data/probes_llada_masked/smoke_test"),
    Path("data/probes_llada/full_run"),
    Path("data/probes_llada/smoke_test"),
]
PROBE_FAMILY = "masked_only_probes"

OUT_DIR = Path("data/liseco_indexed_eval") / ("smoke_test" if SMOKE_TEST else "full_run")

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",
)

EVAL_START_IDX = 20_000
EVAL_END_IDX = 20_500
RAW_TEXT_WORDS = 50
PROMPT_WORDS = 20

FULL_INTERVALS = [
    (0.00, 0.10),
    (0.20, 0.30),
    (0.40, 0.50),
    (0.60, 0.70),
    (0.80, 0.90),
]

SMOKE_INTERVALS = [
    (0.00, 0.10),
    (0.40, 0.50),
]

SMOKE_N_PROMPTS = 5
FULL_N_PROMPTS = 250

# Each entry can be:
#   - "all"
#   - [23]
#   - [9,10,11,12,...]
LAYER_GROUP_SPECS = [
    [i for i in range(9, 25)],  # layers 9-23 inclusive
]

BAD_PATTERNS = [
    r"Is this .*?\?",
    r"positive or negative\?",
    r"\bAnswer:",
    r"\bYesTitle:",
    r"\bNoTitle:",
    r"\bTitle:",
    r"\bReview:",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()


def find_probes_root() -> Path:
    for root in PROBES_ROOT_CANDIDATES:
        if (root / PROBE_FAMILY).exists():
            return root
    raise FileNotFoundError(
        "Could not find probe root. Expected one of: "
        + ", ".join(str(x) for x in PROBES_ROOT_CANDIDATES)
    )


def infer_layers_from_probe_dir(probes_root: Path, family: str) -> list[int]:
    family_dir = probes_root / family
    layers = []
    for d in sorted(family_dir.glob("layer_*")):
        try:
            layers.append(int(d.name.split("_")[1]))
        except Exception:
            continue
    if not layers:
        raise RuntimeError(f"No layer_* directories found in {family_dir}")
    return layers


def canonical_layer_group_name(layer_ids: list[int], all_probe_layers: list[int]) -> str:
    if sorted(layer_ids) == sorted(all_probe_layers):
        return "all_layers"
    if layer_ids == [23]:
        return "layer23"
    return "layers_" + "-".join(str(x) for x in layer_ids)


def resolve_layer_group_specs(
    layer_group_specs: list,
    all_probe_layers: list[int],
) -> list[dict]:
    resolved = []
    seen = set()

    for spec in layer_group_specs:
        if spec == "all":
            layer_ids = list(all_probe_layers)
        elif isinstance(spec, list) and len(spec) > 0:
            layer_ids = sorted({int(x) for x in spec})
            missing = [x for x in layer_ids if x not in all_probe_layers]
            if missing:
                raise RuntimeError(
                    f"Requested layer group {layer_ids}, but missing probes for layers: {missing}"
                )
        else:
            raise RuntimeError(
                f"Invalid LAYER_GROUP_SPECS entry: {spec!r}. "
                f"Use 'all' or a non-empty list of ints."
            )

        key = tuple(layer_ids)
        if key in seen:
            continue
        seen.add(key)

        resolved.append(
            {
                "config_name": canonical_layer_group_name(layer_ids, all_probe_layers),
                "layer_ids": layer_ids,
            }
        )

    if not resolved:
        raise RuntimeError("No valid layer groups resolved from LAYER_GROUP_SPECS.")

    return resolved


def load_indexed_eval_texts(
    start_idx: int,
    end_idx: int,
    raw_words: int,
    prompt_words: int,
    n_limit: int | None = None,
) -> tuple[list[str], list[str], dict]:
    split = f"train[{start_idx}:{end_idx}]"
    ds = load_dataset("fancyzhx/amazon_polarity", split=split)

    raw_texts: list[str] = []
    prompt_texts: list[str] = []
    labels: list[int] = []

    for ex in ds:
        text = ex.get("content") or ex.get("text") or (
            f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        )
        raw = _truncate(text, raw_words)
        prompt = _truncate(text, prompt_words)
        if len(prompt.split()) < 5:
            continue
        raw_texts.append(raw)
        prompt_texts.append(prompt)
        labels.append(int(ex["label"]))

    if n_limit is not None:
        raw_texts = raw_texts[:n_limit]
        prompt_texts = prompt_texts[:n_limit]
        labels = labels[:n_limit]

    meta = {
        "dataset": "fancyzhx/amazon_polarity",
        "split": split,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_loaded": len(raw_texts),
        "raw_text_words": raw_words,
        "prompt_words": prompt_words,
        "label_counts": {
            "negative_0": int(sum(1 for x in labels if x == 0)),
            "positive_1": int(sum(1 for x in labels if x == 1)),
        },
    }
    return raw_texts, prompt_texts, meta


def compute_text_metrics(texts: list[str], device_for_sentiment: str) -> dict:
    if len(texts) == 0:
        return {
            "sentiment": {
                "mean_negative": None,
                "negative_fraction": None,
                "per_text": [],
            },
            "perplexity": {
                "mean_ppl": None,
                "per_text": [],
            },
        }

    sent = compute_sentiment_metrics(texts, device=device_for_sentiment)
    ppl = compute_perplexity(texts, device="cpu")
    return {
        "sentiment": {
            "mean_negative": sent["mean_negative"],
            "negative_fraction": sent["negative_fraction"],
            "per_text": sent["scores"].tolist(),
        },
        "perplexity": {
            "mean_ppl": ppl["mean_ppl"],
            "per_text": ppl["per_text_ppl"],
        },
    }


def save_eval_set(
    raw_texts: list[str],
    prompt_texts: list[str],
    meta: dict,
) -> dict:
    eval_dir = OUT_DIR / "eval_set"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("[eval_set] Computing baseline metrics for raw texts …")
    raw_metrics = compute_text_metrics(raw_texts, device_for_sentiment=DEVICE)

    print("[eval_set] Computing baseline metrics for prompt texts …")
    prompt_metrics = compute_text_metrics(prompt_texts, device_for_sentiment=DEVICE)

    payload = {
        "meta": meta,
        "raw_texts": raw_texts,
        "prompt_texts": prompt_texts,
        "baseline_raw": raw_metrics,
        "baseline_prompt": prompt_metrics,
    }

    with open(eval_dir / "eval_set.json", "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def has_bad_pattern(text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in BAD_PATTERNS)


def has_repetition_loop(text: str) -> bool:
    toks = text.split()
    if len(toks) < 12:
        return False

    for n in [2, 3, 4]:
        for i in range(len(toks) - 2 * n + 1):
            if toks[i:i+n] == toks[i+n:i+2*n]:
                return True
    return False


def classify_generation_quality(answer: str) -> dict:
    bad_pattern = has_bad_pattern(answer)
    repetition = has_repetition_loop(answer)
    empty = not answer.strip()
    is_valid = not (bad_pattern or repetition or empty)

    return {
        "is_valid": is_valid,
        "has_bad_pattern": bad_pattern,
        "has_repetition_loop": repetition,
        "is_empty": empty,
    }


def generate_with_liseco(
    model,
    tokenizer,
    prompts: list[str],
    steerer: LiSeCoProbeSteering,
    desc: str,
) -> list[str]:
    answers: list[str] = []
    for prompt in tqdm(prompts, desc=desc, leave=False):
        enc = tokenizer(
            [prompt],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(DEVICE)

        out = llada_generate(
            model=model,
            prompt=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            steps=GEN_PARAMS["steps"],
            gen_length=GEN_PARAMS["gen_length"],
            block_length=GEN_PARAMS["block_length"],
            temperature=GEN_PARAMS["temperature"],
            fill_strategy=GEN_PARAMS["fill_strategy"],
            remasking=None,
            remask_fraction=0.0,
            remask_fixed_count=None,
            remask_start_frac=0.0,
            steering=steerer,
            show_progress=False,
        )

        generated_ids = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers.extend(decoded)

    return answers


def save_and_evaluate(
    prompts: list[str],
    answers: list[str],
    run_tag: str,
    run_info: dict,
) -> dict:
    out_dir = OUT_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]
    quality = [classify_generation_quality(a) for a in answers]

    valid_indices = [i for i, q in enumerate(quality) if q["is_valid"]]
    invalid_indices = [i for i, q in enumerate(quality) if not q["is_valid"]]

    valid_answers = [answers[i] for i in valid_indices]
    valid_combined = [combined[i] for i in valid_indices]

    n_total = len(answers)
    n_valid = len(valid_indices)
    n_invalid = len(invalid_indices)
    invalid_fraction = float(n_invalid / max(1, n_total))

    bad_pattern_fraction = float(sum(q["has_bad_pattern"] for q in quality) / max(1, n_total))
    repetition_fraction = float(sum(q["has_repetition_loop"] for q in quality) / max(1, n_total))
    empty_fraction = float(sum(q["is_empty"] for q in quality) / max(1, n_total))

    print(f"  [eval] valid={n_valid}/{n_total} invalid={n_invalid}/{n_total}")

    if n_valid > 0:
        print("  [eval] Sentiment (answer, valid only) …")
        sent_ans = compute_sentiment_metrics(valid_answers, device=DEVICE)
        print("  [eval] Sentiment (combined, valid only) …")
        sent_comb = compute_sentiment_metrics(valid_combined, device=DEVICE)

        print("  [eval] Perplexity (answer, valid only) …")
        ppl_ans = compute_perplexity(valid_answers, device="cpu")
        print("  [eval] Perplexity (combined, valid only) …")
        ppl_comb = compute_perplexity(valid_combined, device="cpu")

        sent_ans_scores = {idx: float(score) for idx, score in zip(valid_indices, sent_ans["scores"])}
        sent_comb_scores = {idx: float(score) for idx, score in zip(valid_indices, sent_comb["scores"])}
        ppl_ans_scores = {idx: float(score) for idx, score in zip(valid_indices, ppl_ans["per_text_ppl"])}
        ppl_comb_scores = {idx: float(score) for idx, score in zip(valid_indices, ppl_comb["per_text_ppl"])}
    else:
        sent_ans = {"mean_negative": None, "negative_fraction": None, "scores": []}
        sent_comb = {"mean_negative": None, "negative_fraction": None, "scores": []}
        ppl_ans = {"mean_ppl": None, "per_text_ppl": []}
        ppl_comb = {"mean_ppl": None, "per_text_ppl": []}
        sent_ans_scores = {}
        sent_comb_scores = {}
        ppl_ans_scores = {}
        ppl_comb_scores = {}

    with open(out_dir / "generations.jsonl", "w") as f:
        for i in range(len(prompts)):
            row = {
                "prompt": prompts[i],
                "answer": answers[i],
                "combined": combined[i],
                "is_valid": quality[i]["is_valid"],
                "has_bad_pattern": quality[i]["has_bad_pattern"],
                "has_repetition_loop": quality[i]["has_repetition_loop"],
                "is_empty": quality[i]["is_empty"],
                "sent_answer": sent_ans_scores.get(i),
                "sent_combined": sent_comb_scores.get(i),
                "ppl_answer": ppl_ans_scores.get(i),
                "ppl_combined": ppl_comb_scores.get(i),
            }
            f.write(json.dumps(row) + "\n")

    with open(out_dir / "sentiment.json", "w") as f:
        json.dump(
            {
                "computed_on": "valid_generations_only",
                "n_total": n_total,
                "n_valid": n_valid,
                "n_invalid": n_invalid,
                "invalid_fraction": invalid_fraction,
                "answer": {
                    "mean_negative": sent_ans["mean_negative"],
                    "negative_fraction": sent_ans["negative_fraction"],
                    "per_text_valid_only": list(sent_ans_scores.values()),
                },
                "combined": {
                    "mean_negative": sent_comb["mean_negative"],
                    "negative_fraction": sent_comb["negative_fraction"],
                    "per_text_valid_only": list(sent_comb_scores.values()),
                },
            },
            f,
            indent=2,
        )

    with open(out_dir / "perplexity.json", "w") as f:
        json.dump(
            {
                "computed_on": "valid_generations_only",
                "n_total": n_total,
                "n_valid": n_valid,
                "n_invalid": n_invalid,
                "invalid_fraction": invalid_fraction,
                "answer": {
                    "mean_ppl": ppl_ans["mean_ppl"],
                    "per_text_valid_only": list(ppl_ans_scores.values()),
                },
                "combined": {
                    "mean_ppl": ppl_comb["mean_ppl"],
                    "per_text_valid_only": list(ppl_comb_scores.values()),
                },
            },
            f,
            indent=2,
        )

    mean_ans_words_valid = (
        float(sum(len(a.split()) for a in valid_answers) / max(1, len(valid_answers)))
        if n_valid > 0 else None
    )

    metrics = {
        "run_tag": run_tag,
        "num_samples_total": n_total,
        "num_samples_valid": n_valid,
        "num_samples_invalid": n_invalid,
        "invalid_fraction": invalid_fraction,
        "bad_pattern_fraction": bad_pattern_fraction,
        "repetition_fraction": repetition_fraction,
        "empty_fraction": empty_fraction,
        "run_info": run_info,
        "sent_answer_mean": sent_ans["mean_negative"],
        "sent_answer_fraction": sent_ans["negative_fraction"],
        "sent_combined_mean": sent_comb["mean_negative"],
        "ppl_answer_mean": ppl_ans["mean_ppl"],
        "ppl_combined_mean": ppl_comb["mean_ppl"],
        "mean_answer_words_valid_only": mean_ans_words_valid,
        "evaluation_subset": "valid_generations_only",
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(
        f"  [done] valid={n_valid}/{n_total} "
        f"invalid_frac={invalid_fraction:.3f} "
        f"sent_ans={metrics['sent_answer_mean'] if metrics['sent_answer_mean'] is not None else 'NA'} "
        f"ppl_ans={metrics['ppl_answer_mean'] if metrics['ppl_answer_mean'] is not None else 'NA'}"
    )
    return metrics


def make_run_tag(config_name: str, n_prompts: int, alpha_min: float, alpha_max: float) -> str:
    base = (
        f"liseco__{config_name}"
        f"__n{n_prompts}"
        f"__amin{alpha_min:.2f}__amax{alpha_max:.2f}"
    )
    return f"clean__{base}"


def main():
    torch.manual_seed(42)

    intervals = SMOKE_INTERVALS if SMOKE_TEST else FULL_INTERVALS
    n_prompts = SMOKE_N_PROMPTS if SMOKE_TEST else FULL_N_PROMPTS
    mode = "SMOKE TEST" if SMOKE_TEST else "FULL RUN"

    probes_root = find_probes_root()
    all_probe_layers = infer_layers_from_probe_dir(probes_root, PROBE_FAMILY)

    config_specs = resolve_layer_group_specs(LAYER_GROUP_SPECS, all_probe_layers)

    # Preload probes for all requested groups
    for cfg in config_specs:
        cfg["probe_by_layer"] = load_probe_params(
            probes_root=probes_root,
            family=PROBE_FAMILY,
            layer_ids=cfg["layer_ids"],
        )

    print("[eval_set] Loading indexed evaluation slice …")
    raw_texts, prompt_texts, eval_meta = load_indexed_eval_texts(
        start_idx=EVAL_START_IDX,
        end_idx=EVAL_END_IDX,
        raw_words=RAW_TEXT_WORDS,
        prompt_words=PROMPT_WORDS,
        n_limit=n_prompts,
    )
    eval_payload = save_eval_set(raw_texts=raw_texts, prompt_texts=prompt_texts, meta=eval_meta)

    prompts = prompt_texts

    print(f"\n{'='*72}")
    print(f"LiSeCo EXPERIMENT — {mode}")
    print(f"intervals={len(intervals)}")
    print(f"probe_root={probes_root} family={PROBE_FAMILY}")
    print(f"available_probe_layers={all_probe_layers}")
    print(f"configs={[cfg['config_name'] for cfg in config_specs]}")
    print(f"n_prompts={len(prompts)}")
    print(f"eval_slice=train[{EVAL_START_IDX}:{EVAL_END_IDX}]")
    print(f"{'='*72}")

    print(f"\n[model] Loading {MODEL_NAME} …")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    new_summary: dict[str, dict] = {}
    total = len(intervals) * len(config_specs)
    step_idx = 0

    for cfg in config_specs:
        config_name = cfg["config_name"]
        layer_ids = cfg["layer_ids"]
        probe_by_layer = cfg["probe_by_layer"]

        for alpha_min, alpha_max in intervals:
            step_idx += 1
            run_tag = make_run_tag(config_name, len(prompts), alpha_min, alpha_max)

            print(f"\n[{step_idx}/{total}] {run_tag}")

            done = OUT_DIR / run_tag / "metrics.json"
            if done.exists():
                print(f"  [skip] Already computed: {done}")
                with open(done) as f:
                    new_summary[run_tag] = json.load(f)
                continue

            steerer = LiSeCoProbeSteering(
                probe_by_layer=probe_by_layer,
                layer_ids=layer_ids,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )

            answers = generate_with_liseco(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                steerer=steerer,
                desc=f"Generating [{run_tag}]",
            )

            run_info = {
                "model": MODEL_NAME,
                "gen_params": GEN_PARAMS,
                "liseco": {
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                    "layer_subset": layer_ids,
                    "layer_mode_name": config_name,
                    "probe_family": PROBE_FAMILY,
                    "probes_root": str(probes_root),
                    "control_scope": "every_denoising_step_every_selected_layer_currently_masked_tokens_only",
                    "formula": "project z=w^T x+b into [logit(alpha_min), logit(alpha_max)] via minimum-norm delta along w",
                },
                "hook_runtime_stats": {
                    "forward_calls": steerer.forward_calls,
                    "layer_hook_calls": steerer.layer_hook_calls,
                    "total_masked_positions_seen": steerer.total_masked_positions_seen,
                    "total_projected_positions": steerer.total_projected_positions,
                },
                "n_prompts": len(prompts),
                "smoke_test": SMOKE_TEST,
                "eval_set_meta": eval_payload["meta"],
                "baseline_prompt_sentiment_mean": eval_payload["baseline_prompt"]["sentiment"]["mean_negative"],
                "baseline_prompt_ppl_mean": eval_payload["baseline_prompt"]["perplexity"]["mean_ppl"],
                "prompt_subset": {
                    "config_name": config_name,
                    "n_prompts_used": len(prompts),
                    "uses_full_prompt_set": True,
                    "uses_first_k_only": None,
                },
                "evaluation_filtering": {
                    "bad_patterns": BAD_PATTERNS,
                    "repetition_loop_filter": True,
                    "metrics_computed_on": "valid_generations_only",
                },
            }

            metrics = save_and_evaluate(
                prompts=prompts,
                answers=answers,
                run_tag=run_tag,
                run_info=run_info,
            )
            new_summary[run_tag] = metrics

    summary_path = OUT_DIR / "summary.json"
    existing_runs: dict[str, dict] = {}
    existing_configs: list[dict] = []

    if summary_path.exists():
        with open(summary_path) as f:
            existing_payload = json.load(f)
        existing_runs = dict(existing_payload.get("runs", {}))
        existing_configs = list(existing_payload.get("evaluated_configs", []))

    merged_runs = dict(existing_runs)
    merged_runs.update(new_summary)

    seen_keys = {
        (cfg.get("name"), tuple(cfg.get("layer_ids", [])), cfg.get("n_prompts"))
        for cfg in existing_configs
    }

    for cfg in config_specs:
        config_entry = {
            "name": cfg["config_name"],
            "layer_ids": cfg["layer_ids"],
            "n_prompts": len(prompts),
        }
        config_key = (config_entry["name"], tuple(config_entry["layer_ids"]), config_entry["n_prompts"])
        if config_key not in seen_keys:
            existing_configs.append(config_entry)
            seen_keys.add(config_key)

    merged_payload = {
        "runs": merged_runs,
        "mode": mode,
        "eval_meta": eval_payload["meta"],
        "probe_root": str(probes_root),
        "probe_family": PROBE_FAMILY,
        "available_probe_layers": all_probe_layers,
        "evaluated_configs": existing_configs,
    }

    with open(summary_path, "w") as f:
        json.dump(merged_payload, f, indent=2)

    print(f"\n{'='*80}")
    print(f"LiSeCo RESULTS — {mode}")
    print(f"{'='*80}")
    print(f"{'Run tag':<80} {'ValidFrac':>9} {'SentAns':>8} {'SentComb':>8} {'PPL':>7}")
    print("-" * 120)
    for tag, m in merged_runs.items():
        valid_frac = (
            1.0 - float(m["invalid_fraction"])
            if m.get("invalid_fraction") is not None else float("nan")
        )
        print(
            f"{tag:<80} "
            f"{valid_frac:>9.3f} "
            f"{m.get('sent_answer_mean', float('nan')) if m.get('sent_answer_mean') is not None else float('nan'):>8.4f} "
            f"{m.get('sent_combined_mean', float('nan')) if m.get('sent_combined_mean') is not None else float('nan'):>8.4f} "
            f"{m.get('ppl_answer_mean', float('nan')) if m.get('ppl_answer_mean') is not None else float('nan'):>7.1f}"
        )

    print(f"\n[output] Summary → {summary_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()