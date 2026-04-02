from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activations_modeling.gaussian.gaussian_models import fit_or_load_gaussian_models
from eval.fluency_metrics import compute_perplexity
from eval.sentiment_metrics import compute_sentiment_metrics
from fill_scoring.h2_gaussian_fill import H2GaussianFillScorer
from llada.generate import generate as llada_generate
from llada.model import load_model, load_tokenizer
from steering.precomputed_steering import HiddenStateCaptureSteering, SteeringHiddenStateBuffer

SMOKE_TEST = False

DIFFMEAN_OUT_ROOT = Path("data/liseco_vs_diffmean_alignment/full_run")
DIFFMEAN_VARIANT = "real_full_pooled"
OUT_DIR = Path("data/diffmean_h2_gaussian") / (
    "smoke_test" if SMOKE_TEST else "full_run"
)

MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

STEER_LAYERS = list(range(1, 31))
# Memory-safe default mirrors the SAE test's monitored subset.
# If needed, switch to all layers with: list(range(1, 31)).
MONITORED_GAUSS_LAYERS =  list(range(1, 31))
GAUSS_CACHE_PATH = Path("data/activations_modelling/gaussian/monitored_layer_diag_gaussians.pt")

DIFFMEAN_LAMBDAS = [4.0, 12.0, 18.0]
ALL_ALPHA_MIX = [1.0, 0.875, 0.75, 0.5, 0.25, 0.0]

SMOKE_LAMBDAS = [4.0]
SMOKE_ALPHA_MIX = [1.0, 0.0]
SMOKE_MAX_PROMPTS = 5

EVAL_START_IDX = 50_000
EVAL_END_IDX = 50_250
PROMPT_WORDS = 20
N_DEBUG_EXAMPLES = 2

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",
)

BAD_PATTERNS = [
    r"Is this .*?\\?",
    r"positive or negative\\?",
    r"\\bAnswer:",
    r"\\bYesTitle:",
    r"\\bNoTitle:",
    r"\\bTitle:",
    r"\\bReview:",
]


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def has_bad_pattern(text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in BAD_PATTERNS)


def has_repetition_loop(text: str) -> bool:
    toks = text.split()
    if len(toks) < 12:
        return False
    for n in [2, 3, 4]:
        for i in range(len(toks) - 2 * n + 1):
            if toks[i:i + n] == toks[i + n:i + 2 * n]:
                return True
    return False


def classify_generation_quality(answer: str) -> dict[str, bool]:
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


def load_holdout_prompts(
    start_idx: int = EVAL_START_IDX,
    end_idx: int = EVAL_END_IDX,
    prompt_words: int = PROMPT_WORDS,
) -> tuple[list[str], list[int], dict[str, Any]]:
    split = f"train[{start_idx}:{end_idx}]"
    ds = load_dataset("fancyzhx/amazon_polarity", split=split)

    prompts: list[str] = []
    labels: list[int] = []
    for ex in ds:
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        prompt = _truncate(text, prompt_words)
        if len(prompt.split()) < 5:
            continue
        prompts.append(prompt)
        labels.append(int(ex["label"]))

    meta = {
        "dataset": "fancyzhx/amazon_polarity",
        "split": split,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "prompt_words": prompt_words,
        "n_prompts": len(prompts),
    }
    return prompts, labels, meta


def load_mean_vectors(probe_root: Path, variant: str, layer_ids: list[int]) -> dict[int, torch.Tensor]:
    candidate_paths = [
        probe_root / variant / "mean_diff_vectors.pt",
        Path("data/liseco_vs_diffmean_alignment/full_run/probes") / variant / "mean_diff_vectors.pt",
        Path("data/alignment_variants_v4/full_run/probes") / variant / "mean_diff_vectors.pt",
    ]
    p = next((cp for cp in candidate_paths if cp.exists()), None)
    if p is None:
        raise FileNotFoundError(
            "Could not find mean_diff_vectors.pt for variant "
            f"{variant}. Tried: {[str(cp) for cp in candidate_paths]}"
        )
    obj = torch.load(p, map_location="cpu", weights_only=False)
    out: dict[int, torch.Tensor] = {}
    for k, v in obj.items():
        li = int(k)
        if li in layer_ids:
            out[li] = v.float().cpu()
    missing = [li for li in layer_ids if li not in out]
    if missing:
        raise RuntimeError(f"Missing layers {missing} in {p}")
    return out


def evaluate_texts_filtered(prompts: list[str], answers: list[str]) -> dict[str, Any]:
    combined = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]
    quality = [classify_generation_quality(a) for a in answers]
    valid_idx = [i for i, q in enumerate(quality) if q["is_valid"]]

    valid_answers = [answers[i] for i in valid_idx]
    valid_combined = [combined[i] for i in valid_idx]

    n_total = len(answers)
    n_valid = len(valid_idx)
    n_invalid = n_total - n_valid

    if n_valid > 0:
        # Keep sentiment model off GPU so generation does not OOM between runs.
        sent_ans = compute_sentiment_metrics(valid_answers, device="cpu")
        ppl_ans = compute_perplexity(valid_answers, device="cpu")
        sent_mean = sent_ans["mean_negative"]
        ppl_mean = ppl_ans["mean_ppl"]
    else:
        sent_mean = None
        ppl_mean = None

    bad_pattern_count = int(sum(q["has_bad_pattern"] for q in quality))
    metrics = {
        "evaluation_subset": "valid_generations_only",
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_fraction": float(n_invalid / max(1, n_total)),
        "invalid_percent": float(100.0 * n_invalid / max(1, n_total)),
        "bad_pattern_count": bad_pattern_count,
        "bad_pattern_percent": float(100.0 * bad_pattern_count / max(1, n_total)),
        "sent_answer_mean": sent_mean,
        "ppl_answer_mean": ppl_mean,
        "mean_answer_words_all": float(sum(len(a.split()) for a in answers) / max(1, len(answers))),
        "n_valid_combined": len(valid_combined),
    }
    return {"metrics": metrics, "quality": quality}


def generate_h2_gaussian(
    *,
    model,
    tokenizer,
    prompts: list[str],
    steering_vectors_by_layer: dict[int, torch.Tensor],
    monitored_layers: list[int],
    gaussian_models,
    steer_target_class: str,
    alpha_mix: float,
    enable_debug: bool,
    desc: str,
):
    answers: list[str] = []
    debug_all: list[list[dict]] = []

    for i, prompt in enumerate(tqdm(prompts, desc=desc, leave=False)):
        enc = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)

        hs_buffer = SteeringHiddenStateBuffer()
        steerer = HiddenStateCaptureSteering(
            vectors=steering_vectors_by_layer,
            layer_ids=STEER_LAYERS,
            alpha=1.0,
            capture_layer_ids=monitored_layers,
            hidden_state_buffer=hs_buffer,
        )

        debug_this_example = enable_debug and i < N_DEBUG_EXAMPLES
        fill_scorer = H2GaussianFillScorer(
            buffer=hs_buffer,
            gaussian_models_by_layer=gaussian_models,
            monitored_layers=monitored_layers,
            steer_target_class=steer_target_class,
            alpha_mix=alpha_mix,
            enable_debug=debug_this_example,
        )

        out = llada_generate(
            model=model,
            prompt=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            steps=GEN_PARAMS["steps"],
            gen_length=GEN_PARAMS["gen_length"],
            block_length=GEN_PARAMS["block_length"],
            temperature=GEN_PARAMS["temperature"],
            fill_strategy=GEN_PARAMS["fill_strategy"],
            fill_scorer=fill_scorer,
            remasking=None,
            steering=steerer,
            show_progress=False,
        )

        answer_ids = out[:, enc["input_ids"].shape[1]:]
        answers.extend(tokenizer.batch_decode(answer_ids, skip_special_tokens=True))

        if debug_this_example:
            debug_all.append(getattr(fill_scorer, "debug_records", []))
        else:
            debug_all.append([])

        # Memory guard for long full runs: free prompt-local tensors/objects early.
        del answer_ids
        del out
        del enc
        del fill_scorer
        del steerer
        del hs_buffer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return answers, debug_all


def save_run(
    prompts: list[str],
    answers: list[str],
    debug_all: list[list[dict]],
    run_tag: str,
    run_info: dict[str, Any],
) -> dict[str, Any]:
    out_dir = OUT_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_payload = evaluate_texts_filtered(prompts, answers)
    metrics = eval_payload["metrics"]
    quality = eval_payload["quality"]

    with (out_dir / "generations.jsonl").open("w") as f:
        for i in range(len(prompts)):
            f.write(json.dumps({
                "prompt": prompts[i],
                "answer": answers[i],
                "is_valid": quality[i]["is_valid"],
                "has_bad_pattern": quality[i]["has_bad_pattern"],
                "has_repetition_loop": quality[i]["has_repetition_loop"],
                "is_empty": quality[i]["is_empty"],
            }) + "\n")

    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "run_info.json", run_info)

    for ei, records in enumerate(debug_all):
        if not records:
            continue
        _write_json(
            out_dir / f"debug_example_{ei:03d}.json",
            {
                "example_index": ei,
                "prompt": prompts[ei],
                "answer": answers[ei],
                "diffmean_lambda": run_info["diffmean_lambda"],
                "alpha_mix": run_info["fill_scoring"]["alpha_mix"],
                "fill_steps": records,
            },
        )

    print(
        f"  [done] invalid={metrics['invalid_percent']:.1f}% "
        f"valid={metrics['n_valid']}/{metrics['n_total']} "
        f"sent_valid={metrics['sent_answer_mean']} "
        f"ppl_valid={metrics['ppl_answer_mean']}"
    )
    return metrics


def analyze_debug(out_dir: Path, run_tag_a: str, run_tag_b: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_a": run_tag_a,
        "run_b": run_tag_b,
        "totals": {},
        "examples": [],
    }

    steps = 0
    rank_change_vs_baseline = 0
    rank_change_between_runs = 0
    score_in_unit_interval = 0
    score_checked = 0

    for ei in range(N_DEBUG_EXAMPLES):
        p_a = out_dir / run_tag_a / f"debug_example_{ei:03d}.json"
        p_b = out_dir / run_tag_b / f"debug_example_{ei:03d}.json"
        if not p_a.exists() or not p_b.exists():
            continue

        d_a = json.loads(p_a.read_text())
        d_b = json.loads(p_b.read_text())
        s_a = d_a.get("fill_steps", [])
        s_b = d_b.get("fill_steps", [])
        n = min(len(s_a), len(s_b))

        ex = {
            "example_index": ei,
            "steps": n,
            "rank_change_vs_baseline": 0,
            "rank_change_between_runs": 0,
            "score_in_unit_interval": 0,
            "score_checked": 0,
        }

        for i in range(n):
            ra = s_a[i]
            rb = s_b[i]
            base = rb.get("baseline_fill_score", [])
            gau = rb.get("gaussian_fill_score", [])
            mix_a = ra.get("mixed_fill_score", [])
            mix_b = rb.get("mixed_fill_score", [])

            if len(gau) > 0:
                score_checked += len(gau)
                ex["score_checked"] += len(gau)
                ok = sum(1 for v in gau if 0.0 - 1e-6 <= v <= 1.0 + 1e-6)
                score_in_unit_interval += ok
                ex["score_in_unit_interval"] += ok

            if len(base) >= 2 and len(base) == len(gau):
                rank_base = sorted(range(len(base)), key=lambda j: base[j], reverse=True)
                rank_gau = sorted(range(len(gau)), key=lambda j: gau[j], reverse=True)
                if rank_base != rank_gau:
                    rank_change_vs_baseline += 1
                    ex["rank_change_vs_baseline"] += 1

            if len(mix_a) >= 2 and len(mix_a) == len(mix_b):
                rank_a = sorted(range(len(mix_a)), key=lambda j: mix_a[j], reverse=True)
                rank_b = sorted(range(len(mix_b)), key=lambda j: mix_b[j], reverse=True)
                if rank_a != rank_b:
                    rank_change_between_runs += 1
                    ex["rank_change_between_runs"] += 1

            steps += 1

        summary["examples"].append(ex)

    summary["totals"] = {
        "steps_compared": steps,
        "rank_change_vs_baseline_steps": rank_change_vs_baseline,
        "rank_change_vs_baseline_fraction": float(rank_change_vs_baseline / max(1, steps)),
        "rank_change_between_runs_steps": rank_change_between_runs,
        "rank_change_between_runs_fraction": float(rank_change_between_runs / max(1, steps)),
        "gaussian_score_in_unit_interval": score_in_unit_interval,
        "gaussian_score_checked": score_checked,
        "gaussian_score_unit_fraction": float(score_in_unit_interval / max(1, score_checked)),
    }
    return summary


def _plot_hist(values: list[float], title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(values) > 0:
        ax.hist(values, bins=60, color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_smoke_score_diagnostics(out_dir: Path, run_tags: list[str], lam: float) -> dict[str, Any]:
    gaussian_values: list[float] = []
    mixed_values: list[float] = []
    corr_values: list[float] = []
    n_candidate_sets = 0

    for run_tag in run_tags:
        for ei in range(N_DEBUG_EXAMPLES):
            p = out_dir / run_tag / f"debug_example_{ei:03d}.json"
            if not p.exists():
                continue
            payload = json.loads(p.read_text())
            for step in payload.get("fill_steps", []):
                base = step.get("baseline_fill_score", [])
                gau = step.get("gaussian_fill_score", [])
                mix = step.get("mixed_fill_score", [])
                if len(gau) == 0:
                    continue

                n_candidate_sets += 1
                gaussian_values.extend(float(v) for v in gau)
                mixed_values.extend(float(v) for v in mix)

                if len(base) >= 2 and len(base) == len(gau):
                    b = np.asarray(base, dtype=np.float64)
                    g = np.asarray(gau, dtype=np.float64)
                    if np.std(b) > 1e-10 and np.std(g) > 1e-10:
                        corr_values.append(float(np.corrcoef(b, g)[0, 1]))

    hist_g = out_dir / f"gaussian_score_hist__lam{lam:.1f}.png"
    hist_m = out_dir / f"mixed_score_hist__lam{lam:.1f}.png"
    _plot_hist(
        gaussian_values,
        title=f"Normalized Gaussian score distribution (lam={lam:.1f})",
        xlabel="gaussian_fill_score",
        out_path=hist_g,
    )
    _plot_hist(
        mixed_values,
        title=f"Mixed fill score distribution (lam={lam:.1f})",
        xlabel="mixed_fill_score",
        out_path=hist_m,
    )

    diagnostics = {
        "lambda": lam,
        "run_tags": run_tags,
        "n_candidate_sets": n_candidate_sets,
        "n_gaussian_scores": len(gaussian_values),
        "n_mixed_scores": len(mixed_values),
        "gaussian_score_mean": float(np.mean(gaussian_values)) if gaussian_values else None,
        "gaussian_score_std": float(np.std(gaussian_values)) if gaussian_values else None,
        "gaussian_score_q05": float(np.quantile(gaussian_values, 0.05)) if gaussian_values else None,
        "gaussian_score_q50": float(np.quantile(gaussian_values, 0.50)) if gaussian_values else None,
        "gaussian_score_q95": float(np.quantile(gaussian_values, 0.95)) if gaussian_values else None,
        "mixed_score_mean": float(np.mean(mixed_values)) if mixed_values else None,
        "mixed_score_std": float(np.std(mixed_values)) if mixed_values else None,
        "avg_corr_baseline_vs_gaussian": float(np.mean(corr_values)) if corr_values else None,
        "median_corr_baseline_vs_gaussian": float(np.median(corr_values)) if corr_values else None,
        "n_corr_candidate_sets": len(corr_values),
        "hist_gaussian_path": str(hist_g),
        "hist_mixed_path": str(hist_m),
    }
    return diagnostics


def main() -> None:
    _seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lambdas = SMOKE_LAMBDAS if SMOKE_TEST else DIFFMEAN_LAMBDAS
    alpha_sweep = SMOKE_ALPHA_MIX if SMOKE_TEST else ALL_ALPHA_MIX

    prompts, labels, eval_meta = load_holdout_prompts(
        start_idx=EVAL_START_IDX,
        end_idx=EVAL_END_IDX,
        prompt_words=PROMPT_WORDS,
    )
    if SMOKE_TEST:
        prompts = prompts[:SMOKE_MAX_PROMPTS]
        labels = labels[:SMOKE_MAX_PROMPTS]

    _write_json(
        OUT_DIR / "eval_set.json",
        {
            "meta": eval_meta,
            "labels": labels,
            "prompts": prompts,
            "smoke_test": SMOKE_TEST,
        },
    )

    gaussian_models = fit_or_load_gaussian_models(
        cache_path=GAUSS_CACHE_PATH,
        monitored_layers=MONITORED_GAUSS_LAYERS,
        force_refit=False,
    )

    direction_definition = gaussian_models[MONITORED_GAUSS_LAYERS[0]].direction_definition

    probe_root = DIFFMEAN_OUT_ROOT / "probes"
    base_vectors = load_mean_vectors(probe_root, DIFFMEAN_VARIANT, STEER_LAYERS)

    print(f"[model] loading {MODEL_NAME} on {DEVICE}")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    summary: dict[str, Any] = {
        "mode": "SMOKE" if SMOKE_TEST else "FULL",
        "direction_definition": direction_definition,
        "gaussian_score_definition": "negative_diagonal_mahalanobis",
        "gaussian_normalization": "per-step per-layer minmax over candidate tokens to [0,1]",
        "monitored_layers": MONITORED_GAUSS_LAYERS,
        "results": {},
    }

    for lam in lambdas:
        scaled_vectors = {li: lam * v for li, v in base_vectors.items()}
        steer_target_class = "positive" if lam >= 0 else "negative"

        for alpha_mix in alpha_sweep:
            run_tag = f"h2g__{DIFFMEAN_VARIANT}__lam{lam:.1f}__mix{alpha_mix:.2f}"
            print(f"[run] {run_tag} target={steer_target_class}")

            answers, debug_all = generate_h2_gaussian(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                steering_vectors_by_layer=scaled_vectors,
                monitored_layers=MONITORED_GAUSS_LAYERS,
                gaussian_models=gaussian_models,
                steer_target_class=steer_target_class,
                alpha_mix=alpha_mix,
                enable_debug=SMOKE_TEST,
                desc=f"H2G lam={lam:.1f} mix={alpha_mix:.2f}",
            )

            run_info = {
                "model": MODEL_NAME,
                "device": DEVICE,
                "diffmean_lambda": lam,
                "steer_target_class": steer_target_class,
                "direction_definition": direction_definition,
                "fill_scoring": {
                    "heuristic": "H2_Gaussian_fill_rank",
                    "alpha_mix": alpha_mix,
                    "fill_strategy": GEN_PARAMS["fill_strategy"],
                },
                "gaussian_models": {
                    "cache": str(GAUSS_CACHE_PATH),
                    "layers": MONITORED_GAUSS_LAYERS,
                    "score_definition": "negative_diagonal_mahalanobis",
                    "normalization": "per-step per-layer minmax over candidates",
                },
                "n_prompts": len(prompts),
                "smoke_test": SMOKE_TEST,
            }

            metrics = save_run(prompts, answers, debug_all, run_tag, run_info)
            summary["results"][run_tag] = metrics

    _write_json(OUT_DIR / "summary.json", summary)

    if SMOKE_TEST and 1.0 in alpha_sweep and 0.0 in alpha_sweep:
        for lam in lambdas:
            tag_a = f"h2g__{DIFFMEAN_VARIANT}__lam{lam:.1f}__mix1.00"
            tag_b = f"h2g__{DIFFMEAN_VARIANT}__lam{lam:.1f}__mix0.00"
            cmp = analyze_debug(OUT_DIR, tag_a, tag_b)
            out_cmp = OUT_DIR / f"gaussian_alpha_compare__lam{lam:.1f}.json"
            _write_json(out_cmp, cmp)
            diag = build_smoke_score_diagnostics(OUT_DIR, [tag_a, tag_b], lam=lam)
            out_diag = OUT_DIR / f"gaussian_score_diagnostics__lam{lam:.1f}.json"
            _write_json(out_diag, diag)
            t = cmp["totals"]
            print(
                f"[smoke-check lam={lam:.1f}] "
                f"score_unit={t['gaussian_score_in_unit_interval']}/{t['gaussian_score_checked']} "
                f"rank_vs_baseline={t['rank_change_vs_baseline_steps']}/{t['steps_compared']} "
                f"rank_between_runs={t['rank_change_between_runs_steps']}/{t['steps_compared']}"
            )
            print(f"[smoke-check] saved -> {out_cmp}")
            print(
                f"[smoke-score lam={lam:.1f}] avg_corr(base,gauss)={diag['avg_corr_baseline_vs_gaussian']} "
                f"hist_g={diag['hist_gaussian_path']} hist_m={diag['hist_mixed_path']}"
            )
            print(f"[smoke-score] saved -> {out_diag}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
