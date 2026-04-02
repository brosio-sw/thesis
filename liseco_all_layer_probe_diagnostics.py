from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity
from llada.generate import generate as llada_generate
from llada.model import load_model, load_tokenizer
from steering.liseco_probe_steering import LiSeCoProbeSteering, ProbeParams, load_probe_params
from steering.precomputed_steering import _get_llada_layers


SMOKE_TEST = False
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"

OUT_ROOT = Path("data/alignment_variants_v4/full_run")
PROBE_ROOT = OUT_ROOT / "probes"
ACT_ROOT = OUT_ROOT / "activations"
OUT_DIR = Path("data/liseco_all_layer_probe_diagnostics") / ("smoke_test" if SMOKE_TEST else "full_run")

VARIANTS = ["real_full_pooled", "masked_tokenwise"]
INTERVALS = [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]

PROMPT_WORDS = 20
GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",
)

TEST_N = 2 if SMOKE_TEST else 100
TRAIN_MAX_ROWS = 5 if SMOKE_TEST else None
TEST_BATCH_SIZE = 10

BAD_PATTERNS = [
    r"Is this .*?\?",
    r"positive or negative\?",
    r"\bAnswer:",
    r"\bYesTitle:",
    r"\bNoTitle:",
    r"\bTitle:",
    r"\bReview:",
]


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _init_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def _append_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerows(rows)


def _iter_csv_rows(path: Path):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def _collect_layer_rows(path: Path, layer: int) -> list[dict[str, Any]]:
    out = []
    for row in _iter_csv_rows(path):
        if int(row["layer"]) == layer:
            out.append(row)
    return out


def _truncate(text: str, n_words: int) -> str:
    return " ".join(text.split()[:n_words]).strip()


def has_bad_pattern(text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in BAD_PATTERNS)


def has_repetition_loop(text: str) -> bool:
    toks = text.split()
    if len(toks) < 12:
        return False
    for n in [2, 3, 4]:
        for i in range(len(toks) - 2 * n + 1):
            if toks[i : i + n] == toks[i + n : i + 2 * n]:
                return True
    return False


def is_valid_generation(answer: str) -> bool:
    return not (not answer.strip() or has_bad_pattern(answer) or has_repetition_loop(answer))


def infer_probe_layers(variant: str) -> list[int]:
    p = PROBE_ROOT / variant
    return sorted(int(d.name.split("_")[1]) for d in p.glob("layer_*") if d.is_dir())


def interval_tag(alpha_min: float, alpha_max: float) -> str:
    return f"a{alpha_min:.2f}_{alpha_max:.2f}".replace(".", "p")


def load_test_prompts(n: int) -> list[str]:
    ds = load_dataset("fancyzhx/amazon_polarity", split="test")
    prompts: list[str] = []
    for ex in ds:
        text = ex.get("content") or ex.get("text") or f"{ex.get('title', '')} {ex.get('content', '')}".strip()
        p = _truncate(text, PROMPT_WORDS)
        if len(p.split()) >= 5:
            prompts.append(p)
        if len(prompts) >= n:
            break
    return prompts


def load_source_label_map() -> dict[int, int]:
    label_by_source: dict[int, int] = {}
    variant_dir = ACT_ROOT / "real_full_pooled"
    for meta_path in sorted(variant_dir.glob("part*_meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        for row in meta["rows"]:
            label_by_source[int(row["source_text_idx"])] = int(row["label"])
    return label_by_source


def stream_train_tables_for_variant(
    variant: str,
    layer_ids: list[int],
    probes: dict[int, ProbeParams],
    label_by_source: dict[int, int],
    var_dir: Path,
) -> tuple[int, int]:
    raw_fields = ["variant", "layer", "label", "z", "score"]
    prepost_fields = ["variant", "layer", "label", "interval_tag", "z_pre", "z_post", "score_pre", "score_post", "corrected", "delta_z"]

    raw_csv = var_dir / f"train_raw_projection_{variant}.csv"
    _init_csv(raw_csv, raw_fields)
    prepost_csv_by_tag: dict[str, Path] = {}
    for alpha_min, alpha_max in INTERVALS:
        tag = interval_tag(alpha_min, alpha_max)
        p = var_dir / f"train_liseco_prepost_{variant}_{tag}.csv"
        _init_csv(p, prepost_fields)
        prepost_csv_by_tag[tag] = p

    n_raw = 0
    n_prepost = 0
    processed_rows = 0

    variant_dir = ACT_ROOT / variant
    pt_files = sorted(variant_dir.glob("part*.pt"))

    for pt in pt_files:
        acts = torch.load(pt, map_location="cpu", weights_only=False)
        with open(pt.with_name(pt.stem + "_meta.json")) as f:
            meta = json.load(f)
        rows = meta["rows"]

        take_n = len(rows)
        if TRAIN_MAX_ROWS is not None:
            remaining = TRAIN_MAX_ROWS - processed_rows
            if remaining <= 0:
                break
            take_n = min(take_n, remaining)
        if take_n <= 0:
            continue

        labels: list[int] = []
        for r in rows[:take_n]:
            if variant == "real_full_pooled":
                labels.append(int(r["label"]))
            else:
                labels.append(int(label_by_source.get(int(r["source_text_idx"]), int(r.get("target_label", 0)))))

        raw_batch: list[dict[str, Any]] = []
        prepost_batch_by_tag: dict[str, list[dict[str, Any]]] = {interval_tag(a, b): [] for a, b in INTERVALS}

        for li in layer_ids:
            X = acts[li][:take_n].float().cpu()
            if X.numel() == 0:
                continue
            z, score = compute_z_score(X, probes[li])

            for i in range(X.shape[0]):
                raw_batch.append(
                    {
                        "variant": variant,
                        "layer": li,
                        "label": labels[i],
                        "z": float(z[i].item()),
                        "score": float(score[i].item()),
                    }
                )

            for alpha_min, alpha_max in INTERVALS:
                tag = interval_tag(alpha_min, alpha_max)
                z_post, corrected, dz = offline_liseco_project(z, probes[li], alpha_min, alpha_max)
                score_post = torch.sigmoid(z_post)
                for i in range(X.shape[0]):
                    prepost_batch_by_tag[tag].append(
                        {
                            "variant": variant,
                            "layer": li,
                            "label": labels[i],
                            "interval_tag": tag,
                            "z_pre": float(z[i].item()),
                            "z_post": float(z_post[i].item()),
                            "score_pre": float(score[i].item()),
                            "score_post": float(score_post[i].item()),
                            "corrected": bool(corrected[i].item()),
                            "delta_z": float(dz[i].item()),
                        }
                    )

        _append_csv_rows(raw_csv, raw_batch, raw_fields)
        n_raw += len(raw_batch)

        for tag, rows_tag in prepost_batch_by_tag.items():
            _append_csv_rows(prepost_csv_by_tag[tag], rows_tag, prepost_fields)
            n_prepost += len(rows_tag)

        processed_rows += take_n
        del acts
        gc.collect()

    return n_raw, n_prepost


def load_probes_for_variant(variant: str, layer_ids: list[int]) -> dict[int, ProbeParams]:
    return load_probe_params(PROBE_ROOT, variant, layer_ids)


def compute_z_score(X: torch.Tensor, probe: ProbeParams) -> tuple[torch.Tensor, torch.Tensor]:
    w = probe.weight.view(-1).float()
    b = probe.bias.view(-1).float()[0]
    z = X.float() @ w + b
    score = torch.sigmoid(z)
    return z, score


def offline_liseco_project(z_pre: torch.Tensor, probe: ProbeParams, alpha_min: float, alpha_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_min = torch.logit(torch.tensor(max(1e-6, min(1 - 1e-6, alpha_min)), dtype=torch.float32))
    z_max = torch.logit(torch.tensor(max(1e-6, min(1 - 1e-6, alpha_max)), dtype=torch.float32))
    coef = torch.zeros_like(z_pre)
    low = z_pre < z_min
    high = z_pre > z_max
    norm_sq = float(probe.norm_sq)
    if low.any():
        coef[low] = (z_min - z_pre[low]) / (norm_sq + 1e-8)
    if high.any():
        coef[high] = (z_max - z_pre[high]) / (norm_sq + 1e-8)
    z_post = z_pre + coef * norm_sq
    corrected = low | high
    return z_post, corrected, coef * norm_sq


def forward_layer_cache(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_ids: list[int]) -> dict[int, torch.Tensor]:
    layers = _get_llada_layers(model)
    cache: dict[int, torch.Tensor] = {}
    hooks = []

    def _mk_hook(li: int):
        def _hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            cache[li] = hs.detach().float().cpu()
        return _hook

    for li in layer_ids:
        hooks.append(layers[li].register_forward_hook(_mk_hook(li)))

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    return cache


def generate_one(model, tokenizer, prompt: str, steering=None) -> str:
    enc = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)
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
        steering=steering,
        show_progress=False,
    )
    answer_ids = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]


def sentiment_labels_from_answers(answers: list[str]) -> tuple[list[int], list[float]]:
    sent = compute_sentiment_metrics(answers, device=DEVICE)
    neg_scores = sent["scores"].tolist()
    labels = [int(s >= 0.5) for s in neg_scores]
    return labels, neg_scores


def evaluate_test_run_metrics(answers: list[str]) -> dict[str, Any]:
    quality = [is_valid_generation(a) for a in answers]
    n_total = len(answers)
    valid_indices = [i for i, ok in enumerate(quality) if ok]
    valid_answers = [answers[i] for i in valid_indices]
    n_valid = len(valid_answers)
    n_invalid = n_total - n_valid
    invalid_fraction = float(n_invalid / max(1, n_total))

    if n_valid > 0:
        sent = compute_sentiment_metrics(valid_answers, device=DEVICE)
        ppl = compute_perplexity(valid_answers, device="cpu")
        sent_answer_mean = float(sent["mean_negative"])
        sent_answer_fraction = float(sent["negative_fraction"])
        ppl_answer_mean = float(ppl["mean_ppl"])
    else:
        sent_answer_mean = None
        sent_answer_fraction = None
        ppl_answer_mean = None

    return {
        "evaluation_subset": "valid_generations_only",
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_fraction": invalid_fraction,
        "sent_answer_mean": sent_answer_mean,
        "sent_answer_fraction": sent_answer_fraction,
        "ppl_answer_mean": ppl_answer_mean,
    }


def plot_hist_two_classes(rows: list[dict[str, Any]], value_key: str, out_png: Path, title: str, boundary: float) -> None:
    vals0 = [float(r[value_key]) for r in rows if int(r["label"]) == 0]
    vals1 = [float(r[value_key]) for r in rows if int(r["label"]) == 1]
    if not vals0 and not vals1:
        return
    plt.figure(figsize=(7, 4))
    if vals0:
        plt.hist(vals0, bins=40, alpha=0.5, label="label=0")
    if vals1:
        plt.hist(vals1, bins=40, alpha=0.5, label="label=1")
    plt.axvline(boundary, linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def plot_hist_prepost(rows: list[dict[str, Any]], pre_key: str, post_key: str, out_png: Path, title: str) -> None:
    pre = [float(r[pre_key]) for r in rows]
    post = [float(r[post_key]) for r in rows]
    if not pre and not post:
        return
    plt.figure(figsize=(7, 4))
    if pre:
        plt.hist(pre, bins=40, alpha=0.5, label="pre")
    if post:
        plt.hist(post, bins=40, alpha=0.5, label="post")
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()




def build_test_raw_table_for_variant(
    variant: str,
    layer_ids: list[int],
    probes: dict[int, ProbeParams],
    prompts: list[str],
    model,
    tokenizer,
) -> tuple[int, dict[str, Any]]:
    fields = ["variant", "example_idx", "layer", "label", "sentiment_neg_score", "token_pos", "is_generation_token", "pooling", "z", "score"]
    out_csv = OUT_DIR / variant / f"test_raw_projection_{variant}.csv"
    _init_csv(out_csv, fields)

    n_rows = 0
    n_total = 0
    n_invalid = 0
    valid_answers_all: list[str] = []

    for batch_start in tqdm(range(0, len(prompts), TEST_BATCH_SIZE), desc=f"test raw [{variant}]", leave=False):
        batch_prompts = prompts[batch_start : batch_start + TEST_BATCH_SIZE]
        batch_answers = [generate_one(model, tokenizer, p, steering=None) for p in batch_prompts]
        batch_valid = [is_valid_generation(a) for a in batch_answers]

        n_total += len(batch_answers)
        n_invalid += sum(1 for ok in batch_valid if not ok)
        valid_local_ids = [i for i, ok in enumerate(batch_valid) if ok]
        valid_local_answers = [batch_answers[i] for i in valid_local_ids]
        valid_answers_all.extend(valid_local_answers)

        sent_labels, sent_scores = sentiment_labels_from_answers(valid_local_answers) if valid_local_answers else ([], [])
        label_by_local = {idx: sent_labels[j] for j, idx in enumerate(valid_local_ids)}
        score_by_local = {idx: sent_scores[j] for j, idx in enumerate(valid_local_ids)}

        batch_rows: list[dict[str, Any]] = []
        for i_local, prompt in enumerate(batch_prompts):
            if not batch_valid[i_local]:
                continue
            answer = batch_answers[i_local]
            combined = f"{prompt} {answer}".strip()
            enc = tokenizer([combined], return_tensors="pt", padding=True, truncation=True).to(model.device)
            cache = forward_layer_cache(model, enc["input_ids"], enc["attention_mask"], layer_ids)
            attn = enc["attention_mask"][0].detach().cpu().bool()
            prompt_len = int(tokenizer([prompt], return_tensors="pt", truncation=True)["input_ids"].shape[1])
            example_idx = batch_start + i_local

            for li in layer_ids:
                hs = cache[li][0]
                hs_valid = hs[attn]
                if variant == "real_full_pooled":
                    pooled = hs_valid.mean(dim=0, keepdim=True)
                    z, score = compute_z_score(pooled, probes[li])
                    batch_rows.append({
                        "variant": variant,
                        "example_idx": example_idx,
                        "layer": li,
                        "label": int(label_by_local[i_local]),
                        "sentiment_neg_score": float(score_by_local[i_local]),
                        "token_pos": -1,
                        "is_generation_token": False,
                        "pooling": "mean",
                        "z": float(z[0].item()),
                        "score": float(score[0].item()),
                    })
                else:
                    z, score = compute_z_score(hs_valid, probes[li])
                    valid_pos = torch.nonzero(attn, as_tuple=False).squeeze(-1).tolist()
                    for j, pos in enumerate(valid_pos):
                        batch_rows.append({
                            "variant": variant,
                            "example_idx": example_idx,
                            "layer": li,
                            "label": int(label_by_local[i_local]),
                            "sentiment_neg_score": float(score_by_local[i_local]),
                            "token_pos": int(pos),
                            "is_generation_token": bool(int(pos) >= prompt_len),
                            "pooling": "tokenwise",
                            "z": float(z[j].item()),
                            "score": float(score[j].item()),
                        })

            del cache

        _append_csv_rows(out_csv, batch_rows, fields)
        n_rows += len(batch_rows)
        gc.collect()

    n_valid = n_total - n_invalid
    if n_valid > 0:
        sent = compute_sentiment_metrics(valid_answers_all, device=DEVICE)
        ppl = compute_perplexity(valid_answers_all, device="cpu")
        run_metrics = {
            "evaluation_subset": "valid_generations_only",
            "n_total": n_total,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "invalid_fraction": float(n_invalid / max(1, n_total)),
            "sent_answer_mean": float(sent["mean_negative"]),
            "sent_answer_fraction": float(sent["negative_fraction"]),
            "ppl_answer_mean": float(ppl["mean_ppl"]),
        }
    else:
        run_metrics = {
            "evaluation_subset": "valid_generations_only",
            "n_total": n_total,
            "n_valid": 0,
            "n_invalid": n_invalid,
            "invalid_fraction": float(n_invalid / max(1, n_total)),
            "sent_answer_mean": None,
            "sent_answer_fraction": None,
            "ppl_answer_mean": None,
        }

    return n_rows, run_metrics


def build_test_liseco_prepost_for_variant_interval(
    variant: str,
    layer_ids: list[int],
    probes: dict[int, ProbeParams],
    prompts: list[str],
    model,
    tokenizer,
    alpha_min: float,
    alpha_max: float,
) -> tuple[int, dict[str, Any]]:
    tag = interval_tag(alpha_min, alpha_max)

    fields = [
        "variant",
        "interval_tag",
        "example_idx",
        "forward_call_idx",
        "layer",
        "batch_idx",
        "token_pos",
        "label",
        "sentiment_neg_score",
        "z_pre",
        "z_post",
        "score_pre",
        "score_post",
        "delta_z",
        "corrected",
        "delta_norm",
        "alpha_min",
        "alpha_max",
        "z_min",
        "z_max",
    ]
    out_csv = OUT_DIR / variant / f"test_liseco_prepost_{variant}_{tag}.csv"
    _init_csv(out_csv, fields)

    n_rows = 0
    n_total = 0
    n_invalid = 0
    valid_answers_all: list[str] = []

    for batch_start in tqdm(range(0, len(prompts), TEST_BATCH_SIZE), desc=f"test liseco [{variant} {tag}]", leave=False):
        batch_prompts = prompts[batch_start : batch_start + TEST_BATCH_SIZE]
        batch_answers: list[str] = []
        batch_valid: list[bool] = []
        batch_diag_rows: dict[int, list[dict[str, Any]]] = {}

        for i_local, p in enumerate(batch_prompts):
            example_idx = batch_start + i_local
            steerer = LiSeCoProbeSteering(
                probe_by_layer=probes,
                layer_ids=layer_ids,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                legacy_mode=False,
            )
            steerer.set_diagnostics_context(example_idx=example_idx, variant=variant, interval_tag=tag)
            a = generate_one(model, tokenizer, p, steering=steerer)
            batch_answers.append(a)
            ok = is_valid_generation(a)
            batch_valid.append(ok)
            batch_diag_rows[i_local] = steerer.pop_diagnostics_rows()

        n_total += len(batch_answers)
        n_invalid += sum(1 for ok in batch_valid if not ok)
        valid_local_ids = [i for i, ok in enumerate(batch_valid) if ok]
        valid_local_answers = [batch_answers[i] for i in valid_local_ids]
        valid_answers_all.extend(valid_local_answers)

        sent_labels, sent_scores = sentiment_labels_from_answers(valid_local_answers) if valid_local_answers else ([], [])
        label_by_local = {idx: sent_labels[j] for j, idx in enumerate(valid_local_ids)}
        score_by_local = {idx: sent_scores[j] for j, idx in enumerate(valid_local_ids)}

        rows_batch: list[dict[str, Any]] = []
        for i_local in valid_local_ids:
            for r in batch_diag_rows.get(i_local, []):
                rr = dict(r)
                rr["label"] = int(label_by_local[i_local])
                rr["sentiment_neg_score"] = float(score_by_local[i_local])
                rows_batch.append(rr)

        _append_csv_rows(out_csv, rows_batch, fields)
        n_rows += len(rows_batch)
        gc.collect()

    n_valid = n_total - n_invalid
    if n_valid > 0:
        sent = compute_sentiment_metrics(valid_answers_all, device=DEVICE)
        ppl = compute_perplexity(valid_answers_all, device="cpu")
        run_metrics = {
            "evaluation_subset": "valid_generations_only",
            "n_total": n_total,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "invalid_fraction": float(n_invalid / max(1, n_total)),
            "sent_answer_mean": float(sent["mean_negative"]),
            "sent_answer_fraction": float(sent["negative_fraction"]),
            "ppl_answer_mean": float(ppl["mean_ppl"]),
        }
    else:
        run_metrics = {
            "evaluation_subset": "valid_generations_only",
            "n_total": n_total,
            "n_valid": 0,
            "n_invalid": n_invalid,
            "invalid_fraction": float(n_invalid / max(1, n_total)),
            "sent_answer_mean": None,
            "sent_answer_fraction": None,
            "ppl_answer_mean": None,
        }

    return n_rows, run_metrics


def save_and_plot_family_outputs(
    variant: str,
    layer_ids: list[int],
    test_raw_metrics: dict[str, Any],
    test_liseco_metrics_by_interval: dict[str, dict[str, Any]],
) -> None:
    var_dir = OUT_DIR / variant
    var_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON outputs
    _write_json(var_dir / f"test_raw_metrics_{variant}.json", test_raw_metrics)
    for tag, m in test_liseco_metrics_by_interval.items():
        _write_json(var_dir / f"test_liseco_metrics_{variant}_{tag}.json", m)

    # Family 1 plots from saved CSV
    train_raw_csv = var_dir / f"train_raw_projection_{variant}.csv"
    for li in layer_ids:
        rows_li = _collect_layer_rows(train_raw_csv, li)
        plot_hist_two_classes(rows_li, "z", var_dir / "plots" / "train_raw" / "logit" / f"layer_{li:02d}.png", f"Train raw z {variant} L{li}", 0.0)
        plot_hist_two_classes(rows_li, "score", var_dir / "plots" / "train_raw" / "sigmoid" / f"layer_{li:02d}.png", f"Train raw score {variant} L{li}", 0.5)

    # Family 2 plots from saved CSV
    for alpha_min, alpha_max in INTERVALS:
        tag = interval_tag(alpha_min, alpha_max)
        p = var_dir / f"train_liseco_prepost_{variant}_{tag}.csv"
        for li in layer_ids:
            rows_li = _collect_layer_rows(p, li)
            plot_hist_prepost(rows_li, "z_pre", "z_post", var_dir / "plots" / "train_liseco_prepost" / tag / "logit" / f"layer_{li:02d}.png", f"Train LiSeCo z {variant} {tag} L{li}")
            plot_hist_prepost(rows_li, "score_pre", "score_post", var_dir / "plots" / "train_liseco_prepost" / tag / "sigmoid" / f"layer_{li:02d}.png", f"Train LiSeCo score {variant} {tag} L{li}")

    # Family 3 plots from saved CSV
    test_raw_csv = var_dir / f"test_raw_projection_{variant}.csv"
    for li in layer_ids:
        rows_li = _collect_layer_rows(test_raw_csv, li)
        plot_hist_two_classes(rows_li, "z", var_dir / "plots" / "test_raw" / "logit" / f"layer_{li:02d}.png", f"Test raw z {variant} L{li}", 0.0)
        plot_hist_two_classes(rows_li, "score", var_dir / "plots" / "test_raw" / "sigmoid" / f"layer_{li:02d}.png", f"Test raw score {variant} L{li}", 0.5)

    # Family 4 plots from saved CSV
    for alpha_min, alpha_max in INTERVALS:
        tag = interval_tag(alpha_min, alpha_max)
        p = var_dir / f"test_liseco_prepost_{variant}_{tag}.csv"
        for li in layer_ids:
            rows_li = _collect_layer_rows(p, li)
            plot_hist_prepost(rows_li, "z_pre", "z_post", var_dir / "plots" / "test_liseco_prepost" / tag / "logit" / f"layer_{li:02d}.png", f"Test LiSeCo z {variant} {tag} L{li}")
            plot_hist_prepost(rows_li, "score_pre", "score_post", var_dir / "plots" / "test_liseco_prepost" / tag / "sigmoid" / f"layer_{li:02d}.png", f"Test LiSeCo score {variant} {tag} L{li}")


def main() -> None:
    _seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_by_source = load_source_label_map()
    prompts = load_test_prompts(TEST_N)
    print(f"[info] smoke={SMOKE_TEST} test_prompts={len(prompts)} train_max_rows={TRAIN_MAX_ROWS}")

    print(f"[model] loading {MODEL_NAME} on {DEVICE} ...")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)
    llada_layers = _get_llada_layers(model)
    print(f"[debug] model_type={type(model).__name__} n_layers={len(llada_layers)}")

    summary: dict[str, Any] = {
        "smoke_test": SMOKE_TEST,
        "model": MODEL_NAME,
        "device": DEVICE,
        "variants": {},
        "intervals": [{"alpha_min": a, "alpha_max": b, "tag": interval_tag(a, b)} for a, b in INTERVALS],
        "n_test_prompts": len(prompts),
        "train_max_rows": TRAIN_MAX_ROWS,
    }

    for variant in VARIANTS:
        print(f"\n[variant] {variant}")
        layer_ids = infer_probe_layers(variant)
        probes = load_probes_for_variant(variant, layer_ids)
        var_dir = OUT_DIR / variant
        var_dir.mkdir(parents=True, exist_ok=True)

        n_train_raw_rows, n_train_prepost_rows = stream_train_tables_for_variant(
            variant=variant,
            layer_ids=layer_ids,
            probes=probes,
            label_by_source=label_by_source,
            var_dir=var_dir,
        )

        n_test_raw_rows, test_raw_metrics = build_test_raw_table_for_variant(
            variant=variant,
            layer_ids=layer_ids,
            probes=probes,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
        )

        n_test_liseco_rows_by_interval: dict[str, int] = {}
        test_liseco_metrics_by_interval: dict[str, dict[str, Any]] = {}
        for a_min, a_max in INTERVALS:
            tag = interval_tag(a_min, a_max)
            n_rows, run_metrics = build_test_liseco_prepost_for_variant_interval(
                variant=variant,
                layer_ids=layer_ids,
                probes=probes,
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                alpha_min=a_min,
                alpha_max=a_max,
            )
            n_test_liseco_rows_by_interval[tag] = n_rows
            test_liseco_metrics_by_interval[tag] = run_metrics

        save_and_plot_family_outputs(
            variant=variant,
            layer_ids=layer_ids,
            test_raw_metrics=test_raw_metrics,
            test_liseco_metrics_by_interval=test_liseco_metrics_by_interval,
        )

        summary["variants"][variant] = {
            "layers": layer_ids,
            "n_train_raw_rows": n_train_raw_rows,
            "n_train_prepost_rows": n_train_prepost_rows,
            "n_test_raw_rows": n_test_raw_rows,
            "n_test_liseco_rows_by_interval": n_test_liseco_rows_by_interval,
            "test_raw_metrics": test_raw_metrics,
            "test_liseco_metrics_by_interval": test_liseco_metrics_by_interval,
        }

    _write_json(OUT_DIR / "summary.json", summary)
    print(f"\n[done] diagnostics written to {OUT_DIR}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
