import os
import sys
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from llada.model import load_model, load_tokenizer
from llada.generate import generate as llada_generate
from eval.sentiment_metrics import compute_sentiment_metrics
from eval.fluency_metrics import compute_perplexity
from steering.liseco_probe_steering import LiSeCoProbeSteering, load_probe_params
from liseco_test import find_probes_root, infer_layers_from_probe_dir, PROBE_FAMILY, load_indexed_eval_texts, generate_with_liseco

MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_PARAMS = dict(
    temperature=0.0,
    steps=30,
    gen_length=30,
    block_length=30,
    fill_strategy="low_confidence",
)

def run_diagnostic():
    torch.manual_seed(42)
    n_prompts = 20

    intervals = [
        (0.04, 0.06),
        (0.24, 0.26),
    ]

    probes_root = find_probes_root()
    all_probe_layers = infer_layers_from_probe_dir(probes_root, PROBE_FAMILY)
    layer_configs = {
        "layer23_only": [23],
        "all_layers": all_probe_layers
    }

    print("[eval_set] Loading 20 prompts …")
    raw_texts, prompt_texts, eval_meta = load_indexed_eval_texts(
        start_idx=20000,
        end_idx=20500,
        raw_words=50,
        prompt_words=20,
        n_limit=n_prompts,
    )

    print(f"\n[model] Loading {MODEL_NAME} …")
    model = load_model(model_name=MODEL_NAME, device=DEVICE)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)

    results = {}
    
    # Run Baseline first
    print("\n--- Running Baseline (No Control) ---")
    baseline_answers = generate_with_liseco(
        model=model, tokenizer=tokenizer, prompts=prompt_texts, steerer=None, desc="Generating Baseline"
    )
    baseline_sent = compute_sentiment_metrics(baseline_answers, device=DEVICE)["mean_negative"]
    results["baseline"] = {
        "answers": baseline_answers,
        "sentiment_mean": baseline_sent,
        "stats": None
    }

    # Run LiSeCo conditions
    for layer_mode, layer_ids in layer_configs.items():
        for (amin, amax) in intervals:
            tag = f"liseco_{layer_mode}_amin{amin}_amax{amax}"
            print(f"\n--- Running {tag} ---")
            
            probe_by_layer = load_probe_params(probes_root=probes_root, family=PROBE_FAMILY, layer_ids=layer_ids)
            steerer = LiSeCoProbeSteering(
                probe_by_layer=probe_by_layer,
                layer_ids=layer_ids,
                alpha_min=amin,
                alpha_max=amax,
            )

            answers = generate_with_liseco(
                model=model, tokenizer=tokenizer, prompts=prompt_texts, steerer=steerer, desc=f"Generating {tag}"
            )
            
            sent = compute_sentiment_metrics(answers, device=DEVICE)["mean_negative"]
            
            mean_delta = steerer.sum_delta_norm / max(1, steerer.total_projected_positions)
            frac_modified = steerer.total_projected_positions / max(1, steerer.total_masked_positions_seen)

            stats = {
                "forward_calls": steerer.forward_calls,
                "layer_hook_calls": steerer.layer_hook_calls,
                "masked_pos_seen": steerer.total_masked_positions_seen,
                "projected_pos": steerer.total_projected_positions,
                "frac_modified": frac_modified,
                "mean_delta_norm": mean_delta,
                "max_delta_norm": steerer.max_delta_norm
            }
            
            results[tag] = {
                "answers": answers,
                "sentiment_mean": sent,
                "stats": stats
            }

    # Analyze and Output Results
    print("\n\n" + "="*80)
    print("DIAGNOSTIC REPORT")
    print("="*80)
    for tag, res in results.items():
        print(f"\nCondition: {tag}")
        print(f"Sentiment mean: {res['sentiment_mean']:.4f}")
        if res["stats"]:
            st = res["stats"]
            print(f"Hooks: forward_calls={st['forward_calls']}, layer_hook_calls={st['layer_hook_calls']}")
            print(f"Positions: {st['projected_pos']} / {st['masked_pos_seen']} ({st['frac_modified']:.2%} modified)")
            print(f"Magnitude stats: mean_||delta|| = {st['mean_delta_norm']:.4f}, max_||delta|| = {st['max_delta_norm']:.4f}")

    print("\nSAMPLE COMPARISON")
    print("-" * 80)
    for i in range(3): # compare first 3 prompts
        print(f"\n[PROMPT {i}] {prompt_texts[i]}")
        print(f"  [Baseline]     {results['baseline']['answers'][i]}")
        for tag in results.keys():
            if tag == "baseline": continue
            print(f"  [{tag}] {results[tag]['answers'][i]}")

if __name__ == "__main__":
    run_diagnostic()