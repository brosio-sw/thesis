# LLaDA Toxicity-Reduction — Master Thesis

Reduce LLM toxicity using **LLaDA** (Large Language Diffusion with mAsking)
by combining remasking heuristics with activation-steering.

## Project structure

```
thesis/
├── data/
│   ├── toxicity/       # RTP prompts & labels (downloaded at runtime)
│   ├── generations/    # Outputs per method (generations.jsonl + metrics.json)
│   └── processed/      # Aggregated tables & plots
├── eval/
│   ├── toxicity_metrics.py   # Detoxify-based scoring
│   ├── fluency_metrics.py    # GPT-2 perplexity + distinct-n
│   ├── aggregate_results.py  # Merge per-run JSONs → CSV/Markdown table
│   └── pareto_plots.py       # Toxicity vs. fluency Pareto plots
├── steering/
│   ├── base.py               # Abstract interface
│   └── mean_steering.py      # S1: mean activation steering (Turner/Zou style)
├── remasking/
│   ├── base.py               # Abstract interface
│   ├── random.py             # Uniform random remasking (baseline)
│   ├── low_confidence.py     # H1: low-confidence remasking (LLaDA default)
│   └── remdm_conf.py         # ReMDM-conf (Wang et al. 2025)
├── llada/
│   ├── model.py              # Load LLaDA model + tokenizer
│   └── generate.py           # generate() with pluggable remasking + steering
├── scripts/
│   ├── run_generation.py     # End-to-end: load → generate → score → save
│   └── run_evaluation.py     # Re-score saved generation files
└── requirements.txt
```

## Setup

```bash
# Install Miniconda if needed, then:
conda create -n thesis python=3.10 -y
conda activate thesis
pip install -r requirements.txt
```

## Quick start (H1 + S1)

```bash
conda activate thesis
cd ~/thesis

# Baseline (no steering, low-confidence remasking)
python scripts/run_generation.py --method baseline --num_prompts 200

# H1 + S1 (low-confidence remasking + mean activation steering)
python scripts/run_generation.py \
    --method h1s1 \
    --remasking low_confidence \
    --remask_fraction 0.1 \
    --steering mean_steering \
    --alpha 15.0 \
    --build_steering_cache \
    --num_prompts 200
    
# Aggregate results
python eval/aggregate_results.py
python eval/pareto_plots.py
```

## Key references
- **LLaDA**: Nie et al. (2025) arXiv:2502.09992
- **ReMDM**: Wang et al. (2025) arXiv:2503.00307
- **RealToxicityPrompts**: Gehman et al. (2020)
- **Activation Addition / S1**: Turner et al. (2023) arXiv:2308.10248
- **Representation Engineering**: Zou et al. (2023) arXiv:2310.01405
- **Detoxify**: Hanu & Unitary (2020)
