from __future__ import annotations

"""
train_liseco_probe_directions_instruct.py

Train one LiSeCo-style linear probe per layer from saved real-negative and
real-positive activation tensors, then export probe-derived steering vectors.

Input activation files (per class):
- real_negative.pt
- real_positive.pt

Each file format:
    { layer_idx: Tensor[N_examples, hidden_dim] }

Steering vector construction per layer:
1) Train binary linear probe with BCEWithLogitsLoss.
2) Extract learned weight vector W.
3) Unit-normalize: W_hat = W / (||W|| + eps).
4) Compute mean-diff norm: md_norm = || mean(pos) - mean(neg) ||.
5) Align sign to mean-diff direction and save:
       v_liseco = aligned(W_hat) * md_norm

Sign convention is chosen so subtracting alpha * v_liseco in
MeanActivationSteeringMaskedOnly pushes toward NEGATIVE sentiment.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42

DEFAULT_ACTIVATIONS_ROOT = Path("data/speed_adapt/instruct_real_sentiment_activations/full_run/activations")
DEFAULT_OUT_DIR = Path("data/speed_adapt/liseco_probe_directions_instruct")

DEFAULT_VAL_FRAC = 0.2
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 2e-3
DEFAULT_WEIGHT_DECAY = 1e-4

EPS = 1e-8

SMOKE_TEST = False
SMOKE_MAX_PER_CLASS = 128
SMOKE_EPOCHS = 8


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _read_activation_file(path: Path) -> dict[int, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing activation file: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[int, torch.Tensor] = {}
    for k, v in obj.items():
        out[int(k)] = v.float().cpu()
    if not out:
        raise RuntimeError(f"No layers found in activation file: {path}")
    return out


def load_real_class_activations(activations_root: Path) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    neg = _read_activation_file(activations_root / "real_negative.pt")
    pos = _read_activation_file(activations_root / "real_positive.pt")

    neg_layers = sorted(neg.keys())
    pos_layers = sorted(pos.keys())
    if neg_layers != pos_layers:
        raise RuntimeError(f"Layer mismatch between class files: neg={neg_layers}, pos={pos_layers}")

    for li in neg_layers:
        if neg[li].ndim != 2 or pos[li].ndim != 2:
            raise RuntimeError(f"Layer {li} must be rank-2 [N, D]")
        if neg[li].shape[1] != pos[li].shape[1]:
            raise RuntimeError(
                f"Hidden dim mismatch at layer {li}: neg={neg[li].shape[1]} pos={pos[li].shape[1]}"
            )

    return neg, pos


def _build_split_indices(n_neg: int, n_pos: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    neg_idx = np.arange(n_neg, dtype=np.int64)
    pos_idx = np.arange(n_pos, dtype=np.int64)
    rng.shuffle(neg_idx)
    rng.shuffle(pos_idx)

    n_neg_val = max(1, int(round(n_neg * val_frac)))
    n_pos_val = max(1, int(round(n_pos * val_frac)))

    if n_neg - n_neg_val < 1:
        n_neg_val = n_neg - 1
    if n_pos - n_pos_val < 1:
        n_pos_val = n_pos - 1

    if n_neg_val <= 0 or n_pos_val <= 0:
        raise RuntimeError("Too few samples to create train/val split for both classes")

    neg_val = neg_idx[:n_neg_val]
    neg_train = neg_idx[n_neg_val:]

    pos_val = pos_idx[:n_pos_val]
    pos_train = pos_idx[n_pos_val:]

    train_idx = np.concatenate([neg_train, n_neg + pos_train], axis=0)
    val_idx = np.concatenate([neg_val, n_neg + pos_val], axis=0)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _compute_binary_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    acc = float((preds == y).float().mean().item())

    pred_pos = preds == 1.0
    pred_neg = preds == 0.0
    true_pos = y == 1.0
    true_neg = y == 0.0

    tp = float((pred_pos & true_pos).float().sum().item())
    fp = float((pred_pos & true_neg).float().sum().item())
    fn = float((pred_neg & true_pos).float().sum().item())
    tn = float((pred_neg & true_neg).float().sum().item())

    precision_pos = tp / max(1.0, tp + fp)
    recall_pos = tp / max(1.0, tp + fn)

    precision_neg = tn / max(1.0, tn + fn)
    recall_neg = tn / max(1.0, tn + fp)

    return {
        "accuracy": acc,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
    }


def train_single_layer_probe(
    *,
    X_neg: torch.Tensor,
    X_pos: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    _seed_everything(seed)

    X = torch.cat([X_neg, X_pos], dim=0).float()
    y = torch.cat(
        [
            torch.zeros(X_neg.shape[0], dtype=torch.float32),
            torch.ones(X_pos.shape[0], dtype=torch.float32),
        ],
        dim=0,
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze(-1)
            val_loss = float(criterion(val_logits, y_val).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to produce checkpoint state")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_logits = model(X_train).squeeze(-1)
        val_logits = model(X_val).squeeze(-1)
        train_loss = float(criterion(train_logits, y_train).item())
        val_loss = float(criterion(val_logits, y_val).item())

    train_metrics = _compute_binary_metrics(train_logits, y_train)
    val_metrics = _compute_binary_metrics(val_logits, y_val)

    metrics = {
        "best_epoch": int(best_epoch),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "train_precision_pos": train_metrics["precision_pos"],
        "train_recall_pos": train_metrics["recall_pos"],
        "val_precision_pos": val_metrics["precision_pos"],
        "val_recall_pos": val_metrics["recall_pos"],
        "train_precision_neg": train_metrics["precision_neg"],
        "train_recall_neg": train_metrics["recall_neg"],
        "val_precision_neg": val_metrics["precision_neg"],
        "val_recall_neg": val_metrics["recall_neg"],
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "label_mapping": {
            "0": "negative",
            "1": "positive",
        },
    }

    weight = model.weight.detach().cpu().view(-1).float()
    bias = model.bias.detach().cpu().view(-1).float()

    artifact = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(X.shape[1]),
        "weight": weight,
        "bias": bias,
    }
    return metrics, artifact


def build_probe_direction_vector(
    *,
    weight: torch.Tensor,
    mean_diff: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    weight = weight.float().cpu()
    mean_diff = mean_diff.float().cpu()

    w_norm = float(torch.norm(weight).item())
    md_norm = float(torch.norm(mean_diff).item())

    w_hat = weight / (w_norm + eps)
    alignment_before = float(torch.dot(w_hat, mean_diff).item())

    sign_flip = alignment_before < 0.0
    if sign_flip:
        w_hat = -w_hat

    v_liseco = w_hat * md_norm

    metadata = {
        "formula": "v_liseco = aligned(W / (||W|| + eps)) * ||mean(pos)-mean(neg)||",
        "unit_normalized_first": True,
        "rescaled_by_mean_diff_norm": True,
        "eps": float(eps),
        "mean_diff_definition": "mean(pos) - mean(neg)",
        "sign_alignment_rule": "flip if dot(W_hat, mean_diff) < 0 so v aligns with mean_diff",
        "steering_application": "hidden_state = hidden_state - alpha * v_liseco",
        "target_effect": "subtracting alpha*v_liseco pushes toward negative sentiment",
        "w_norm": w_norm,
        "mean_diff_norm": md_norm,
        "alignment_before_sign_flip": alignment_before,
        "sign_flipped": bool(sign_flip),
        "alignment_after_sign_flip": float(torch.dot(v_liseco, mean_diff).item()) / max(md_norm, eps),
    }
    return v_liseco, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activations-root",
        type=Path,
        default=DEFAULT_ACTIVATIONS_ROOT,
        help="Directory containing real_negative.pt and real_positive.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Base output directory",
    )
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--layers", type=str, default="all", help="Comma list or 'all'")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-per-class", type=int, default=SMOKE_MAX_PER_CLASS)
    parser.set_defaults(smoke_test=SMOKE_TEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(SEED)

    run_dir = args.out_dir / ("smoke_test" if args.smoke_test else "full_run")
    probes_dir = run_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    neg_by_layer, pos_by_layer = load_real_class_activations(args.activations_root)
    available_layers = sorted(neg_by_layer.keys())

    if args.layers == "all":
        layers = available_layers
    else:
        requested = sorted({int(x.strip()) for x in args.layers.split(",") if x.strip()})
        missing = [li for li in requested if li not in available_layers]
        if missing:
            raise RuntimeError(f"Requested layers missing from activations: {missing}")
        layers = requested

    if args.smoke_test:
        n_neg = min(args.smoke_max_per_class, neg_by_layer[layers[0]].shape[0])
        n_pos = min(args.smoke_max_per_class, pos_by_layer[layers[0]].shape[0])
        for li in layers:
            neg_by_layer[li] = neg_by_layer[li][:n_neg]
            pos_by_layer[li] = pos_by_layer[li][:n_pos]
        if args.layers == "all":
            preferred = [li for li in [9, 25] if li in layers]
            if len(preferred) >= 2:
                layers = preferred
            elif len(layers) >= 2:
                layers = [layers[0], layers[-1]]
        epochs = SMOKE_EPOCHS
    else:
        epochs = args.epochs

    n_neg = int(neg_by_layer[layers[0]].shape[0])
    n_pos = int(pos_by_layer[layers[0]].shape[0])

    train_idx, val_idx = _build_split_indices(n_neg=n_neg, n_pos=n_pos, val_frac=args.val_frac, seed=SEED)

    steering_vectors: dict[int, torch.Tensor] = {}
    steering_meta_by_layer: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for li in layers:
        layer_dir = probes_dir / f"layer_{li:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        X_neg = neg_by_layer[li]
        X_pos = pos_by_layer[li]
        if X_neg.shape[1] != X_pos.shape[1]:
            raise RuntimeError(f"Hidden dim mismatch at layer {li}")

        metrics, artifact = train_single_layer_probe(
            X_neg=X_neg,
            X_pos=X_pos,
            train_idx=train_idx,
            val_idx=val_idx,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=SEED + li,
        )

        mean_diff = X_pos.mean(dim=0) - X_neg.mean(dim=0)
        v_liseco, v_meta = build_probe_direction_vector(
            weight=artifact["weight"],
            mean_diff=mean_diff,
            eps=EPS,
        )

        steering_vectors[li] = v_liseco.float().cpu()
        steering_meta_by_layer[str(li)] = v_meta

        torch.save(
            {
                "state_dict": artifact["state_dict"],
                "input_dim": artifact["input_dim"],
                "label_mapping": {"0": "negative", "1": "positive"},
            },
            layer_dir / "probe.pt",
        )
        torch.save(artifact["state_dict"], layer_dir / "probe_state_dict.pt")
        torch.save(
            {
                "weight": artifact["weight"],
                "bias": artifact["bias"],
                "input_dim": artifact["input_dim"],
            },
            layer_dir / "probe_weight_bias.pt",
        )
        _write_json(layer_dir / "metrics.json", metrics)
        _write_json(layer_dir / "steering_vector_meta.json", v_meta)

        summary_rows.append(
            {
                "layer": li,
                "val_accuracy": metrics["val_accuracy"],
                "train_accuracy": metrics["train_accuracy"],
                "val_loss": metrics["val_loss"],
                "train_loss": metrics["train_loss"],
                "mean_diff_norm": v_meta["mean_diff_norm"],
                "w_norm": v_meta["w_norm"],
                "sign_flipped": int(v_meta["sign_flipped"]),
            }
        )

    torch.save({int(k): v for k, v in steering_vectors.items()}, run_dir / "steering_vectors.pt")

    summary_rows = sorted(summary_rows, key=lambda r: r["layer"])
    _write_json(
        run_dir / "summary.json",
        {
            "run_mode": "smoke_test" if args.smoke_test else "full_run",
            "activations_root": str(args.activations_root),
            "layers_trained": layers,
            "n_negative": n_neg,
            "n_positive": n_pos,
            "val_frac": args.val_frac,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "vector_build_metadata": {
                "formula": "v_liseco = aligned(W / (||W|| + eps)) * ||mean(pos)-mean(neg)||",
                "unit_normalized_first": True,
                "rescaled_by_mean_diff_norm": True,
                "sign_convention": "align with mean(pos)-mean(neg) so subtracting alpha*v pushes negative",
            },
            "rows": summary_rows,
            "per_layer_vector_metadata": steering_meta_by_layer,
            "steering_vectors_path": str(run_dir / "steering_vectors.pt"),
        },
    )

    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "val_accuracy",
                "train_accuracy",
                "val_loss",
                "train_loss",
                "mean_diff_norm",
                "w_norm",
                "sign_flipped",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] wrote probe outputs under {run_dir}")


if __name__ == "__main__":
    main()
