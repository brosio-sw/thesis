"""
train_probe_llada.py – Train LiSeCo-style linear probes on masked LLaDA activations only.

Uses outputs from:
  data/steering_masked_activations/
    - activations/manifest.json
    - activations/tail_all_masked_partXXX.pt
    - activations/tail_all_masked_partXXX_meta.json
    - texts/tail_all.json

Behavior:
  1) Load masked activations only (from chunked files)
  2) Load original clean texts
  3) Load per-row negativity targets from chunk metadata
     (target = negativity score of the LLaDA reconstruction for that exact
      corrupted/noise-level sample)
  4) Split train/val by original text id so all noise levels / valid rows of a
     sentence stay together
  5) Train one sigmoid linear probe per layer
  6) Save per-layer probe weights + metrics + summaries

Probe model (per layer):
    f_t(x) = sigmoid(W_t^T x + b_t)
Trained with MSE against continuous negativity score in [0, 1].

Usage:
  /home/ambroise/miniconda3/envs/thesis/bin/python train_probe_llada.py --smoke
  /home/ambroise/miniconda3/envs/thesis/bin/python train_probe_llada.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/steering_masked_activations")
ACT_DIR = DATA_DIR / "activations"
TEXT_DIR = DATA_DIR / "texts"

OUT_ROOT = Path("data/probes_llada_masked")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MaskedSourceData:
    texts: list[str]                               # [N_texts]
    activations: dict[int, torch.Tensor]           # layer -> [N_rows, D]
    owner_text_idx: np.ndarray                     # [N_rows]
    mask_ratio: np.ndarray                         # [N_rows]
    target_negative: np.ndarray                    # [N_rows]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_masked_source() -> MaskedSourceData:
    texts_payload = _read_json(TEXT_DIR / "tail_all.json")
    manifest = _read_json(ACT_DIR / "manifest.json")

    texts = list(texts_payload["texts"])

    files = manifest.get("files", [])
    if not files:
        raise RuntimeError("No chunk files listed in activations/manifest.json")

    act_parts: list[dict[int, torch.Tensor]] = []
    owner_parts: list[np.ndarray] = []
    ratio_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    expected_layers = None
    expected_dim = None

    for file_info in files:
        acts_file = ACT_DIR / file_info["acts_file"]
        meta_file = ACT_DIR / file_info["meta_file"]

        if not acts_file.exists():
            raise RuntimeError(f"Missing activation chunk: {acts_file}")
        if not meta_file.exists():
            raise RuntimeError(f"Missing activation metadata chunk: {meta_file}")

        act = torch.load(acts_file, map_location="cpu", weights_only=False)
        act = {int(k): v.float().cpu() for k, v in act.items()}

        meta = _read_json(meta_file)
        owner = np.asarray(meta["owner_text_idx"], dtype=np.int64)
        ratio = np.asarray(meta["mask_ratio"], dtype=np.float32)
        target_negative = np.asarray(meta["target_negative"], dtype=np.float32)

        layer_ids = sorted(act.keys())
        if not layer_ids:
            raise RuntimeError(f"No activations found in chunk: {acts_file}")

        n_rows = int(act[layer_ids[0]].shape[0])

        # Handle fully empty chunks gracefully.
        if n_rows == 0:
            act_dim = None
        else:
            act_dim = int(act[layer_ids[0]].shape[1])

        for layer in layer_ids[1:]:
            if int(act[layer].shape[0]) != n_rows:
                raise RuntimeError(
                    f"Activation row mismatch within chunk {acts_file.name}: "
                    f"layer {layer_ids[0]} has {n_rows}, layer {layer} has {int(act[layer].shape[0])}"
                )
            if n_rows > 0 and int(act[layer].shape[1]) != act_dim:
                raise RuntimeError(
                    f"Activation dim mismatch within chunk {acts_file.name}: "
                    f"layer {layer_ids[0]} has dim {act_dim}, layer {layer} has dim {int(act[layer].shape[1])}"
                )

        if len(owner) != n_rows:
            raise RuntimeError(
                f"Metadata/activation mismatch in {meta_file.name}: owner_text_idx={len(owner)} rows={n_rows}"
            )
        if len(ratio) != n_rows:
            raise RuntimeError(
                f"Metadata/activation mismatch in {meta_file.name}: mask_ratio={len(ratio)} rows={n_rows}"
            )
        if len(target_negative) != n_rows:
            raise RuntimeError(
                f"Metadata/activation mismatch in {meta_file.name}: target_negative={len(target_negative)} rows={n_rows}"
            )

        if expected_layers is None:
            expected_layers = layer_ids
            if n_rows > 0:
                expected_dim = act_dim
        else:
            if layer_ids != expected_layers:
                raise RuntimeError(
                    f"Layer set mismatch in {acts_file.name}: expected {expected_layers}, got {layer_ids}"
                )
            if n_rows > 0 and expected_dim is not None and act_dim != expected_dim:
                raise RuntimeError(
                    f"Hidden dim mismatch in {acts_file.name}: expected {expected_dim}, got {act_dim}"
                )
            if n_rows > 0 and expected_dim is None:
                expected_dim = act_dim

        act_parts.append(act)
        owner_parts.append(owner)
        ratio_parts.append(ratio)
        target_parts.append(target_negative)

    if expected_layers is None:
        raise RuntimeError("No activation layers found across chunks.")

    activations = {
        layer: torch.cat([part[layer] for part in act_parts], dim=0)
        for layer in expected_layers
    }
    owner_text_idx = np.concatenate(owner_parts, axis=0) if owner_parts else np.empty((0,), dtype=np.int64)
    mask_ratio = np.concatenate(ratio_parts, axis=0) if ratio_parts else np.empty((0,), dtype=np.float32)
    target_negative = np.concatenate(target_parts, axis=0) if target_parts else np.empty((0,), dtype=np.float32)

    n_rows = int(activations[expected_layers[0]].shape[0])

    if len(owner_text_idx) != n_rows:
        raise RuntimeError(
            f"Metadata/activation mismatch after concat: owner_text_idx={len(owner_text_idx)} rows={n_rows}"
        )
    if len(mask_ratio) != n_rows:
        raise RuntimeError(
            f"Metadata/activation mismatch after concat: mask_ratio={len(mask_ratio)} rows={n_rows}"
        )
    if len(target_negative) != n_rows:
        raise RuntimeError(
            f"Metadata/activation mismatch after concat: target_negative={len(target_negative)} rows={n_rows}"
        )

    if len(owner_text_idx) > 0 and int(owner_text_idx.max()) >= len(texts):
        raise RuntimeError(
            f"owner_text_idx out of range: max={int(owner_text_idx.max())}, n_texts={len(texts)}"
        )

    manifest_total_rows = manifest.get("total_rows")
    if manifest_total_rows is not None and int(manifest_total_rows) != n_rows:
        raise RuntimeError(
            f"Manifest total_rows mismatch: manifest={manifest_total_rows}, concatenated={n_rows}"
        )

    return MaskedSourceData(
        texts=texts,
        activations=activations,
        owner_text_idx=owner_text_idx,
        mask_ratio=mask_ratio,
        target_negative=target_negative,
    )


def build_index_dataset(src: MaskedSourceData) -> dict:
    records = []
    first_layer = sorted(src.activations.keys())[0]
    n_rows = int(src.activations[first_layer].shape[0])

    for row_idx in range(n_rows):
        owner = int(src.owner_text_idx[row_idx])
        records.append(
            {
                "row_idx": row_idx,
                "owner_text_idx": owner,
                "group_id": f"text::{owner}",
                "mask_ratio": float(src.mask_ratio[row_idx]),
                "text": src.texts[owner],
                "sentiment_score": float(src.target_negative[row_idx]),
            }
        )

    return {"examples": records}


def grouped_split_by_text(
    records: list[dict],
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    by_group: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_group.setdefault(rec["group_id"], []).append(i)

    group_ids = list(by_group.keys())
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_val_groups = max(1, int(round(n_groups * val_frac)))
    if n_groups - n_val_groups < 1:
        n_val_groups = n_groups - 1
    if n_val_groups <= 0:
        raise RuntimeError(f"Too few groups for split: {n_groups}")

    val_groups = set(group_ids[:n_val_groups])

    train_idx: list[int] = []
    val_idx: list[int] = []

    for gid, rec_idxs in by_group.items():
        if gid in val_groups:
            val_idx.extend(rec_idxs)
        else:
            train_idx.extend(rec_idxs)

    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def build_layer_matrix(
    src: MaskedSourceData,
    records: list[dict],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[float] = []

    acts = src.activations[layer]
    for rec in records:
        row_idx = int(rec["row_idx"])
        xs.append(acts[row_idx])
        ys.append(float(rec["sentiment_score"]))

    X = torch.stack(xs, dim=0).float()
    y = torch.tensor(ys, dtype=torch.float32)
    return X, y


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = float(np.sqrt((yt ** 2).sum()) * np.sqrt((yp ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((yt * yp).sum() / denom)


def train_single_layer_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict, dict]:
    _seed_everything(seed)
    device = "cpu"

    X_train = X[train_idx].to(device)
    y_train = y[train_idx].to(device)
    X_val = X[val_idx].to(device)
    y_val = y[val_idx].to(device)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best = {"val_mse": float("inf"), "state_dict": None, "epoch": -1}

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            pred = torch.sigmoid(model(xb)).squeeze(-1)
            loss = criterion(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = torch.sigmoid(model(X_val)).squeeze(-1)
            val_mse = float(nn.functional.mse_loss(pred_val, y_val).item())

        if val_mse < best["val_mse"]:
            best["val_mse"] = val_mse
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch

    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_val = torch.sigmoid(model(X_val)).squeeze(-1).cpu().numpy().astype(np.float32)
        y_val_np = y_val.cpu().numpy().astype(np.float32)

    metrics = {
        "best_epoch": int(best["epoch"]),
        "val_mse": float(best["val_mse"]),
        "val_mae": float(np.mean(np.abs(pred_val - y_val_np))),
        "val_r2": float(_r2_score(y_val_np, pred_val)),
        "val_pearson": float(_pearson(y_val_np, pred_val)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }

    weight = model.weight.detach().cpu().view(-1).float()
    bias = model.bias.detach().cpu().view(-1).float()

    artifact = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": int(X.shape[1]),
        "weight": weight,
        "bias": bias,
    }
    return metrics, artifact


def train_probe_family(
    src: MaskedSourceData,
    out_dir: Path,
    layers_to_train: list[int],
    val_frac: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    subset_frac: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    index_payload = build_index_dataset(src)
    records = index_payload["examples"]

    if subset_frac < 1.0:
        rng = random.Random(seed)
        by_group: dict[str, list[dict]] = {}
        for r in records:
            by_group.setdefault(r["group_id"], []).append(r)

        group_ids = list(by_group.keys())
        rng.shuffle(group_ids)

        k = max(2, int(round(len(group_ids) * subset_frac)))
        chosen = set(group_ids[:k])

        subset_records = []
        for gid in chosen:
            subset_records.extend(by_group[gid])
        records = subset_records

    train_idx, val_idx = grouped_split_by_text(records, val_frac=val_frac, seed=seed)

    with open(out_dir / "dataset_index.json", "w") as f:
        json.dump(
            {
                "n_examples": len(records),
                "val_frac": val_frac,
                "subset_frac": subset_frac,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "examples": records,
            },
            f,
            indent=2,
        )

    summary_rows = []
    per_layer_metrics = {}

    for layer in layers_to_train:
        X, y = build_layer_matrix(src, records, layer)
        layer_dir = out_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        metrics, artifact = train_single_layer_probe(
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed + layer,
        )

        torch.save(artifact, layer_dir / "probe.pt")
        torch.save(artifact["state_dict"], layer_dir / "probe_state_dict.pt")
        torch.save(
            {
                "weight": artifact["weight"],
                "bias": artifact["bias"],
                "input_dim": artifact["input_dim"],
            },
            layer_dir / "probe_weight_bias.pt",
        )

        with open(layer_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        per_layer_metrics[str(layer)] = metrics
        summary_rows.append(
            {
                "layer": layer,
                "val_mse": metrics["val_mse"],
                "val_mae": metrics["val_mae"],
                "val_r2": metrics["val_r2"],
                "val_pearson": metrics["val_pearson"],
                "best_epoch": metrics["best_epoch"],
            }
        )

    summary_rows_sorted = sorted(summary_rows, key=lambda x: x["layer"])

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "layers_trained": layers_to_train,
                "rows": summary_rows_sorted,
                "per_layer": per_layer_metrics,
            },
            f,
            indent=2,
        )

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "val_mse", "val_mae", "val_r2", "val_pearson", "best_epoch"],
        )
        writer.writeheader()
        writer.writerows(summary_rows_sorted)

    return {
        "n_examples": len(records),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "layers_trained": layers_to_train,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LiSeCo-style linear probes on masked LLaDA activations.")
    p.add_argument("--smoke", action="store_true", help="Small subset + 1-2 layers smoke test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--layers", type=str, default="all", help="Comma list (e.g. 0,23) or 'all'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    run_name = "smoke_test" if args.smoke else "full_run"
    out_root = OUT_ROOT / run_name
    probes_out = out_root / "masked_only_probes"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Run mode: {run_name}")
    print(f"[setup] Loading masked source from {DATA_DIR}/")

    src = load_masked_source()

    all_layers = sorted(src.activations.keys())
    if args.layers == "all":
        layers = all_layers
    else:
        layers = sorted({int(x.strip()) for x in args.layers.split(",") if x.strip()})

    subset_frac = 1.0
    if args.smoke:
        subset_frac = 0.05
        if args.layers == "all":
            layers = [0, 23] if 23 in all_layers else [all_layers[0], all_layers[-1]]

    first_layer = all_layers[0]
    manifest = {
        "run_name": run_name,
        "smoke": bool(args.smoke),
        "subset_frac": subset_frac,
        "all_layers": all_layers,
        "layers_to_train": layers,
        "n_texts": len(src.texts),
        "n_examples": int(src.activations[first_layer].shape[0]),
        "activation_dim": int(src.activations[first_layer].shape[1]),
        "score_mean": float(src.target_negative.mean()) if len(src.target_negative) > 0 else None,
        "score_std": float(src.target_negative.std()) if len(src.target_negative) > 0 else None,
        "mask_ratio_unique": sorted({float(x) for x in src.mask_ratio.tolist()}),
        "target_definition": "per-row negativity score of the LLaDA reconstruction from corrupted input",
    }

    with open(out_root / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("[check] Alignment verified.")
    print(f"[check] n_texts={manifest['n_texts']}")
    print(f"[check] n_examples={manifest['n_examples']}")
    print(f"[check] activation_dim={manifest['activation_dim']}")
    print(f"[check] layers={layers}")

    info = train_probe_family(
        src=src,
        out_dir=probes_out,
        layers_to_train=layers,
        val_frac=args.val_frac,
        seed=args.seed,
        epochs=(25 if args.smoke else args.epochs),
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        subset_frac=subset_frac,
    )

    with open(out_root / "run_summary.json", "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "smoke": bool(args.smoke),
                "layers": layers,
                "subset_frac": subset_frac,
                "masked_only": info,
            },
            f,
            indent=2,
        )

    print("[done] Probe training complete.")
    print(f"[done] Outputs: {out_root}")
    print(f"[done] Dataset built: n={info['n_examples']}")


if __name__ == "__main__":
    main()