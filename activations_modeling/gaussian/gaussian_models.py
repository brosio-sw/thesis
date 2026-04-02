from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

BASE_DIR = Path("data/steering_masked_activations")
ACTS_DIR = BASE_DIR / "activations"
SCORES_FILE = BASE_DIR / "scores" / "tail_all_sentiment.json"

POS_THRESHOLD = 0.45
NEG_THRESHOLD = 0.55
VAR_FLOOR = 1e-6
SHRINKAGE = 0.05


@dataclass
class RunningDiagStats:
    n: int
    sum_x: np.ndarray
    sum_x2: np.ndarray

    @classmethod
    def init(cls, dim: int) -> "RunningDiagStats":
        return cls(
            n=0,
            sum_x=np.zeros(dim, dtype=np.float64),
            sum_x2=np.zeros(dim, dtype=np.float64),
        )

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.n += x.shape[0]
        self.sum_x += x.sum(axis=0, dtype=np.float64)
        self.sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)


@dataclass
class DiagGaussian:
    mean: torch.Tensor
    var: torch.Tensor
    n_train: int


@dataclass
class LayerGaussianPair:
    positive: DiagGaussian
    negative: DiagGaussian
    direction_definition: str


def discover_parts(acts_dir: Path = ACTS_DIR) -> list[dict]:
    parts = []
    for acts_file in sorted(acts_dir.glob("tail_all_masked_part*.pt")):
        meta_file = acts_dir / f"{acts_file.stem}_meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        parts.append(
            {
                "acts_file": acts_file,
                "meta_file": meta_file,
                "chunk_idx": int(meta.get("chunk_idx", -1)),
            }
        )
    parts.sort(key=lambda x: x["chunk_idx"])
    return parts


def iter_parts(parts: list[dict]) -> Iterable[tuple[dict, dict]]:
    for p in parts:
        acts = torch.load(p["acts_file"], map_location="cpu")
        meta = json.loads(p["meta_file"].read_text())
        yield acts, meta


def extract_negative_scores(meta: dict, per_text_scores: np.ndarray) -> np.ndarray:
    if "target_negative" in meta:
        return np.asarray(meta["target_negative"], dtype=np.float64)
    if "owner_text_idx" in meta:
        idx = np.asarray(meta["owner_text_idx"], dtype=np.int64)
        return per_text_scores[idx]
    raise KeyError("Missing target_negative and owner_text_idx in meta")


def split_pos_neg(neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_mask = neg_scores <= POS_THRESHOLD
    neg_mask = neg_scores >= NEG_THRESHOLD
    return pos_mask, neg_mask


def _finalize_diag(stats: RunningDiagStats) -> DiagGaussian:
    mean = stats.sum_x / max(stats.n, 1)
    var = stats.sum_x2 / max(stats.n, 1) - np.square(mean)
    var = np.maximum(var, VAR_FLOOR)
    avg_var = float(var.mean())
    var = (1.0 - SHRINKAGE) * var + SHRINKAGE * avg_var
    var = np.maximum(var, VAR_FLOOR)
    return DiagGaussian(
        mean=torch.from_numpy(mean.astype(np.float32)),
        var=torch.from_numpy(var.astype(np.float32)),
        n_train=int(stats.n),
    )


def fit_layer_class_gaussians(
    monitored_layers: list[int],
    acts_dir: Path = ACTS_DIR,
    scores_file: Path = SCORES_FILE,
) -> dict[int, LayerGaussianPair]:
    scores = json.loads(scores_file.read_text())
    per_text_scores = np.asarray(scores["per_text"], dtype=np.float64)
    parts = discover_parts(acts_dir)

    first_obj = torch.load(parts[0]["acts_file"], map_location="cpu")
    hidden_dim = int(first_obj[monitored_layers[0]].shape[1])
    del first_obj

    pos_total = 0
    neg_total = 0
    for _, meta in iter_parts(parts):
        y = extract_negative_scores(meta, per_text_scores)
        pos_mask, neg_mask = split_pos_neg(y)
        pos_total += int(pos_mask.sum())
        neg_total += int(neg_mask.sum())

    n_pos_train = pos_total // 2
    n_neg_train = neg_total // 2

    pos_seen = 0
    neg_seen = 0
    pos_stats = {layer: RunningDiagStats.init(hidden_dim) for layer in monitored_layers}
    neg_stats = {layer: RunningDiagStats.init(hidden_dim) for layer in monitored_layers}

    for acts, meta in iter_parts(parts):
        y = extract_negative_scores(meta, per_text_scores)
        pos_mask, neg_mask = split_pos_neg(y)

        pos_idx = np.flatnonzero(pos_mask)
        neg_idx = np.flatnonzero(neg_mask)

        pos_k = max(0, min(n_pos_train - pos_seen, len(pos_idx)))
        neg_k = max(0, min(n_neg_train - neg_seen, len(neg_idx)))

        pos_train_mask = np.zeros(len(y), dtype=bool)
        neg_train_mask = np.zeros(len(y), dtype=bool)

        if pos_k > 0:
            pos_train_mask[pos_idx[:pos_k]] = True
        if neg_k > 0:
            neg_train_mask[neg_idx[:neg_k]] = True

        for layer in monitored_layers:
            x = acts[layer].detach().cpu().numpy().astype(np.float64, copy=False)
            pos_stats[layer].update(x[pos_train_mask])
            neg_stats[layer].update(x[neg_train_mask])

        pos_seen += len(pos_idx)
        neg_seen += len(neg_idx)
        del acts

    out: dict[int, LayerGaussianPair] = {}
    for layer in monitored_layers:
        out[layer] = LayerGaussianPair(
            positive=_finalize_diag(pos_stats[layer]),
            negative=_finalize_diag(neg_stats[layer]),
            direction_definition="direction = mu_pos_train - mu_neg_train",
        )
    return out


def save_gaussian_models(path: Path, models: dict[int, LayerGaussianPair]) -> None:
    payload = {
        "format": "diag_class_gaussians_v1",
        "direction_definition": "direction = mu_pos_train - mu_neg_train",
        "layers": {},
    }
    for layer, m in models.items():
        payload["layers"][str(layer)] = {
            "positive": {
                "mean": m.positive.mean,
                "var": m.positive.var,
                "n_train": m.positive.n_train,
            },
            "negative": {
                "mean": m.negative.mean,
                "var": m.negative.var,
                "n_train": m.negative.n_train,
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_gaussian_models(path: Path) -> dict[int, LayerGaussianPair]:
    obj = torch.load(path, map_location="cpu")
    out: dict[int, LayerGaussianPair] = {}
    for k, v in obj["layers"].items():
        layer = int(k)
        out[layer] = LayerGaussianPair(
            positive=DiagGaussian(
                mean=v["positive"]["mean"].float().cpu(),
                var=v["positive"]["var"].float().cpu(),
                n_train=int(v["positive"]["n_train"]),
            ),
            negative=DiagGaussian(
                mean=v["negative"]["mean"].float().cpu(),
                var=v["negative"]["var"].float().cpu(),
                n_train=int(v["negative"]["n_train"]),
            ),
            direction_definition=obj.get("direction_definition", "direction = mu_pos_train - mu_neg_train"),
        )
    return out


def fit_or_load_gaussian_models(
    cache_path: Path,
    monitored_layers: list[int],
    force_refit: bool = False,
) -> dict[int, LayerGaussianPair]:
    if cache_path.exists() and not force_refit:
        loaded = load_gaussian_models(cache_path)
        missing = [layer for layer in monitored_layers if layer not in loaded]
        if not missing:
            return loaded
        print(
            "[gaussian-models] Cache is missing requested layers "
            f"{missing}. Re-fitting and overwriting {cache_path}."
        )
    models = fit_layer_class_gaussians(monitored_layers=monitored_layers)
    save_gaussian_models(cache_path, models)
    return models
