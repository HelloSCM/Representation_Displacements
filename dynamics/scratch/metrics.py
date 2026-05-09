import gc
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from feature_utils import extract_features_to_numpy
from utils import append_csv_row, safe_name

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None


METRIC_NAMES = ["cross_modality", "cross_object", "direct_similarity"]


def _pca_or_center_layer(
    X: np.ndarray,
    proj_dim: int,
    seed: int,
    device: torch.device,
):
    """Return centered/projection data and projected mean, matching the old PCA-OPA code.

    If proj_dim <= 0, PCA is disabled and we simply center in the original space.
    """
    if proj_dim is not None and proj_dim > 0:
        if PCA is None:
            raise ImportError("scikit-learn is required when proj_dim > 0")
        n_components = min(proj_dim, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components, random_state=seed, whiten=True)
        centered = pca.fit_transform(X)
        mu_orig = pca.mean_
        W = pca.components_
        scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
        mu_proj = np.dot(mu_orig, W.T) / scale
        centered_t = torch.from_numpy(centered).float().to(device)
        mu_t = torch.from_numpy(mu_proj).float().to(device).view(1, -1)
        return centered_t, mu_t

    mu = X.mean(axis=0, keepdims=True)
    centered = X - mu
    centered_t = torch.from_numpy(centered).float().to(device)
    mu_t = torch.from_numpy(mu).float().to(device)
    return centered_t, mu_t


def _prepare_layers(features: np.ndarray, proj_dim: int, seed: int, device: torch.device):
    # features: [N, L, D]
    layers = []
    for layer_idx in range(features.shape[1]):
        layers.append(_pca_or_center_layer(features[:, layer_idx, :], proj_dim, seed, device))
    return layers


@torch.no_grad()
def _batch_compute_metrics(
    A_centered: torch.Tensor,
    A_mu: torch.Tensor,
    B_batch_centered: torch.Tensor,
    B_batch_mu: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
):
    """Batch OPA-align B layers to one A layer and compute the three metrics."""
    # B_batch_centered: [L, N, D], A_centered: [N, D]
    M = torch.matmul(B_batch_centered.transpose(1, 2), A_centered.unsqueeze(0))
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = torch.matmul(U, Vh)

    B_final = torch.matmul(B_batch_centered, R) + B_batch_mu
    A_final = A_centered + A_mu

    A1, A2 = A_final[idx1], A_final[idx2]
    B1, B2 = B_final[:, idx1, :], B_final[:, idx2, :]

    def cos_sim(x, y):
        return F.cosine_similarity(x, y, dim=-1, eps=1e-8).mean(dim=-1)

    direct = cos_sim(A1.unsqueeze(0), B1)
    cross_modality = cos_sim((A1 - A2).unsqueeze(0), B1 - B2)
    cross_object = cos_sim(A1.unsqueeze(0) - B1, A2.unsqueeze(0) - B2)

    return {
        "cross_modality": cross_modality.detach().cpu().numpy(),
        "cross_object": cross_object.detach().cpu().numpy(),
        "direct_similarity": direct.detach().cpu().numpy(),
    }


@torch.no_grad()
def compute_layerwise_metrics(
    current_features: np.ndarray,
    reference_features: np.ndarray,
    proj_dim: int = -1,
    pair_sample_size: int = 1000,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Compute layer-by-layer metrics: current layers x reference layers."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if current_features.shape[0] != reference_features.shape[0]:
        raise ValueError(
            f"Feature N mismatch: current={current_features.shape[0]}, reference={reference_features.shape[0]}"
        )
    if proj_dim <= 0 and current_features.shape[2] != reference_features.shape[2]:
        raise ValueError("Feature dims differ; set --metric_proj_dim > 0 to compare via PCA projection.")

    n = current_features.shape[0]
    rng = np.random.default_rng(seed)
    idx1 = torch.from_numpy(rng.choice(n, pair_sample_size, replace=True)).long().to(device)
    idx2 = torch.from_numpy(rng.choice(n, pair_sample_size, replace=True)).long().to(device)

    A_layers = _prepare_layers(current_features, proj_dim, seed, device)
    B_layers = _prepare_layers(reference_features, proj_dim, seed, device)

    B_centered = torch.stack([x[0] for x in B_layers], dim=0)  # [L2, N, D]
    B_mu = torch.stack([x[1] for x in B_layers], dim=0)        # [L2, 1, D]

    L1 = len(A_layers)
    L2 = len(B_layers)
    result = {name: np.zeros((L1, L2), dtype=np.float32) for name in METRIC_NAMES}

    for i, (A_centered, A_mu) in enumerate(A_layers):
        row = _batch_compute_metrics(A_centered, A_mu, B_centered, B_mu, idx1, idx2)
        for name in METRIC_NAMES:
            result[name][i, :] = row[name]

    del A_layers, B_layers, B_centered, B_mu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def save_metric_matrices(
    mats: Dict[str, np.ndarray],
    out_dir: str,
    current_label: str,
    reference_label: str,
    epoch: int,
    heatmap_center: float = 0.3,
) -> Dict[str, Dict[str, str]]:
    """Save CSV and PDF heatmaps for one comparison."""
    saved = {}
    comparison = f"{safe_name(current_label)}-VS-{safe_name(reference_label)}"
    layer_rows = [f"layer{i}" for i in range(next(iter(mats.values())).shape[0])]
    layer_cols = [f"layer{i}" for i in range(next(iter(mats.values())).shape[1])]

    for metric_name, mat in mats.items():
        metric_dir = os.path.join(out_dir, comparison, metric_name)
        csv_dir = os.path.join(metric_dir, "csv")
        pdf_dir = os.path.join(metric_dir, "pdf")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        df = pd.DataFrame(mat, index=layer_rows, columns=layer_cols)
        csv_path = os.path.join(csv_dir, f"epoch_{epoch:03d}.csv")
        pdf_path = os.path.join(pdf_dir, f"epoch_{epoch:03d}.pdf")
        df.to_csv(csv_path)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap="RdBu_r", center=heatmap_center)
        plt.title(f"{metric_name}\n{current_label} vs {reference_label} | epoch {epoch:03d}")
        plt.xlabel(f"{reference_label} layers")
        plt.ylabel(f"{current_label} layers")
        plt.tight_layout()
        plt.savefig(pdf_path)
        plt.close()

        saved[metric_name] = {"csv": csv_path, "pdf": pdf_path}
    return saved


def summarize_metric_matrix(mat: np.ndarray) -> Dict[str, float]:
    diag_len = min(mat.shape[0], mat.shape[1])
    diag = np.diag(mat[:diag_len, :diag_len])
    return {
        "matrix_mean": float(np.mean(mat)),
        "matrix_max": float(np.max(mat)),
        "diag_mean": float(np.mean(diag)),
        "last_layer_pair": float(mat[-1, -1]),
    }


def _load_or_extract_reference_features(
    spec: Dict,
    val_dir: str,
    cache_dir: Optional[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_samples: int,
):
    cache_path = None
    if cache_dir is not None and spec.get("checkpoint_path") is None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"{safe_name(spec['label'])}_max{max_samples}_all_layers.npy"
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    features = extract_features_to_numpy(
        model_name=spec["model_name"],
        data_dir=val_dir,
        checkpoint_path=spec.get("checkpoint_path"),
        pretrained=spec.get("pretrained", True),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_samples=max_samples,
        all_layers=True,
        desc=f"features {spec['label']}",
    )
    if cache_path is not None:
        np.save(cache_path, features)
    return features


def compute_metric_suite(
    current_spec: Dict,
    reference_specs: List[Dict],
    val_dir: str,
    out_dir: str,
    epoch: int,
    batch_size: int = 64,
    num_workers: int = 8,
    proj_dim: int = -1,
    pair_sample_size: int = 1000,
    seed: int = 42,
    device: Optional[torch.device] = None,
    max_samples: int = -1,
    cache_dir: Optional[str] = None,
    summary_csv: Optional[str] = None,
) -> None:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    current_features = extract_features_to_numpy(
        model_name=current_spec["model_name"],
        data_dir=val_dir,
        checkpoint_path=current_spec.get("checkpoint_path"),
        pretrained=current_spec.get("pretrained", False),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_samples=max_samples,
        all_layers=True,
        desc=f"features {current_spec['label']}",
    )

    for ref in reference_specs:
        ref_features = _load_or_extract_reference_features(
            ref,
            val_dir=val_dir,
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            max_samples=max_samples,
        )
        mats = compute_layerwise_metrics(
            current_features=current_features,
            reference_features=ref_features,
            proj_dim=proj_dim,
            pair_sample_size=pair_sample_size,
            seed=seed,
            device=device,
        )
        save_metric_matrices(
            mats=mats,
            out_dir=out_dir,
            current_label=current_spec["label"],
            reference_label=ref["label"],
            epoch=epoch,
        )
        if summary_csv is not None:
            for metric_name, mat in mats.items():
                summary = summarize_metric_matrix(mat)
                row = {
                    "epoch": epoch,
                    "current": current_spec["label"],
                    "reference": ref["label"],
                    "metric": metric_name,
                    "proj_dim": proj_dim,
                    "pair_sample_size": pair_sample_size,
                    "max_samples": max_samples,
                    **summary,
                }
                append_csv_row(summary_csv, row)

    del current_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
