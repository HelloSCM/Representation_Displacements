#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consensus / private decomposition for multiple ImageNet-1k vision representations.

This version saves every plotted curve/error-band data to CSV files under:
  results/plot_csv/

Main pipeline:
  1. Fit mean + whitening for each model on train features.
  2. Compute pairwise whitened cross-covariances on train features.
  3. Run GPA from the train cross-covariance matrices.
  4. Build cross-model consensus matrix after GPA.
  5. Define common = top-r eigenspace, private = bottom-r eigenspace.
  6. Compute class-level cross-model difference similarity on val class centroids.
  7. Compute projection-strength metrics, Fisher ratio, HSIC.
  8. Optionally train/evaluate frozen and decomposed linear probes.
  9. Optionally evaluate CLIP zero-shot with OpenCLIP text features and visual.proj.

Expected data layout:
  train_root/model_safe_name/features.npy
  train_root/model_safe_name/labels.npy
  val_root/model_safe_name/features.npy
  val_root/model_safe_name/labels.npy
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_model_name(name: str) -> str:
    name = name.strip().replace(".", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name


def read_model_names(path: Path) -> Tuple[List[str], List[str]]:
    raw_names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                raw_names.append(line)
    return raw_names, [safe_model_name(x) for x in raw_names]


def parse_ranks(ranks: str, d: int) -> List[int]:
    vals: List[int] = []
    for part in ranks.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("%"):
            vals.append(max(1, int(round(float(part[:-1]) / 100.0 * d))))
        else:
            vals.append(int(part))
    vals = sorted(set([r for r in vals if 1 <= r <= d]))
    if not vals:
        raise ValueError("No valid ranks were provided.")
    return vals


def batched_ranges(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def load_npy_mmap(path: Path) -> np.ndarray:
    return np.load(str(path), mmap_mode="r")


def as_torch_batch(x_np: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(x_np), device=device)
    if x.dtype != dtype:
        x = x.to(dtype)
    return x


def to_numpy(x: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return x.detach().cpu().numpy().astype(dtype, copy=False)


def maybe_cuda_empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class ModelPaths:
    raw_name: str
    safe_name: str
    train_features: Path
    train_labels: Path
    val_features: Path
    val_labels: Path


@dataclass
class PreprocessParams:
    mean: np.ndarray
    W: np.ndarray
    W_inv: np.ndarray
    eigvals: np.ndarray


# -----------------------------
# Data discovery and validation
# -----------------------------

def build_model_paths(args: argparse.Namespace) -> List[ModelPaths]:
    raw_names, safe_names = read_model_names(Path(args.model_list))
    paths: List[ModelPaths] = []
    train_root = Path(args.train_root)
    val_root = Path(args.val_root)
    for raw, safe in zip(raw_names, safe_names):
        train_dir = train_root / safe
        val_dir = val_root / safe
        actual_name = safe
        if not train_dir.exists() and (train_root / raw).exists():
            train_dir = train_root / raw
            val_dir = val_root / raw
            actual_name = raw
        paths.append(ModelPaths(
            raw_name=raw,
            safe_name=actual_name,
            train_features=train_dir / "features.npy",
            train_labels=train_dir / "labels.npy",
            val_features=val_dir / "features.npy",
            val_labels=val_dir / "labels.npy",
        ))
    missing = []
    for p in paths:
        for q in [p.train_features, p.train_labels, p.val_features, p.val_labels]:
            if not q.exists():
                missing.append(str(q))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return paths


def validate_shapes_and_labels(paths: List[ModelPaths], strict_labels: bool = True) -> Tuple[int, int, int, int, np.ndarray, np.ndarray]:
    log("Validating feature shapes and labels...")
    train0 = load_npy_mmap(paths[0].train_features)
    val0 = load_npy_mmap(paths[0].val_features)
    n_train, d = train0.shape
    n_val, d_val = val0.shape
    if d != d_val:
        raise ValueError(f"Train dim {d} != val dim {d_val}")
    y_train0 = np.load(paths[0].train_labels).astype(np.int64)
    y_val0 = np.load(paths[0].val_labels).astype(np.int64)

    for p in paths:
        xtr = load_npy_mmap(p.train_features)
        xva = load_npy_mmap(p.val_features)
        if xtr.shape != (n_train, d):
            raise ValueError(f"Shape mismatch for {p.safe_name} train: {xtr.shape} vs {(n_train, d)}")
        if xva.shape != (n_val, d):
            raise ValueError(f"Shape mismatch for {p.safe_name} val: {xva.shape} vs {(n_val, d)}")
        if strict_labels:
            yt = np.load(p.train_labels)
            yv = np.load(p.val_labels)
            if not np.array_equal(yt, y_train0):
                raise ValueError(f"Train labels differ for {p.safe_name}")
            if not np.array_equal(yv, y_val0):
                raise ValueError(f"Val labels differ for {p.safe_name}")
    num_classes = int(max(y_train0.max(), y_val0.max())) + 1
    log(f"Validated {len(paths)} models. train={n_train}, val={n_val}, d={d}, classes={num_classes}")
    return n_train, n_val, d, num_classes, y_train0, y_val0


# -----------------------------
# Whitening and GPA
# -----------------------------

def compute_mean_cov_for_model(features_path: Path, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    arr = load_npy_mmap(features_path)
    n, d = arr.shape
    sum_x = torch.zeros(d, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(d, d, dtype=torch.float64, device=device)
    for start, end in tqdm(list(batched_ranges(n, batch_size)), desc=f"mean/cov {features_path.parent.name}"):
        xb = as_torch_batch(arr[start:end], device=device, dtype=torch.float32)
        sum_x += xb.sum(dim=0, dtype=torch.float64)
        sum_xx += (xb.T @ xb).to(torch.float64)
        del xb
    mean = sum_x / float(n)
    cov = (sum_xx - float(n) * torch.outer(mean, mean)) / float(n - 1)
    cov = 0.5 * (cov + cov.T)
    return to_numpy(mean, np.float64), to_numpy(cov, np.float64)


def make_whitening_from_cov(cov: np.ndarray, eps_abs: float, eps_rel: float) -> PreprocessParams:
    cov_t = torch.as_tensor(cov, dtype=torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(cov_t)
    eigvals = torch.clamp(eigvals, min=0.0)
    max_eval = float(torch.max(eigvals).item())
    floor = max(float(eps_abs), float(eps_rel) * max_eval)
    evals = torch.clamp(eigvals, min=floor)
    W = (eigvecs * torch.rsqrt(evals).unsqueeze(0)) @ eigvecs.T
    W_inv = (eigvecs * torch.sqrt(evals).unsqueeze(0)) @ eigvecs.T
    return PreprocessParams(
        mean=np.empty((0,), dtype=np.float32),
        W=to_numpy(W, np.float32),
        W_inv=to_numpy(W_inv, np.float32),
        eigvals=to_numpy(eigvals, np.float64),
    )


def fit_or_load_preprocess(args: argparse.Namespace, paths: List[ModelPaths], d: int, device: torch.device) -> Dict[str, PreprocessParams]:
    out_dir = Path(args.out_dir) / "preprocess"
    ensure_dir(out_dir)
    params: Dict[str, PreprocessParams] = {}
    for p in paths:
        out_path = out_dir / f"{p.safe_name}.npz"
        if out_path.exists() and not args.force_recompute:
            data = np.load(out_path)
            params[p.safe_name] = PreprocessParams(
                mean=data["mean"].astype(np.float32),
                W=data["W"].astype(np.float32),
                W_inv=data["W_inv"].astype(np.float32),
                eigvals=data["eigvals"].astype(np.float64),
            )
            log(f"Loaded preprocess params for {p.safe_name}")
            continue
        log(f"Fitting mean + whitening for {p.safe_name}")
        mean, cov = compute_mean_cov_for_model(p.train_features, args.batch_size_stats, device)
        pp = make_whitening_from_cov(cov, args.whiten_eps_abs, args.whiten_eps_rel)
        pp.mean = mean.astype(np.float32)
        np.savez_compressed(out_path, mean=pp.mean, W=pp.W, W_inv=pp.W_inv, eigvals=pp.eigvals)
        params[p.safe_name] = pp
        log(f"Saved preprocess params to {out_path}")
        maybe_cuda_empty_cache(device)
    return params


def compute_or_load_whitened_crosscov(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    params: Dict[str, PreprocessParams],
    n_train: int,
    d: int,
    device: torch.device,
) -> np.ndarray:
    out_path = Path(args.out_dir) / "whitened_crosscov_train.npy"
    if out_path.exists() and not args.force_recompute:
        log(f"Loaded whitened train cross-covariance from {out_path}")
        return np.load(out_path)

    log("Computing pairwise whitened train cross-covariances.")
    M = len(paths)
    C = torch.zeros(M, M, d, d, dtype=torch.float64, device=device)
    means = [torch.as_tensor(params[p.safe_name].mean, dtype=torch.float32, device=device) for p in paths]
    Ws = [torch.as_tensor(params[p.safe_name].W, dtype=torch.float32, device=device) for p in paths]
    arrays = [load_npy_mmap(p.train_features) for p in paths]

    for start, end in tqdm(list(batched_ranges(n_train, args.batch_size_pairwise)), desc="whitened cross-cov"):
        Ys: List[torch.Tensor] = []
        for m, arr in enumerate(arrays):
            xb = as_torch_batch(arr[start:end], device=device, dtype=torch.float32)
            yb = (xb - means[m]) @ Ws[m]
            Ys.append(yb)
            del xb
        for i in range(M):
            Yi = Ys[i]
            for j in range(i, M):
                block = Yi.T @ Ys[j]
                C[i, j] += block.to(torch.float64)
                if j != i:
                    C[j, i] += block.T.to(torch.float64)
        del Ys

    C /= float(n_train - 1)
    C_np = to_numpy(C, np.float64)
    np.save(out_path, C_np)
    log(f"Saved whitened train cross-covariance to {out_path}")
    maybe_cuda_empty_cache(device)
    return C_np


def gpa_objective(C: np.ndarray, R: np.ndarray) -> float:
    M = C.shape[0]
    obj = 0.0
    count = 0
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            obj += float(np.trace(R[a].T @ C[a, b] @ R[b]))
            count += 1
    return obj / max(count, 1)


def run_or_load_gpa(args: argparse.Namespace, C: np.ndarray) -> np.ndarray:
    out_path = Path(args.out_dir) / "gpa_rotations.npy"
    log_path = Path(args.out_dir) / "gpa_log.csv"
    if out_path.exists() and not args.force_recompute:
        log(f"Loaded GPA rotations from {out_path}")
        return np.load(out_path)

    M, _, d, _ = C.shape
    R = np.stack([np.eye(d, dtype=np.float64) for _ in range(M)], axis=0)
    rows = []
    last_obj = gpa_objective(C, R)
    rows.append({"iter": 0, "objective": last_obj, "delta_R": np.nan})
    log(f"GPA iter 0 objective={last_obj:.8f}")

    for it in range(1, args.gpa_iters + 1):
        R_new = np.empty_like(R)
        for m in range(M):
            B = np.zeros((d, d), dtype=np.float64)
            for k in range(M):
                B += C[m, k] @ R[k]
            B /= float(M)
            U, _, Vt = np.linalg.svd(B, full_matrices=False)
            R_new[m] = U @ Vt
        delta = float(np.sqrt(np.mean((R_new - R) ** 2)))
        R = R_new
        obj = gpa_objective(C, R)
        rows.append({"iter": it, "objective": obj, "delta_R": delta})
        log(f"GPA iter {it} objective={obj:.8f}, delta_R={delta:.6e}")
        if abs(obj - last_obj) <= args.gpa_tol * max(1.0, abs(last_obj)) and delta <= math.sqrt(args.gpa_tol):
            log("GPA converged.")
            break
        last_obj = obj

    np.save(out_path, R.astype(np.float32))
    pd.DataFrame(rows).to_csv(log_path, index=False)
    log(f"Saved GPA rotations to {out_path}")
    return R.astype(np.float32)


def compute_or_load_consensus(args: argparse.Namespace, C: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_path = Path(args.out_dir) / "consensus_spectrum.npz"
    if out_path.exists() and not args.force_recompute:
        data = np.load(out_path)
        log(f"Loaded consensus spectrum from {out_path}")
        return data["S_cross"], data["eigvals_desc"], data["eigvecs_desc"]

    log("Computing GPA-space cross-model consensus matrix.")
    M, _, d, _ = C.shape
    S = np.zeros((d, d), dtype=np.float64)
    count = 0
    R64 = R.astype(np.float64)
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            Zab = R64[a].T @ C[a, b] @ R64[b]
            S += 0.5 * (Zab + Zab.T)
            count += 1
    S /= float(count)
    S = 0.5 * (S + S.T)

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals_desc = eigvals[idx]
    eigvecs_desc = eigvecs[:, idx]

    np.savez_compressed(
        out_path,
        S_cross=S.astype(np.float32),
        eigvals_desc=eigvals_desc.astype(np.float64),
        eigvecs_desc=eigvecs_desc.astype(np.float32),
    )
    log(f"Saved consensus spectrum to {out_path}")
    return S.astype(np.float32), eigvals_desc.astype(np.float64), eigvecs_desc.astype(np.float32)


# -----------------------------
# Feature transforms
# -----------------------------

def make_torch_params(pp: PreprocessParams, R: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    mean = torch.as_tensor(pp.mean, dtype=torch.float32, device=device)
    W = torch.as_tensor(pp.W, dtype=torch.float32, device=device)
    W_inv = torch.as_tensor(pp.W_inv, dtype=torch.float32, device=device)
    Rt = torch.as_tensor(R.T, dtype=torch.float32, device=device)
    WR = W @ torch.as_tensor(R, dtype=torch.float32, device=device)
    return {"mean": mean, "W": W, "W_inv": W_inv, "Rt": Rt, "WR": WR}


def raw_to_z(x: torch.Tensor, tp: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (x - tp["mean"]) @ tp["WR"]


def z_component_to_original(z: torch.Tensor, V: torch.Tensor, tp: Dict[str, torch.Tensor], add_mean: bool = True) -> torch.Tensor:
    B = V.T @ tp["Rt"] @ tp["W_inv"]
    x = (z @ V) @ B
    if add_mean:
        x = x + tp["mean"]
    return x


def raw_to_component_original(x: torch.Tensor, V: torch.Tensor, tp: Dict[str, torch.Tensor], add_mean: bool = True) -> torch.Tensor:
    z = raw_to_z(x, tp)
    return z_component_to_original(z, V, tp, add_mean=add_mean)


def make_original_component_transform(V: torch.Tensor, tp: Dict[str, torch.Tensor], add_mean: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """Precompute the low-rank inverse map for repeated batch transforms."""
    B = V.T @ tp["Rt"] @ tp["W_inv"]
    mean = tp["mean"]

    def transform(x: torch.Tensor) -> torch.Tensor:
        z = raw_to_z(x, tp)
        out = (z @ V) @ B
        if add_mean:
            out = out + mean
        return out

    return transform


# -----------------------------
# Val class centroids and class-level similarity
# -----------------------------

def compute_or_load_val_centroids(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    params: Dict[str, PreprocessParams],
    R: np.ndarray,
    n_val: int,
    d: int,
    num_classes: int,
    y_val: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gpa_path = Path(args.out_dir) / "val_class_centroids_gpa.npy"
    orig_path = Path(args.out_dir) / "val_class_centroids_orig.npy"
    counts_path = Path(args.out_dir) / "val_class_counts.npy"
    if gpa_path.exists() and orig_path.exists() and counts_path.exists() and not args.force_recompute:
        log("Loaded cached val class centroids.")
        return np.load(gpa_path), np.load(orig_path), np.load(counts_path)

    log("Computing val class centroids in GPA space and original space.")
    M = len(paths)
    C_gpa = np.zeros((M, num_classes, d), dtype=np.float32)
    C_orig = np.zeros((M, num_classes, d), dtype=np.float32)
    counts = np.bincount(y_val, minlength=num_classes).astype(np.float64)
    y_all = torch.as_tensor(y_val, dtype=torch.long, device=device)

    for mi, p in enumerate(paths):
        log(f"Val centroids for {p.safe_name}")
        arr = load_npy_mmap(p.val_features)
        tp = make_torch_params(params[p.safe_name], R[mi], device)
        sums_z = torch.zeros(num_classes, d, dtype=torch.float64, device=device)
        sums_x = torch.zeros(num_classes, d, dtype=torch.float64, device=device)
        for start, end in tqdm(list(batched_ranges(n_val, args.batch_size_transform)), desc=f"centroids {p.safe_name}"):
            xb = as_torch_batch(arr[start:end], device=device, dtype=torch.float32)
            labels = y_all[start:end]
            z = raw_to_z(xb, tp)
            sums_z.index_add_(0, labels, z.to(torch.float64))
            sums_x.index_add_(0, labels, xb.to(torch.float64))
            del xb, z, labels
        denom = torch.as_tensor(counts[:, None], dtype=torch.float64, device=device)
        C_gpa[mi] = to_numpy(sums_z / denom, np.float32)
        C_orig[mi] = to_numpy(sums_x / denom, np.float32)
        maybe_cuda_empty_cache(device)

    np.save(gpa_path, C_gpa)
    np.save(orig_path, C_orig)
    np.save(counts_path, counts)
    log("Saved val class centroids.")
    return C_gpa, C_orig, counts


def center_rows(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def difference_cosine_mean(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cosine between all class-pair difference vectors using Gram matrices."""
    A = A.astype(np.float64, copy=False)
    B = B.astype(np.float64, copy=False)
    Gaa = A @ A.T
    Gbb = B @ B.T
    Gab = A @ B.T
    diag_ab = np.diag(Gab)
    num = diag_ab[:, None] + diag_ab[None, :] - Gab - Gab.T
    da2 = np.diag(Gaa)[:, None] + np.diag(Gaa)[None, :] - 2.0 * Gaa
    db2 = np.diag(Gbb)[:, None] + np.diag(Gbb)[None, :] - 2.0 * Gbb
    denom = np.sqrt(np.maximum(da2 * db2, eps))
    cos = num / denom
    idx = np.triu_indices(A.shape[0], k=1)
    return float(np.mean(cos[idx]))


def run_cross_model_similarity(args: argparse.Namespace, paths: List[ModelPaths], C_gpa: np.ndarray, eigvecs_desc: np.ndarray, ranks: List[int]) -> pd.DataFrame:
    out_csv = Path(args.out_dir) / "cross_model_difference_similarity.csv"
    log("Computing class-level cross-model difference similarity.")
    rows = []
    M = len(paths)
    d = C_gpa.shape[-1]
    centered = np.stack([center_rows(C_gpa[m]) for m in range(M)], axis=0)
    # raw is rank-independent but repeated for plot convenience.
    raw_cache: Dict[Tuple[int, int], float] = {}
    for a, b in itertools.combinations(range(M), 2):
        raw_cache[(a, b)] = difference_cosine_mean(centered[a], centered[b], eps=args.cos_eps)

    for r in tqdm(ranks, desc="cross-model similarity ranks"):
        Vtop = eigvecs_desc[:, :r]
        Vbot = eigvecs_desc[:, -r:]
        for a, b in itertools.combinations(range(M), 2):
            rho_raw = raw_cache[(a, b)]
            rho_c = difference_cosine_mean(centered[a] @ Vtop, centered[b] @ Vtop, eps=args.cos_eps)
            rho_p = difference_cosine_mean(centered[a] @ Vbot, centered[b] @ Vbot, eps=args.cos_eps)
            rows.extend([
                {"rank": r, "rank_frac": r / d, "component": "raw", "model_a": paths[a].safe_name, "model_b": paths[b].safe_name, "rho": rho_raw},
                {"rank": r, "rank_frac": r / d, "component": "common_top", "model_a": paths[a].safe_name, "model_b": paths[b].safe_name, "rho": rho_c},
                {"rank": r, "rank_frac": r / d, "component": "private_bottom", "model_a": paths[a].safe_name, "model_b": paths[b].safe_name, "rho": rho_p},
            ])
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv}")
    return df


# -----------------------------
# Projection strength
# -----------------------------

def run_projection_strength(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    C_gpa: np.ndarray,
    C_orig: np.ndarray,
    params: Dict[str, PreprocessParams],
    R: np.ndarray,
    eigvecs_desc: np.ndarray,
    ranks: List[int],
    device: torch.device,
) -> pd.DataFrame:
    out_csv = Path(args.out_dir) / "projection_strength.csv"
    log("Computing projection-strength metrics.")
    rows = []
    M, K, d = C_gpa.shape
    for mi, p in enumerate(paths):
        C = center_rows(C_gpa[mi]).astype(np.float64)
        denom = float(np.sum(C ** 2))
        C0 = center_rows(C_orig[mi]).astype(np.float64)
        denom_orig = float(np.sum(C0 ** 2))
        W_inv = params[p.safe_name].W_inv.astype(np.float64)
        Rt = R[mi].astype(np.float64).T
        orig_map = Rt @ W_inv
        for r in ranks:
            V = eigvecs_desc[:, :r].astype(np.float64)
            C_proj_coords = C @ V
            E = float(np.sum(C_proj_coords ** 2) / max(denom, 1e-30))
            # With centered centroids and an orthogonal projector, this equals the pairwise-difference energy ratio.
            D = E
            C_common_orig_centered = C @ V @ (V.T @ orig_map)
            E_orig = float(np.sum(C_common_orig_centered ** 2) / max(denom_orig, 1e-30))
            rank_frac = r / d
            rows.append({
                "model": p.safe_name,
                "rank": r,
                "rank_frac": rank_frac,
                "E_class_centroid_common": E,
                "LiftE_class_centroid_common": E / rank_frac,
                "D_class_difference_common": D,
                "LiftD_class_difference_common": D / rank_frac,
                "E_original_space_common": E_orig,
                "LiftE_original_space_common": E_orig / rank_frac,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv}")
    return df


# -----------------------------
# Fisher ratio
# -----------------------------

def fisher_from_stream(
    feature_iter: Callable[[], Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    num_classes: int,
    d: int,
    device: torch.device,
) -> float:
    class_sums = torch.zeros(num_classes, d, dtype=torch.float64, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_sum = torch.zeros(d, dtype=torch.float64, device=device)
    total_sumsq = torch.zeros((), dtype=torch.float64, device=device)
    n_total = 0
    for xb, yb in feature_iter():
        xb64 = xb.to(torch.float64)
        yb = yb.to(torch.long)
        class_sums.index_add_(0, yb, xb64)
        ones = torch.ones_like(yb, dtype=torch.float64, device=device)
        counts.index_add_(0, yb, ones)
        total_sum += xb64.sum(dim=0)
        total_sumsq += torch.sum(xb64 * xb64)
        n_total += xb.shape[0]
        del xb, xb64, yb
    valid = counts > 0
    class_mean_norm_term = torch.sum((class_sums[valid] * class_sums[valid]).sum(dim=1) / counts[valid])
    global_norm_term = torch.sum(total_sum * total_sum) / float(n_total)
    between = class_mean_norm_term - global_norm_term
    within = total_sumsq - class_mean_norm_term
    return float((between / torch.clamp(within, min=1e-30)).detach().cpu().item())


def make_val_feature_iter(
    arr: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    device: torch.device,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Callable[[], Iterable[Tuple[torch.Tensor, torch.Tensor]]]:
    n = arr.shape[0]
    y_all = torch.as_tensor(y_val, dtype=torch.long, device=device)

    def iterator():
        for start, end in batched_ranges(n, batch_size):
            xb = as_torch_batch(arr[start:end], device=device, dtype=torch.float32)
            if transform is not None:
                xb = transform(xb)
            yb = y_all[start:end]
            yield xb, yb
    return iterator


def run_fisher_ratio(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    params: Dict[str, PreprocessParams],
    R: np.ndarray,
    eigvecs_desc: np.ndarray,
    ranks: List[int],
    y_val: np.ndarray,
    num_classes: int,
    d: int,
    device: torch.device,
) -> pd.DataFrame:
    out_csv = Path(args.out_dir) / "fisher_ratio.csv"
    log("Computing Fisher ratios.")
    rows = []
    for mi, p in enumerate(paths):
        arr = load_npy_mmap(p.val_features)
        tp = make_torch_params(params[p.safe_name], R[mi], device)
        raw_iter = make_val_feature_iter(arr, y_val, args.batch_size_transform, device, transform=None)
        fisher_raw = fisher_from_stream(raw_iter, num_classes, d, device)
        for r in ranks:
            rows.append({"model": p.safe_name, "rank": r, "rank_frac": r / d, "component": "raw", "fisher": fisher_raw, "space": "original"})
            for comp, V_np in [("common_top", eigvecs_desc[:, :r]), ("private_bottom", eigvecs_desc[:, -r:])]:
                V = torch.as_tensor(V_np, dtype=torch.float32, device=device)
                transform = make_original_component_transform(V, tp, add_mean=True)
                it = make_val_feature_iter(arr, y_val, args.batch_size_transform, device, transform=transform)
                fr = fisher_from_stream(it, num_classes, d, device)
                rows.append({"model": p.safe_name, "rank": r, "rank_frac": r / d, "component": comp, "fisher": fr, "space": "original"})
                del V
                maybe_cuda_empty_cache(device)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv}")
    return df


# -----------------------------
# HSIC
# -----------------------------

def centered_kernel(K: torch.Tensor) -> torch.Tensor:
    return K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()


def normalized_hsic_from_kernels(K: torch.Tensor, L: torch.Tensor, eps: float = 1e-12) -> float:
    Kc = centered_kernel(K)
    Lc = centered_kernel(L)
    xy = torch.sum(Kc * Lc)
    xx = torch.sum(Kc * Kc)
    yy = torch.sum(Lc * Lc)
    val = xy / torch.sqrt(torch.clamp(xx * yy, min=eps))
    return float(val.detach().cpu().item())


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    return X @ X.T


def rbf_kernel_median(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x2 = torch.sum(X * X, dim=1, keepdim=True)
    D2 = torch.clamp(x2 + x2.T - 2.0 * (X @ X.T), min=0.0)
    n = X.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
    vals = D2[mask]
    vals = vals[vals > eps]
    if vals.numel() == 0:
        sigma2 = torch.tensor(1.0, dtype=X.dtype, device=X.device)
    else:
        sigma2 = torch.median(vals)
    return torch.exp(-D2 / torch.clamp(2.0 * sigma2, min=eps))


def hsic_pair(X: torch.Tensor, Y: torch.Tensor, kernel: str) -> float:
    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        K = rbf_kernel_median(X)
        L = rbf_kernel_median(Y)
    else:
        raise ValueError(kernel)
    out = normalized_hsic_from_kernels(K, L)
    del K, L
    return out


def get_subset_z_features(arr: np.ndarray, indices: np.ndarray, tp: Dict[str, torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        xb = as_torch_batch(arr[idx], device=device, dtype=torch.float32)
        z = raw_to_z(xb, tp)
        chunks.append(z.detach())
        del xb
    return torch.cat(chunks, dim=0)


def run_hsic(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    params: Dict[str, PreprocessParams],
    R: np.ndarray,
    C_gpa: np.ndarray,
    eigvecs_desc: np.ndarray,
    ranks: List[int],
    y_val: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    out_csv = Path(args.out_dir) / "hsic_common_private.csv"
    log("Computing normalized HSIC between common and private components.")
    rng = np.random.default_rng(args.seed)
    n_val = len(y_val)
    n_hsic = min(args.hsic_max_samples, n_val)
    subset_idx = np.sort(rng.choice(n_val, size=n_hsic, replace=False))
    rows = []
    for mi, p in enumerate(paths):
        arr = load_npy_mmap(p.val_features)
        tp = make_torch_params(params[p.safe_name], R[mi], device)
        Zsub = get_subset_z_features(arr, subset_idx, tp, device, args.batch_size_transform)
        C = torch.as_tensor(center_rows(C_gpa[mi]), dtype=torch.float32, device=device)
        for r in tqdm(ranks, desc=f"HSIC {p.safe_name}"):
            Vtop = torch.as_tensor(eigvecs_desc[:, :r], dtype=torch.float32, device=device)
            Vbot = torch.as_tensor(eigvecs_desc[:, -r:], dtype=torch.float32, device=device)
            X_img = Zsub @ Vtop
            Y_img = Zsub @ Vbot
            X_cls = C @ Vtop
            Y_cls = C @ Vbot
            for kernel in ["linear", "rbf"]:
                rows.append({
                    "model": p.safe_name,
                    "rank": r,
                    "rank_frac": r / eigvecs_desc.shape[0],
                    "level": "image",
                    "kernel": kernel,
                    "hsic_norm": hsic_pair(X_img, Y_img, kernel),
                    "n": n_hsic,
                })
                rows.append({
                    "model": p.safe_name,
                    "rank": r,
                    "rank_frac": r / eigvecs_desc.shape[0],
                    "level": "class_centroid",
                    "kernel": kernel,
                    "hsic_norm": hsic_pair(X_cls, Y_cls, kernel),
                    "n": C.shape[0],
                })
            del Vtop, Vbot, X_img, Y_img, X_cls, Y_cls
            maybe_cuda_empty_cache(device)
        del Zsub, C
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv}")
    return df


# -----------------------------
# Linear probes
# -----------------------------

def make_feature_loader_from_memmap(
    arr: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Callable[[int], Iterable[Tuple[torch.Tensor, torch.Tensor]]]:
    n = arr.shape[0]
    labels_np = labels.astype(np.int64, copy=False)

    def loader(epoch: int = 0):
        if shuffle:
            rng = np.random.default_rng(seed + epoch)
            perm = rng.permutation(n)
        else:
            perm = np.arange(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            idx_sorted = np.sort(idx)
            xb = as_torch_batch(arr[idx_sorted], device=device, dtype=torch.float32)
            if transform is not None:
                xb = transform(xb)
            yb = torch.as_tensor(labels_np[idx_sorted], dtype=torch.long, device=device)
            yield xb, yb
    return loader


def train_linear_probe(
    train_loader_fn: Callable[[int], Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    d: int,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    amp: bool,
    desc: str,
) -> torch.nn.Linear:
    head = torch.nn.Linear(d, num_classes).to(device)
    torch.nn.init.normal_(head.weight, std=0.01)
    torch.nn.init.zeros_(head.bias)
    if optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        total = 0
        correct = 0
        iterator = tqdm(train_loader_fn(epoch), desc=f"{desc} epoch {epoch+1}/{epochs}")
        for xb, yb in iterator:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = head(xb)
                loss = F.cross_entropy(logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.detach().cpu().item()) * xb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu().item())
            total += xb.shape[0]
            del xb, yb, logits, loss
        log(f"{desc} epoch {epoch+1}: train_loss={total_loss/max(total,1):.5f}, train_acc={correct/max(total,1):.4f}")
    return head


@torch.no_grad()
def evaluate_linear_probe(head: torch.nn.Linear, loader_fn: Callable[[int], Iterable[Tuple[torch.Tensor, torch.Tensor]]]) -> float:
    head.eval()
    correct = 0
    total = 0
    for xb, yb in loader_fn(0):
        logits = head(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().detach().cpu().item())
        total += xb.shape[0]
        del xb, yb, logits, pred
    return correct / max(total, 1)


def train_or_load_raw_heads(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    y_train: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    d: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.nn.Linear], pd.DataFrame]:
    head_dir = Path(args.out_dir) / "linear_heads_raw"
    ensure_dir(head_dir)
    rows = []
    heads: Dict[str, torch.nn.Linear] = {}
    for p in paths:
        head_path = head_dir / f"{p.safe_name}.pt"
        head = torch.nn.Linear(d, num_classes).to(device)
        if head_path.exists() and not args.force_retrain_heads:
            ckpt = torch.load(head_path, map_location=device)
            head.load_state_dict(ckpt["state_dict"])
            log(f"Loaded raw linear head for {p.safe_name}")
        else:
            log(f"Training raw linear head for {p.safe_name}")
            train_arr = load_npy_mmap(p.train_features)
            loader = make_feature_loader_from_memmap(train_arr, y_train, args.probe_batch_size, device, shuffle=True, seed=args.seed, transform=None)
            head = train_linear_probe(loader, d, num_classes, device, args.probe_epochs, args.probe_lr, args.probe_weight_decay, args.probe_optimizer, args.probe_amp, desc=f"raw-head {p.safe_name}")
            torch.save({"state_dict": head.state_dict(), "model": p.safe_name}, head_path)
            log(f"Saved raw linear head to {head_path}")
        val_arr = load_npy_mmap(p.val_features)
        val_loader = make_feature_loader_from_memmap(val_arr, y_val, args.probe_batch_size, device, shuffle=False, seed=args.seed, transform=None)
        raw_acc = evaluate_linear_probe(head, val_loader)
        log(f"Raw val accuracy {p.safe_name}: {raw_acc:.4f}")
        rows.append({"model": p.safe_name, "component": "raw", "accuracy": raw_acc})
        heads[p.safe_name] = head
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_dir) / "linear_probe_raw_heads.csv", index=False)
    return heads, df


def run_frozen_linear_probe(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    params: Dict[str, PreprocessParams],
    R: np.ndarray,
    eigvecs_desc: np.ndarray,
    ranks: List[int],
    y_train: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    d: int,
    device: torch.device,
) -> pd.DataFrame:
    log("Running frozen-head linear probing: train on raw, test on decomposed components.")
    heads, raw_df = train_or_load_raw_heads(args, paths, y_train, y_val, num_classes, d, device)
    rows = []
    raw_acc_map = {row["model"]: row["accuracy"] for _, row in raw_df.iterrows()}
    for mi, p in enumerate(paths):
        arr_val = load_npy_mmap(p.val_features)
        tp = make_torch_params(params[p.safe_name], R[mi], device)
        head = heads[p.safe_name]
        for r in ranks:
            rows.append({"model": p.safe_name, "rank": r, "rank_frac": r / d, "component": "raw", "accuracy": raw_acc_map[p.safe_name], "protocol": "raw_train_decomp_test"})
            for comp, V_np in [("common_top", eigvecs_desc[:, :r]), ("private_bottom", eigvecs_desc[:, -r:])]:
                V = torch.as_tensor(V_np, dtype=torch.float32, device=device)
                transform = make_original_component_transform(V, tp, add_mean=True)
                val_loader = make_feature_loader_from_memmap(arr_val, y_val, args.probe_batch_size, device, shuffle=False, seed=args.seed, transform=transform)
                acc = evaluate_linear_probe(head, val_loader)
                rows.append({"model": p.safe_name, "rank": r, "rank_frac": r / d, "component": comp, "accuracy": acc, "protocol": "raw_train_decomp_test"})
                log(f"Frozen probe {p.safe_name} r={r} {comp}: acc={acc:.4f}")
                del V
                maybe_cuda_empty_cache(device)
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_dir) / "linear_probe_frozen_head.csv"
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv}")
    return df


def run_decomposed_linear_probe(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    params: Dict[str, PreprocessParams],
    R: np.ndarray,
    eigvecs_desc: np.ndarray,
    ranks: List[int],
    y_train: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    d: int,
    device: torch.device,
) -> pd.DataFrame:
    log("Running decomposed train+test linear probing. This can be very expensive.")
    rows = []
    head_dir = Path(args.out_dir) / "linear_heads_decomposed"
    ensure_dir(head_dir)
    for mi, p in enumerate(paths):
        train_arr = load_npy_mmap(p.train_features)
        val_arr = load_npy_mmap(p.val_features)
        tp = make_torch_params(params[p.safe_name], R[mi], device)
        for r in ranks:
            for comp, V_np in [("common_top", eigvecs_desc[:, :r]), ("private_bottom", eigvecs_desc[:, -r:])]:
                head_path = head_dir / f"{p.safe_name}_{comp}_r{r}.pt"
                V = torch.as_tensor(V_np, dtype=torch.float32, device=device)
                transform = make_original_component_transform(V, tp, add_mean=True)
                if head_path.exists() and not args.force_retrain_heads:
                    head = torch.nn.Linear(d, num_classes).to(device)
                    ckpt = torch.load(head_path, map_location=device)
                    head.load_state_dict(ckpt["state_dict"])
                    log(f"Loaded decomposed head {p.safe_name} {comp} r={r}")
                else:
                    train_loader = make_feature_loader_from_memmap(train_arr, y_train, args.probe_batch_size, device, shuffle=True, seed=args.seed, transform=transform)
                    head = train_linear_probe(train_loader, d, num_classes, device, args.probe_epochs, args.probe_lr, args.probe_weight_decay, args.probe_optimizer, args.probe_amp, desc=f"decomp-head {p.safe_name} {comp} r={r}")
                    torch.save({"state_dict": head.state_dict(), "model": p.safe_name, "component": comp, "rank": r}, head_path)
                val_loader = make_feature_loader_from_memmap(val_arr, y_val, args.probe_batch_size, device, shuffle=False, seed=args.seed, transform=transform)
                acc = evaluate_linear_probe(head, val_loader)
                rows.append({"model": p.safe_name, "rank": r, "rank_frac": r / d, "component": comp, "accuracy": acc, "protocol": "decomp_train_decomp_test"})
                log(f"Decomposed probe {p.safe_name} r={r} {comp}: acc={acc:.4f}")
                del V, head
                maybe_cuda_empty_cache(device)
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_dir) / "linear_probe_decomposed_train_test.csv"
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv}")
    return df


# -----------------------------
# OpenCLIP zero-shot
# -----------------------------

DEFAULT_PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of one {}.",
    "a good photo of a {}.",
    "a photo of a nice {}.",
    "a photo of the {}.",
]


def load_imagenet_class_names(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = []
    for i in range(len(data)):
        item = data[str(i)]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name = item[1]
        else:
            name = str(item)
        names.append(name.replace("_", " "))
    return names


def load_prompt_templates(path: Optional[str]) -> List[str]:
    if path is None or path == "":
        return DEFAULT_PROMPT_TEMPLATES
    with open(path, "r", encoding="utf-8") as f:
        templates = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not templates:
        raise ValueError("Prompt template file is empty.")
    return templates


def format_prompt(template: str, class_name: str) -> str:
    if "{c}" in template:
        return template.format(c=class_name)
    return template.format(class_name)


def extract_visual_projection(openclip_model: torch.nn.Module, visual_input_dim: int, text_dim: int, device: torch.device, visual_proj_path: Optional[str]) -> torch.Tensor:
    if visual_proj_path:
        P = np.load(visual_proj_path)
        P_t = torch.as_tensor(P, dtype=torch.float32, device=device)
    else:
        proj = None
        if hasattr(openclip_model, "visual") and hasattr(openclip_model.visual, "proj"):
            proj = openclip_model.visual.proj
        elif hasattr(openclip_model, "visual") and hasattr(openclip_model.visual, "head") and hasattr(openclip_model.visual.head, "proj"):
            proj = openclip_model.visual.head.proj
        if proj is None:
            if visual_input_dim == text_dim:
                return torch.eye(visual_input_dim, dtype=torch.float32, device=device)
            raise RuntimeError("Could not find open_clip model.visual.proj. Provide --visual-proj-path.")
        if isinstance(proj, torch.nn.Linear):
            P_t = proj.weight.detach().T.to(device=device, dtype=torch.float32)
            if proj.bias is not None:
                log("Warning: visual projection has bias; this script ignores it because CLIP ViT proj is usually bias-free.")
        else:
            P_t = torch.as_tensor(proj.detach(), dtype=torch.float32, device=device)
    if P_t.shape == (visual_input_dim, text_dim):
        return P_t
    if P_t.shape == (text_dim, visual_input_dim):
        return P_t.T
    raise ValueError(f"Unexpected visual projection shape {tuple(P_t.shape)} for visual_input_dim={visual_input_dim}, text_dim={text_dim}")


@torch.no_grad()
def build_openclip_text_features(args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, torch.nn.Module]:
    import open_clip
    class_names = load_imagenet_class_names(Path(args.class_index_json))
    templates = load_prompt_templates(args.prompt_templates_file)
    log(f"Building OpenCLIP text features with {len(templates)} templates.")
    model, _, _ = open_clip.create_model_and_transforms(args.openclip_model, pretrained=args.openclip_pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.openclip_model)
    all_text_features = []
    for template in tqdm(templates, desc="OpenCLIP prompt templates"):
        texts = [format_prompt(template, c) for c in class_names]
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats.float(), dim=-1)
        all_text_features.append(feats)
        del tokens, feats
    text_features = torch.stack(all_text_features, dim=0).mean(dim=0)
    text_features = F.normalize(text_features, dim=-1)
    return text_features, model


@torch.no_grad()
def run_zeroshot(
    args: argparse.Namespace,
    paths: List[ModelPaths],
    params: Dict[str, PreprocessParams],
    R: np.ndarray,
    eigvecs_desc: np.ndarray,
    ranks: List[int],
    y_val: np.ndarray,
    d: int,
    device: torch.device,
) -> pd.DataFrame:
    log("Running CLIP zero-shot evaluation with OpenCLIP text features.")
    clip_name = safe_model_name(args.clip_feature_model)
    clip_indices = [i for i, p in enumerate(paths) if p.safe_name == clip_name or safe_model_name(p.raw_name) == clip_name]
    if len(clip_indices) != 1:
        raise ValueError(f"Could not uniquely identify clip feature model '{clip_name}'. Found indices: {clip_indices}")
    mi = clip_indices[0]
    p = paths[mi]
    text_features, openclip_model = build_openclip_text_features(args, device)
    text_dim = text_features.shape[1]
    visual_proj = extract_visual_projection(openclip_model, d, text_dim, device, args.visual_proj_path)
    logit_scale = float(args.zeroshot_logit_scale)
    if args.use_openclip_logit_scale and hasattr(openclip_model, "logit_scale"):
        logit_scale = float(openclip_model.logit_scale.exp().detach().cpu().item())
    log(f"Zero-shot visual projection={tuple(visual_proj.shape)}, text={tuple(text_features.shape)}, logit_scale={logit_scale:.4f}")

    arr = load_npy_mmap(p.val_features)
    tp = make_torch_params(params[p.safe_name], R[mi], device)
    y_np = y_val.astype(np.int64, copy=False)

    def eval_with_transform(transform: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> float:
        correct = 0
        total = 0
        for start, end in batched_ranges(arr.shape[0], args.probe_batch_size):
            xb = as_torch_batch(arr[start:end], device=device, dtype=torch.float32)
            if transform is not None:
                xb = transform(xb)
            img = xb @ visual_proj
            img = F.normalize(img.float(), dim=-1)
            logits = logit_scale * (img @ text_features.T)
            pred = logits.argmax(dim=1)
            yb = torch.as_tensor(y_np[start:end], dtype=torch.long, device=device)
            correct += int((pred == yb).sum().detach().cpu().item())
            total += xb.shape[0]
            del xb, img, logits, pred, yb
        return correct / max(total, 1)

    rows = []
    raw_acc = eval_with_transform(None)
    log(f"Zero-shot raw {p.safe_name}: acc={raw_acc:.4f}")
    for r in ranks:
        rows.append({"model": p.safe_name, "rank": r, "rank_frac": r / d, "component": "raw", "accuracy": raw_acc})
        for comp, V_np in [("common_top", eigvecs_desc[:, :r]), ("private_bottom", eigvecs_desc[:, -r:])]:
            V = torch.as_tensor(V_np, dtype=torch.float32, device=device)
            transform = make_original_component_transform(V, tp, add_mean=True)
            acc = eval_with_transform(transform)
            rows.append({"model": p.safe_name, "rank": r, "rank_frac": r / d, "component": comp, "accuracy": acc})
            log(f"Zero-shot {p.safe_name} r={r} {comp}: acc={acc:.4f}")
            del V
            maybe_cuda_empty_cache(device)
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_dir) / "clip_zeroshot.csv"
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv}")
    return df


# -----------------------------
# Plots + plot-ready CSV exports
# -----------------------------

def save_plot_table(df: pd.DataFrame, csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    log(f"Saved plot data CSV {csv_path}")


def aggregate_plot_data(df: pd.DataFrame, x: str, y: str, hue: str) -> pd.DataFrame:
    summary = df.groupby([hue, x], dropna=False)[y].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={hue: "series", x: "x", "mean": "y_mean", "std": "y_std", "count": "n"})
    summary["y_sem"] = summary["y_std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    summary["y_lower"] = summary["y_mean"] - summary["y_sem"]
    summary["y_upper"] = summary["y_mean"] + summary["y_sem"]
    summary["x_name"] = x
    summary["y_name"] = y
    summary["series_name"] = hue
    return summary[[ "x_name", "y_name", "series_name", "series", "x", "y_mean", "y_std", "y_sem", "y_lower", "y_upper", "n" ]]


def plot_mean_lines(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    out_path: Path,
    title: str,
    ylabel: str,
    xlabel: str = "rank / d",
    plot_csv_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    summary = aggregate_plot_data(df, x=x, y=y, hue=hue)
    if plot_csv_dir is not None:
        save_plot_table(summary, plot_csv_dir / f"{out_path.stem}.csv")

    plt.figure(figsize=(7, 5))
    for key, sub in summary.groupby("series"):
        sub = sub.sort_values("x")
        xvals = sub["x"].to_numpy(dtype=float)
        yvals = sub["y_mean"].to_numpy(dtype=float)
        lo = sub["y_lower"].to_numpy(dtype=float)
        hi = sub["y_upper"].to_numpy(dtype=float)
        plt.plot(xvals, yvals, marker="o", label=str(key))
        if len(sub) > 1:
            plt.fill_between(xvals, lo, hi, alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return summary


def plot_single_mean_with_csv(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: Path,
    title: str,
    ylabel: str,
    xlabel: str = "rank / d",
    plot_csv_dir: Optional[Path] = None,
    add_y1_line: bool = False,
) -> pd.DataFrame:
    g = df.groupby(x)[y].agg(["mean", "std", "count"]).reset_index().sort_values(x)
    g = g.rename(columns={x: "x", "mean": "y_mean", "std": "y_std", "count": "n"})
    g["y_sem"] = g["y_std"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1))
    g["y_lower"] = g["y_mean"] - g["y_sem"]
    g["y_upper"] = g["y_mean"] + g["y_sem"]
    g["x_name"] = x
    g["y_name"] = y
    g["series_name"] = "aggregate"
    g["series"] = "model_mean"
    g = g[[ "x_name", "y_name", "series_name", "series", "x", "y_mean", "y_std", "y_sem", "y_lower", "y_upper", "n" ]]

    if plot_csv_dir is not None:
        save_plot_table(g, plot_csv_dir / f"{out_path.stem}.csv")

    plt.figure(figsize=(7, 5))
    xvals = g["x"].to_numpy(dtype=float)
    yvals = g["y_mean"].to_numpy(dtype=float)
    lo = g["y_lower"].to_numpy(dtype=float)
    hi = g["y_upper"].to_numpy(dtype=float)
    plt.plot(xvals, yvals, marker="o")
    if len(g) > 1:
        plt.fill_between(xvals, lo, hi, alpha=0.2)
    if add_y1_line:
        plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return g


def plot_projection_strength(df: pd.DataFrame, fig_dir: Path, plot_csv_dir: Path) -> None:
    metrics = [
        ("LiftE_class_centroid_common", "Class-centroid common energy lift"),
        ("LiftD_class_difference_common", "Class-difference common energy lift"),
        ("LiftE_original_space_common", "Original-space common energy lift"),
    ]
    for col, title in metrics:
        # Per-model line plot. CSV contains one row per model/rank and exactly the values used by the lines.
        plot_mean_lines(
            df,
            "rank_frac",
            col,
            "model",
            fig_dir / f"projection_strength_{col}.png",
            title,
            col,
            plot_csv_dir=plot_csv_dir,
        )
        # Model-averaged line plot. CSV contains mean/std/sem/error band.
        plot_single_mean_with_csv(
            df,
            "rank_frac",
            col,
            fig_dir / f"projection_strength_{col}_mean.png",
            title + " (model mean)",
            col,
            plot_csv_dir=plot_csv_dir,
            add_y1_line=True,
        )


def write_plot_manifest(plot_csv_dir: Path, fig_dir: Path) -> None:
    rows = []
    if fig_dir.exists():
        for png in sorted(fig_dir.glob("*.png")):
            csv = plot_csv_dir / f"{png.stem}.csv"
            rows.append({
                "figure": str(png),
                "plot_csv": str(csv),
                "plot_csv_exists": csv.exists(),
            })
    manifest = pd.DataFrame(rows)
    if len(rows) > 0:
        save_plot_table(manifest, plot_csv_dir / "plot_manifest.csv")


def make_all_plots(out_dir: Path) -> None:
    log("Creating plots and plot-ready CSV files from metric CSVs.")
    fig_dir = out_dir / "figures"
    plot_csv_dir = out_dir / "plot_csv"
    ensure_dir(fig_dir)
    ensure_dir(plot_csv_dir)

    candidates = {
        "cross_model_difference_similarity.csv": ("rho", "component", "cross_model_difference_similarity.png", "Class-level cross-model difference similarity", "mean cosine"),
        "fisher_ratio.csv": ("fisher", "component", "fisher_ratio.png", "Fisher ratio", "Fisher ratio"),
        "linear_probe_frozen_head.csv": ("accuracy", "component", "linear_probe_frozen_head.png", "Frozen-head linear probe", "top-1 accuracy"),
        "linear_probe_decomposed_train_test.csv": ("accuracy", "component", "linear_probe_decomposed_train_test.png", "Decomposed train+test linear probe", "top-1 accuracy"),
        "clip_zeroshot.csv": ("accuracy", "component", "clip_zeroshot.png", "CLIP zero-shot", "top-1 accuracy"),
    }
    for fname, (y, hue, out_name, title, ylabel) in candidates.items():
        path = out_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            if "rank_frac" in df.columns:
                plot_mean_lines(df, "rank_frac", y, hue, fig_dir / out_name, title, ylabel, plot_csv_dir=plot_csv_dir)

    ps = out_dir / "projection_strength.csv"
    if ps.exists():
        plot_projection_strength(pd.read_csv(ps), fig_dir, plot_csv_dir)

    hsic = out_dir / "hsic_common_private.csv"
    if hsic.exists():
        df = pd.read_csv(hsic)
        for level in sorted(df["level"].unique()):
            sub = df[df["level"] == level]
            plot_mean_lines(
                sub,
                "rank_frac",
                "hsic_norm",
                "kernel",
                fig_dir / f"hsic_{level}.png",
                f"Normalized HSIC ({level})",
                "normalized HSIC",
                plot_csv_dir=plot_csv_dir,
            )

    write_plot_manifest(plot_csv_dir, fig_dir)
    log(f"Saved figures to {fig_dir}")
    log(f"Saved plot-ready CSV files to {plot_csv_dir}")


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consensus/private decomposition for ImageNet-1k vision features")
    parser.add_argument("--train-root", type=str, default="./features/train")
    parser.add_argument("--val-root", type=str, default="./features/val")
    parser.add_argument("--model-list", type=str, default="./model_list.txt")
    parser.add_argument("--class-index-json", type=str, default="./imagenet_class_index.json")
    parser.add_argument("--out-dir", type=str, default="results_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ranks", type=str, default="8,16,32,64,128,256,384", help="Comma-separated ranks; percentages like 5% are allowed.")
    parser.add_argument("--batch-size-stats", type=int, default=65536)
    parser.add_argument("--batch-size-pairwise", type=int, default=8192)
    parser.add_argument("--batch-size-transform", type=int, default=16384)
    parser.add_argument("--cos-eps", type=float, default=1e-12)

    parser.add_argument("--whiten-eps-abs", type=float, default=1e-6)
    parser.add_argument("--whiten-eps-rel", type=float, default=1e-5)
    parser.add_argument("--gpa-iters", type=int, default=30)
    parser.add_argument("--gpa-tol", type=float, default=1e-7)
    parser.add_argument("--force-recompute", action="store_true", help="Recompute preprocessing/covariance/GPA/centroids even if cached.")

    parser.add_argument("--skip-core-metrics", action="store_true", help="Skip similarity/projection/Fisher/HSIC metrics.")
    parser.add_argument("--skip-fisher", action="store_true")
    parser.add_argument("--skip-hsic", action="store_true")
    parser.add_argument("--hsic-max-samples", type=int, default=3000)

    parser.add_argument("--run-frozen-probe", action="store_true", help="Train raw linear heads and test decomposed components.")
    parser.add_argument("--run-decomp-probe", action="store_true", help="Train separate linear heads on decomposed train features and test decomposed val features.")
    parser.add_argument("--force-retrain-heads", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--probe-batch-size", type=int, default=8192)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--probe-optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--probe-amp", action="store_true")

    parser.add_argument("--run-zeroshot", action="store_true")
    parser.add_argument("--clip-feature-model", type=str, default="vit_base_patch16_clip_224_laion2b", help="Feature directory/model name for CLIP image features.")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--prompt-templates-file", type=str, default=None)
    parser.add_argument("--visual-proj-path", type=str, default=None, help="Optional .npy matrix for timm-CLIP visual projection.")
    parser.add_argument("--zeroshot-logit-scale", type=float, default=100.0)
    parser.add_argument("--use-openclip-logit-scale", action="store_true")

    parser.add_argument("--no-strict-label-check", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Do not create figures or plot-ready CSV files.")
    parser.add_argument("--plots-only", action="store_true", help="Only regenerate figures and plot-ready CSV files from existing metric CSVs.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.plots_only:
        make_all_plots(out_dir)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    log(f"Using device: {device}")
    paths = build_model_paths(args)
    n_train, n_val, d, num_classes, y_train, y_val = validate_shapes_and_labels(paths, strict_labels=not args.no_strict_label_check)
    ranks = parse_ranks(args.ranks, d)
    log(f"Using ranks: {ranks}")

    params = fit_or_load_preprocess(args, paths, d, device)
    C = compute_or_load_whitened_crosscov(args, paths, params, n_train, d, device)
    R = run_or_load_gpa(args, C)
    _, eigvals_desc, eigvecs_desc = compute_or_load_consensus(args, C, R)
    pd.DataFrame({"index_desc": np.arange(len(eigvals_desc)), "eigenvalue": eigvals_desc}).to_csv(out_dir / "consensus_eigenvalues.csv", index=False)

    C_gpa, C_orig, _ = compute_or_load_val_centroids(args, paths, params, R, n_val, d, num_classes, y_val, device)

    if not args.skip_core_metrics:
        run_cross_model_similarity(args, paths, C_gpa, eigvecs_desc, ranks)
        run_projection_strength(args, paths, C_gpa, C_orig, params, R, eigvecs_desc, ranks, device)
        if not args.skip_fisher:
            run_fisher_ratio(args, paths, params, R, eigvecs_desc, ranks, y_val, num_classes, d, device)
        if not args.skip_hsic:
            run_hsic(args, paths, params, R, C_gpa, eigvecs_desc, ranks, y_val, device)

    if args.run_frozen_probe:
        run_frozen_linear_probe(args, paths, params, R, eigvecs_desc, ranks, y_train, y_val, num_classes, d, device)

    if args.run_decomp_probe:
        run_decomposed_linear_probe(args, paths, params, R, eigvecs_desc, ranks, y_train, y_val, num_classes, d, device)

    if args.run_zeroshot:
        run_zeroshot(args, paths, params, R, eigvecs_desc, ranks, y_val, d, device)

    if not args.no_plots:
        make_all_plots(out_dir)

    log("Done.")


if __name__ == "__main__":
    main()
