import argparse
from pathlib import Path
import numpy as np
import torch

from .config import RunConfig, validate_config
from .model_selector import resolve_models
from .core.alignment import set_seed, build_concept_intersection, split_train_val
from .core.pca_opa import pca_whiten_fit_transform, opa_matrix, metrics_with_fixed_rotation
from .io.readers import load_layered_pt, load_layered_npy, VISION_PATTERN, LANG_PATTERN
from .io.writers import write_pair


def parser_build():
    p = argparse.ArgumentParser(description="Parallel metrics (anonymous release)")
    p.add_argument("run", nargs="?")
    p.add_argument("--mode", choices=["concept", "sample", "cross"], required=True)
    p.add_argument("--vision-dir", type=Path, default=None)
    p.add_argument("--language-dir", type=Path, default=None)
    p.add_argument("--sample-dir", type=Path, default=None)

    p.add_argument("--vision-model-list", type=Path, default=None)
    p.add_argument("--language-model-list", type=Path, default=None)
    p.add_argument("--sample-model-list", type=Path, default=None)

    p.add_argument("--vision-model", type=str, default=None)
    p.add_argument("--language-model", type=str, default=None)
    p.add_argument("--sample-model", type=str, default=None)

    p.add_argument("--output-root", type=Path, default=Path("./outputs_parallel"))
    p.add_argument("--proj-dim", type=int, default=100)
    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def _to_np_from_dict(d: dict, keys: list[str]) -> np.ndarray:
    return torch.stack([d[k] for k in keys]).numpy()


def _concept_pair_rows(v_layers, l_layers, v_store, l_store, keys_v, keys_l, cfg, device):
    rows = []
    total = len(keys_v)
    train_idx, val_idx = split_train_val(total, cfg.train_size, cfg.seed)

    for vl in v_layers:
        for ll in l_layers:
            v_np = _to_np_from_dict(v_store[vl], keys_v)
            l_np = _to_np_from_dict(l_store[ll], keys_l)

            v_proj, v_mu = pca_whiten_fit_transform(v_np, cfg.proj_dim, cfg.seed)
            l_proj, l_mu = pca_whiten_fit_transform(l_np, cfg.proj_dim, cfg.seed)

            v_t = torch.from_numpy(v_proj).float().to(device)
            l_t = torch.from_numpy(l_proj).float().to(device)
            v_mu_t = torch.from_numpy(v_mu).float().to(device).view(1, -1)
            l_mu_t = torch.from_numpy(l_mu).float().to(device).view(1, -1)

            r = opa_matrix(v_t[train_idx], l_t[train_idx])
            m = metrics_with_fixed_rotation(v_t[val_idx], v_mu_t, l_t[val_idx], l_mu_t, r)
            rows.append({"vision_layer": vl, "language_layer": ll, **m})
    return rows, len(train_idx), len(val_idx)


def _sample_pair_rows(a_layers, b_layers, a_store, b_store, cfg, device):
    rows = []
    any_layer = a_layers[0]
    total = a_store[any_layer].shape[0]
    train_idx, val_idx = split_train_val(total, cfg.train_size, cfg.seed)

    for al in a_layers:
        for bl in b_layers:
            a_proj, a_mu = pca_whiten_fit_transform(a_store[al], cfg.proj_dim, cfg.seed)
            b_proj, b_mu = pca_whiten_fit_transform(b_store[bl], cfg.proj_dim, cfg.seed)

            a_t = torch.from_numpy(a_proj).float().to(device)
            b_t = torch.from_numpy(b_proj).float().to(device)
            a_mu_t = torch.from_numpy(a_mu).float().to(device).view(1, -1)
            b_mu_t = torch.from_numpy(b_mu).float().to(device).view(1, -1)

            r = opa_matrix(a_t[train_idx], b_t[train_idx])
            m = metrics_with_fixed_rotation(a_t[val_idx], a_mu_t, b_t[val_idx], b_mu_t, r)
            rows.append({"layer_a": al, "layer_b": bl, **m})
    return rows, len(train_idx), len(val_idx)


def main():
    args = parser_build().parse_args()
    cfg = RunConfig(**{k.replace('-', '_'): v for k, v in vars(args).items() if k != 'run'})
    validate_config(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    if cfg.mode in {"concept", "cross"}:
        v_models = resolve_models(cfg.vision_model, cfg.vision_model_list)
        l_models = resolve_models(cfg.language_model, cfg.language_model_list)
        for vm in v_models:
            v_layers = load_layered_pt(cfg.vision_dir, VISION_PATTERN, vm)
            sample_v = v_layers[sorted(v_layers.keys())[0]]
            for lm in l_models:
                l_layers = load_layered_pt(cfg.language_dir, LANG_PATTERN, lm)
                sample_l = l_layers[sorted(l_layers.keys())[0]]
                keys_v, keys_l = build_concept_intersection(sample_v, sample_l)
                rows, tr_n, val_n = _concept_pair_rows(
                    sorted(v_layers.keys()), sorted(l_layers.keys()), v_layers, l_layers, keys_v, keys_l, cfg, device
                )
                pair = f"{lm}-VS-{vm}"
                summary = {
                    "mode": cfg.mode,
                    "vision_dataset": "ImageNet22k",
                    "language_dataset": "CommonWords79k",
                    "vision_model": vm,
                    "language_model": lm,
                    "train_size": tr_n,
                    "val_size": val_n,
                    "proj_dim": cfg.proj_dim,
                }
                write_pair(cfg.output_root / cfg.mode, pair, rows, summary)

    if cfg.mode == "sample":
        s_models = resolve_models(cfg.sample_model, cfg.sample_model_list)
        for i in range(len(s_models)):
            for j in range(i, len(s_models)):
                m1, m2 = s_models[i], s_models[j]
                a_layers = load_layered_npy(cfg.sample_dir, m1)
                b_layers = load_layered_npy(cfg.sample_dir, m2)
                rows, tr_n, val_n = _sample_pair_rows(
                    sorted(a_layers.keys()), sorted(b_layers.keys()), a_layers, b_layers, cfg, device
                )
                pair = f"{m1}-VS-{m2}"
                summary = {
                    "mode": "sample",
                    "dataset": "Flickr30k",
                    "model_a": m1,
                    "model_b": m2,
                    "train_size": tr_n,
                    "val_size": val_n,
                    "proj_dim": cfg.proj_dim,
                }
                write_pair(cfg.output_root / "sample", pair, rows, summary)


if __name__ == "__main__":
    main()
