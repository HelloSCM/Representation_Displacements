import argparse
from pathlib import Path
import torch

from .config import RunConfig, validate_config
from .model_selector import resolve_models
from .core.metrics import process_metrics
from .core.whitening import compute_pca_projection, apply_pca
from .core.wordnet_utils import get_vision_parent, get_language_parent
from .io.readers import read_vision_features_imagenet22k, read_language_features_commonwords79k
from .io.writers import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orthogonal metrics for anonymous release")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run metric computation")
    run.add_argument("--modality", choices=["vision", "language"], required=True)
    run.add_argument("--input-root", type=Path, required=True, help="Input feature directory")
    run.add_argument("--model-list", type=Path, default=None, help="Path to model_list.txt")
    run.add_argument("--model", type=str, default=None, help="Run a single model only")
    run.add_argument("--class-feat-root", type=Path, default=None, help="ImageNet22k class feature directory (vision only)")
    run.add_argument("--output-root", type=Path, default=Path("./outputs"), help="Relative output directory")
    run.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--pca-components", type=int, default=-1, help="-1 disables PCA; positive integer enables PCA")
    return parser


def run_command(args):
    cfg = RunConfig(
        modality=args.modality,
        input_root=args.input_root,
        output_root=args.output_root,
        model_list=args.model_list,
        model=args.model,
        class_feat_root=args.class_feat_root,
        device=args.device,
        pca_components=args.pca_components,
    )
    validate_config(cfg)

    device = torch.device(cfg.device)
    models = resolve_models(cfg.model, cfg.model_list)

    for model in models:
        if cfg.modality == "vision":
            class_features = read_vision_features_imagenet22k(cfg.input_root, cfg.class_feat_root, model, device)
            parent_fn = get_vision_parent
            dataset_tag = "imagenet22k"
            dataset_name = "ImageNet22k"
        else:
            class_features = read_language_features_commonwords79k(cfg.input_root, model, device)
            parent_fn = get_language_parent
            dataset_tag = "commonwords79k"
            dataset_name = "CommonWords79k"

        if cfg.pca_components > 0:
            x = torch.stack(list(class_features.values()))
            mu, w = compute_pca_projection(x, cfg.pca_components)
            class_features = apply_pca(class_features, mu, w)

        metrics, raw = process_metrics(class_features, parent_fn)
        if metrics is None:
            print(f"[{model}] skipped: no eligible classes/words")
            continue

        summary = {
            "model": model,
            "modality": cfg.modality,
            "dataset": dataset_name,
            "dataset_tag": dataset_tag,
            "pca_components": cfg.pca_components,
            "metrics": metrics,
        }
        write_outputs(cfg.output_root, model, metrics, raw, summary)
        print(f"[{model}] done -> {cfg.output_root}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_command(args)
