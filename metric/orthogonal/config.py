from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunConfig:
    modality: str
    input_root: Path
    output_root: Path
    model_list: Optional[Path]
    model: Optional[str]
    class_feat_root: Optional[Path]
    device: str
    pca_components: int


def validate_config(cfg: RunConfig) -> None:
    if cfg.modality not in {"vision", "language"}:
        raise ValueError("modality must be one of: vision, language")

    if not cfg.input_root.exists():
        raise ValueError(f"input root does not exist: {cfg.input_root}")

    if cfg.model is None and cfg.model_list is None:
        raise ValueError("--model-list is required when --model is not provided")

    if cfg.model_list is not None and not cfg.model_list.exists():
        raise ValueError(f"model list does not exist: {cfg.model_list}")

    if cfg.modality == "vision" and cfg.class_feat_root is None:
        raise ValueError("--class-feat-root is required for vision mode (ImageNet22k)")

    if cfg.class_feat_root is not None and not cfg.class_feat_root.exists():
        raise ValueError(f"class feature root does not exist: {cfg.class_feat_root}")

    if cfg.pca_components != -1 and cfg.pca_components <= 0:
        raise ValueError("pca_components must be -1 (disabled) or a positive integer")
