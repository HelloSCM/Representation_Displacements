from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunConfig:
    mode: str
    vision_dir: Optional[Path]
    language_dir: Optional[Path]
    sample_dir: Optional[Path]
    vision_model_list: Optional[Path]
    language_model_list: Optional[Path]
    sample_model_list: Optional[Path]
    vision_model: Optional[str]
    language_model: Optional[str]
    sample_model: Optional[str]
    output_root: Path
    proj_dim: int
    train_size: int
    seed: int
    device: str


def validate_config(cfg: RunConfig) -> None:
    if cfg.mode not in {"concept", "sample", "cross"}:
        raise ValueError("mode must be one of: concept, sample, cross")
    if cfg.proj_dim <= 0:
        raise ValueError("proj_dim must be > 0")
    if cfg.train_size <= 0:
        raise ValueError("train_size must be > 0")

    if cfg.mode in {"concept", "cross"}:
        if cfg.vision_dir is None or cfg.language_dir is None:
            raise ValueError("concept/cross mode requires --vision-dir and --language-dir")
        if not cfg.vision_dir.exists() or not cfg.language_dir.exists():
            raise ValueError("vision/language input directory does not exist")

    if cfg.mode == "sample":
        if cfg.sample_dir is None:
            raise ValueError("sample mode requires --sample-dir")
        if not cfg.sample_dir.exists():
            raise ValueError("sample input directory does not exist")

    for p in [cfg.vision_model_list, cfg.language_model_list, cfg.sample_model_list]:
        if p is not None and not p.exists():
            raise ValueError(f"model list does not exist: {p}")
