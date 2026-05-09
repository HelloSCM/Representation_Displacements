#!/usr/bin/env python3
"""
Evaluation utilities for VisionConnector + LanguageGuider.

This file supports two use cases:

1) Import from train.py:

    import eval as eval_mod

    eval_mod.add_eval_args(parser)
    eval_mod.evaluate_all(model, args, device=dist_info.device,
                          epoch=-1, global_step=global_step, tag="before_train")

2) Standalone evaluation from a checkpoint:

    python eval.py \
      --checkpoint ./checkpoints/final.pt \
      --arch base \
      --teacher-name llm2vec_qwen3_4b

Feature assumptions:
    - safetensors key: "features"
    - dtype: float32
    - normalization_applied: false
    - CLIP vision features are pre-visual.proj backbone features
    - CLIP language features are post-text_projection encode_text features
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from ..models.vision_connector import VisionConnector
from ..models.language_guider import LanguageGuider


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add evaluation args to an existing train.py parser."""
    group = parser.add_argument_group("Evaluation")

    group.add_argument("--eval-enabled", dest="eval_enabled", action="store_true")
    group.add_argument("--no-eval-enabled", dest="eval_enabled", action="store_false")
    parser.set_defaults(eval_enabled=True)

    group.add_argument("--eval-before-train", dest="eval_before_train", action="store_true")
    group.add_argument("--no-eval-before-train", dest="eval_before_train", action="store_false")
    parser.set_defaults(eval_before_train=True)

    group.add_argument("--eval-every-epoch", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    group.add_argument(
        "--eval-datasets",
        type=str,
        default="imagenet,coco,winoground,mmvp",
        help="Comma-separated subset from: imagenet,coco,winoground,mmvp",
    )

    group.add_argument("--imagenet-val-root", type=str, default="./imagenet_val_feat")
    group.add_argument("--coco-val-root", type=str, default="./coco_val_feat")
    group.add_argument("--winoground-root", type=str, default="./winoground_feat")
    group.add_argument("--mmvp-root", type=str, default="./mmvp_feat")

    group.add_argument("--eval-batch-size", type=int, default=4096)
    group.add_argument("--eval-sim-chunk-size", type=int, default=1024)
    group.add_argument("--eval-device", type=str, default="", help="Default: same as train device")
    group.add_argument("--eval-skip-missing", action="store_true")

    group.add_argument("--eval-output-jsonl", type=str, default="")
    group.add_argument("--eval-write-json", dest="eval_write_json", action="store_true")
    group.add_argument("--no-eval-write-json", dest="eval_write_json", action="store_false")
    parser.set_defaults(eval_write_json=True)

    return parser


# -----------------------------------------------------------------------------
# Distributed / generic helpers
# -----------------------------------------------------------------------------


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _barrier() -> None:
    if _dist_ready():
        dist.barrier()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _parse_dataset_list(s: str) -> List[str]:
    out = [x.strip().lower() for x in s.split(",") if x.strip()]
    valid = {"imagenet", "coco", "winoground", "mmvp"}
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(f"Unknown eval dataset(s): {bad}. Valid: {sorted(valid)}")
    return out


def _getattr_default(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _resolve_device(args: argparse.Namespace, device: Optional[torch.device]) -> torch.device:
    eval_device = _getattr_default(args, "eval_device", "")
    if eval_device:
        return torch.device(eval_device)
    if device is not None:
        return device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_feature(path: Path, *, check_float32: bool = True) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(str(path))
    obj = load_file(str(path), device="cpu")
    if "features" not in obj:
        raise KeyError(f"{path} does not contain tensor named 'features'.")
    x = obj["features"]
    if check_float32 and x.dtype != torch.float32:
        logging.warning("%s dtype is %s, expected float32; casting in eval.", path, x.dtype)
    return x


def maybe_check_meta(feature_path: Path, meta_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    if meta_path is None:
        meta_path = feature_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    meta = load_json(meta_path)
    if meta.get("normalization_applied") not in (False, None):
        logging.warning("%s says normalization_applied=%s", meta_path, meta.get("normalization_applied"))
    return meta


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def eval_targets(arch: str) -> Dict[str, str]:
    if arch == "base":
        return {
            "clip_vision": "clip_base_vision",
            "dino_cls": "dino_base_cls",
            "dino_gap": "dino_base_gap",
            "clip_language": "clip_base_language",
        }
    if arch == "large":
        return {
            "clip_vision": "clip_large_vision",
            "dino_cls": "dino_large_cls",
            "dino_gap": "dino_large_gap",
            "clip_language": "clip_large_language",
        }
    raise ValueError(f"Unsupported arch={arch!r}")


def _iter_chunks(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


@torch.no_grad()
def encode_vision_batches(
    vision: nn.Module,
    clip_feat: torch.Tensor,
    dino_cls: torch.Tensor,
    dino_gap: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not (clip_feat.shape[0] == dino_cls.shape[0] == dino_gap.shape[0]):
        raise ValueError(
            f"Vision feature row mismatch: clip={clip_feat.shape}, "
            f"dino_cls={dino_cls.shape}, dino_gap={dino_gap.shape}"
        )
    outs: List[torch.Tensor] = []
    n = clip_feat.shape[0]
    for start, end in _iter_chunks(n, batch_size):
        z = vision(
            clip_feat=clip_feat[start:end].to(device=device, dtype=torch.float32, non_blocking=True),
            dino_cls=dino_cls[start:end].to(device=device, dtype=torch.float32, non_blocking=True),
            dino_gap=dino_gap[start:end].to(device=device, dtype=torch.float32, non_blocking=True),
        )
        outs.append(z.detach().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def encode_text_batches(
    language: nn.Module,
    clip_text_feat: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    n = clip_text_feat.shape[0]
    for start, end in _iter_chunks(n, batch_size):
        x = clip_text_feat[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        if hasattr(language, "encode_text"):
            z = language.encode_text(x)
        else:
            out = language(x, compute_relation_loss=False, return_dict=True)
            z = out["text_z"]
        outs.append(z.detach().cpu())
    return torch.cat(outs, dim=0)


# -----------------------------------------------------------------------------
# ImageNet-1K zero-shot
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_imagenet(
    model: nn.Module,
    *,
    root: str,
    arch: str,
    device: torch.device,
    batch_size: int = 4096,
    skip_missing: bool = False,
) -> Dict[str, float]:
    root_path = Path(root)
    targets = eval_targets(arch)
    core = _unwrap_model(model)
    vision = core.vision
    language = core.language

    classes_path = root_path / "manifest" / "classes.json"
    text_path = root_path / targets["clip_language"] / "imagenet1k_templates.safetensors"
    text_meta_path = root_path / targets["clip_language"] / "imagenet1k_templates.meta.json"

    if not classes_path.exists() or not text_path.exists() or not text_meta_path.exists():
        if skip_missing:
            logging.warning("Skipping ImageNet eval because required files are missing under %s", root_path)
            return {}
        raise FileNotFoundError(f"Missing ImageNet eval files under {root_path}")

    classes_meta = load_json(classes_path)
    text_meta = load_json(text_meta_path)
    text_raw = load_feature(text_path)

    num_classes = int(text_meta.get("num_classes", classes_meta.get("num_classes", 1000)))
    num_templates = int(text_meta.get("num_templates", classes_meta.get("num_templates", 80)))

    text_raw = text_raw.view(num_classes * num_templates, -1)
    text_z = encode_text_batches(language, text_raw, batch_size=batch_size, device=device)
    text_z = text_z.view(num_classes, num_templates, -1)

    # Template ensemble: apply connector to each template, then average normalized templates.
    text_classifier = F.normalize(text_z, dim=-1).mean(dim=1)
    text_classifier = F.normalize(text_classifier, dim=-1).to(device=device, dtype=torch.float32)

    class_records = text_meta.get("class_records") or classes_meta.get("classes")
    if not class_records:
        raise ValueError("Could not find class_records/classes for ImageNet evaluation.")
    class_pos_by_idx = {int(r["class_idx"]): i for i, r in enumerate(class_records)}

    total = 0
    correct1 = 0
    correct5 = 0

    classes = classes_meta.get("classes", class_records)
    for record in classes:
        class_idx = int(record["class_idx"])
        wnid = str(record["wnid"])
        shard_id = f"{class_idx:04d}_{wnid}"
        label_pos = class_pos_by_idx[class_idx]

        paths = {
            "clip": root_path / targets["clip_vision"] / f"{shard_id}.safetensors",
            "dino_cls": root_path / targets["dino_cls"] / f"{shard_id}.safetensors",
            "dino_gap": root_path / targets["dino_gap"] / f"{shard_id}.safetensors",
        }
        if any(not p.exists() for p in paths.values()):
            if skip_missing:
                logging.warning("Skipping missing ImageNet shard %s", shard_id)
                continue
            missing = [str(p) for p in paths.values() if not p.exists()]
            raise FileNotFoundError(f"Missing ImageNet shard files: {missing}")

        clip = load_feature(paths["clip"])
        dino_cls = load_feature(paths["dino_cls"])
        dino_gap = load_feature(paths["dino_gap"])

        img_z = encode_vision_batches(
            vision,
            clip,
            dino_cls,
            dino_gap,
            batch_size=batch_size,
            device=device,
        ).to(device=device, dtype=torch.float32)

        logits = img_z @ text_classifier.T
        k = min(5, logits.shape[1])
        topk = logits.topk(k=k, dim=-1).indices
        labels = torch.full((logits.shape[0],), label_pos, device=device, dtype=torch.long)

        correct1 += int((topk[:, 0] == labels).sum().item())
        correct5 += int((topk == labels[:, None]).any(dim=1).sum().item())
        total += int(logits.shape[0])

    top1 = 100.0 * correct1 / max(1, total)
    top5 = 100.0 * correct5 / max(1, total)

    return {
        "imagenet/top1": top1,
        "imagenet/top5": top5,
        "imagenet/num_images": float(total),
    }


# -----------------------------------------------------------------------------
# COCO retrieval
# -----------------------------------------------------------------------------


@torch.no_grad()
def _recall_i2t(
    image_z: torch.Tensor,
    text_z: torch.Tensor,
    image_ids: Sequence[Any],
    image_id_to_caption_indices: Dict[str, Sequence[int]],
    *,
    device: torch.device,
    chunk_size: int,
) -> Dict[str, float]:
    text_z_dev = text_z.to(device=device, dtype=torch.float32)
    total = image_z.shape[0]
    r1 = 0
    r5 = 0

    for start, end in _iter_chunks(total, chunk_size):
        img = image_z[start:end].to(device=device, dtype=torch.float32)
        sim = img @ text_z_dev.T
        top5 = sim.topk(k=min(5, sim.shape[1]), dim=-1).indices.cpu().tolist()
        for local_i, preds in enumerate(top5):
            global_i = start + local_i
            image_id = str(image_ids[global_i])
            pos = set(int(x) for x in image_id_to_caption_indices[image_id])
            r1 += int(preds[0] in pos)
            r5 += int(any(p in pos for p in preds[:5]))

    return {
        "coco/i2t_r1": 100.0 * r1 / max(1, total),
        "coco/i2t_r5": 100.0 * r5 / max(1, total),
    }


@torch.no_grad()
def _recall_t2i(
    text_z: torch.Tensor,
    image_z: torch.Tensor,
    caption_to_image_feature_indices: Sequence[int],
    *,
    device: torch.device,
    chunk_size: int,
) -> Dict[str, float]:
    image_z_dev = image_z.to(device=device, dtype=torch.float32)
    total = text_z.shape[0]
    r1 = 0
    r5 = 0

    for start, end in _iter_chunks(total, chunk_size):
        txt = text_z[start:end].to(device=device, dtype=torch.float32)
        sim = txt @ image_z_dev.T
        top5 = sim.topk(k=min(5, sim.shape[1]), dim=-1).indices.cpu().tolist()
        for local_i, preds in enumerate(top5):
            global_i = start + local_i
            pos = int(caption_to_image_feature_indices[global_i])
            r1 += int(preds[0] == pos)
            r5 += int(pos in preds[:5])

    return {
        "coco/t2i_r1": 100.0 * r1 / max(1, total),
        "coco/t2i_r5": 100.0 * r5 / max(1, total),
    }


@torch.no_grad()
def evaluate_coco(
    model: nn.Module,
    *,
    root: str,
    arch: str,
    device: torch.device,
    batch_size: int = 4096,
    sim_chunk_size: int = 1024,
    skip_missing: bool = False,
) -> Dict[str, float]:
    root_path = Path(root)
    targets = eval_targets(arch)
    core = _unwrap_model(model)
    vision = core.vision
    language = core.language

    paths = {
        "clip": root_path / targets["clip_vision"] / "images.safetensors",
        "dino_cls": root_path / targets["dino_cls"] / "images.safetensors",
        "dino_gap": root_path / targets["dino_gap"] / "images.safetensors",
        "captions": root_path / targets["clip_language"] / "captions.safetensors",
        "images_meta": root_path / targets["clip_vision"] / "images.meta.json",
        "captions_meta": root_path / targets["clip_language"] / "captions.meta.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        if skip_missing:
            logging.warning("Skipping COCO eval because files are missing: %s", missing)
            return {}
        raise FileNotFoundError(f"Missing COCO eval files: {missing}")

    clip = load_feature(paths["clip"])
    dino_cls = load_feature(paths["dino_cls"])
    dino_gap = load_feature(paths["dino_gap"])
    captions = load_feature(paths["captions"])
    images_meta = load_json(paths["images_meta"])
    captions_meta = load_json(paths["captions_meta"])

    image_z = encode_vision_batches(vision, clip, dino_cls, dino_gap, batch_size=batch_size, device=device)
    text_z = encode_text_batches(language, captions, batch_size=batch_size, device=device)

    image_z = F.normalize(image_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)

    out: Dict[str, float] = {}
    out.update(
        _recall_i2t(
            image_z,
            text_z,
            images_meta["image_ids"],
            captions_meta["image_id_to_caption_indices"],
            device=device,
            chunk_size=sim_chunk_size,
        )
    )
    out.update(
        _recall_t2i(
            text_z,
            image_z,
            captions_meta["caption_to_image_feature_indices"],
            device=device,
            chunk_size=sim_chunk_size,
        )
    )
    out["coco/num_images"] = float(image_z.shape[0])
    out["coco/num_captions"] = float(text_z.shape[0])
    return out


# -----------------------------------------------------------------------------
# Winoground / MMVP pairwise VQA-style evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def _encode_pair_features(
    model: nn.Module,
    *,
    root_path: Path,
    arch: str,
    filename: str,
    batch_size: int,
    device: torch.device,
    skip_missing: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    targets = eval_targets(arch)
    core = _unwrap_model(model)
    vision = core.vision
    language = core.language

    paths = {
        "clip": root_path / targets["clip_vision"] / filename,
        "dino_cls": root_path / targets["dino_cls"] / filename,
        "dino_gap": root_path / targets["dino_gap"] / filename,
        "text": root_path / targets["clip_language"] / filename,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        if skip_missing:
            logging.warning("Skipping pair eval because files are missing: %s", missing)
            return None
        raise FileNotFoundError(f"Missing pair eval files: {missing}")

    clip = load_feature(paths["clip"])
    dino_cls = load_feature(paths["dino_cls"])
    dino_gap = load_feature(paths["dino_gap"])
    text = load_feature(paths["text"])

    if clip.ndim != 3 or text.ndim != 3 or clip.shape[1] != 2 or text.shape[1] != 2:
        raise ValueError(f"Expected pair features [N,2,D], got clip={clip.shape}, text={text.shape}")

    n = clip.shape[0]
    clip_flat = clip.reshape(n * 2, -1)
    dino_cls_flat = dino_cls.reshape(n * 2, -1)
    dino_gap_flat = dino_gap.reshape(n * 2, -1)
    text_flat = text.reshape(n * 2, -1)

    img_z = encode_vision_batches(
        vision,
        clip_flat,
        dino_cls_flat,
        dino_gap_flat,
        batch_size=batch_size,
        device=device,
    ).view(n, 2, -1)
    txt_z = encode_text_batches(language, text_flat, batch_size=batch_size, device=device).view(n, 2, -1)

    img_z = F.normalize(img_z, dim=-1)
    txt_z = F.normalize(txt_z, dim=-1)
    return img_z, txt_z


@torch.no_grad()
def evaluate_winoground(
    model: nn.Module,
    *,
    root: str,
    arch: str,
    device: torch.device,
    batch_size: int = 4096,
    skip_missing: bool = False,
) -> Dict[str, float]:
    root_path = Path(root)
    encoded = _encode_pair_features(
        model,
        root_path=root_path,
        arch=arch,
        filename="winoground.safetensors",
        batch_size=batch_size,
        device=device,
        skip_missing=skip_missing,
    )
    if encoded is None:
        return {}

    img_z, txt_z = encoded
    n = img_z.shape[0]

    text_correct = 0
    image_correct = 0
    group_correct = 0

    for i in range(n):
        s = txt_z[i] @ img_z[i].T
        text_score = bool((s[0, 0] > s[1, 0]) and (s[1, 1] > s[0, 1]))
        image_score = bool((s[0, 0] > s[0, 1]) and (s[1, 1] > s[1, 0]))
        group_score = text_score and image_score
        text_correct += int(text_score)
        image_correct += int(image_score)
        group_correct += int(group_score)

    return {
        "winoground/text": 100.0 * text_correct / max(1, n),
        "winoground/image": 100.0 * image_correct / max(1, n),
        "winoground/group": 100.0 * group_correct / max(1, n),
        "winoground/num_examples": float(n),
    }


@torch.no_grad()
def evaluate_mmvp(
    model: nn.Module,
    *,
    root: str,
    arch: str,
    device: torch.device,
    batch_size: int = 4096,
    skip_missing: bool = False,
) -> Dict[str, float]:
    root_path = Path(root)
    encoded = _encode_pair_features(
        model,
        root_path=root_path,
        arch=arch,
        filename="mmvp.safetensors",
        batch_size=batch_size,
        device=device,
        skip_missing=skip_missing,
    )
    if encoded is None:
        return {}

    img_z, txt_z = encoded
    n = img_z.shape[0]

    meta_path = root_path / eval_targets(arch)["clip_language"] / "mmvp.meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        pairs = meta.get("pairs", [])
    else:
        pairs = []

    question_correct = 0
    pair_correct = 0
    by_category: Dict[str, List[int]] = {}

    for i in range(n):
        s = txt_z[i] @ img_z[i].T
        pred1 = "img1" if s[0, 0] > s[0, 1] else "img2"
        pred2 = "img1" if s[1, 0] > s[1, 1] else "img2"

        if i < len(pairs):
            pair = pairs[i]
            gt1 = str(pair.get("gt1", "img1"))
            gt2 = str(pair.get("gt2", "img2"))
            category = str(pair.get("category", "unknown"))
        else:
            # MMVP default pairing convention from the feature extraction spec.
            gt1, gt2 = "img1", "img2"
            category = "unknown"

        c1 = pred1 == gt1
        c2 = pred2 == gt2
        both = c1 and c2
        question_correct += int(c1) + int(c2)
        pair_correct += int(both)
        by_category.setdefault(category, []).append(int(both))

    out: Dict[str, float] = {
        "mmvp/question_acc": 100.0 * question_correct / max(1, 2 * n),
        "mmvp/pair_acc": 100.0 * pair_correct / max(1, n),
        "mmvp/num_pairs": float(n),
    }

    # Category-level pair accuracy. Keep names JSON-friendly.
    for category, vals in sorted(by_category.items()):
        key = "mmvp/category_pair_acc/" + category.replace(" ", "_").replace("/", "_")
        out[key] = 100.0 * sum(vals) / max(1, len(vals))

    return out


# -----------------------------------------------------------------------------
# Top-level evaluation entry point for train.py
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_all(
    model: nn.Module,
    args: argparse.Namespace,
    *,
    device: Optional[torch.device] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    tag: str = "eval",
) -> Dict[str, float]:
    """
    Run requested evaluations.

    Intended DDP behavior:
        - all ranks should call this function
        - only rank0 actually evaluates
        - all other ranks wait on barriers
    """
    if not _getattr_default(args, "eval_enabled", True):
        return {}

    _barrier()

    results: Dict[str, float] = {}
    if _rank() == 0:
        eval_device = _resolve_device(args, device)
        core = _unwrap_model(model)
        was_training = core.training
        core.eval()
        core.to(eval_device)

        arch = _getattr_default(args, "arch", "base")
        batch_size = int(_getattr_default(args, "eval_batch_size", 4096))
        sim_chunk_size = int(_getattr_default(args, "eval_sim_chunk_size", 1024))
        skip_missing = bool(_getattr_default(args, "eval_skip_missing", False))
        datasets = _parse_dataset_list(_getattr_default(args, "eval_datasets", "imagenet,coco,winoground,mmvp"))

        logging.info("Starting evaluation tag=%s epoch=%s global_step=%s datasets=%s", tag, epoch, global_step, datasets)
        start_time = time.time()

        if "imagenet" in datasets:
            results.update(
                evaluate_imagenet(
                    core,
                    root=_getattr_default(args, "imagenet_val_root", "./imagenet_val_feat"),
                    arch=arch,
                    device=eval_device,
                    batch_size=batch_size,
                    skip_missing=skip_missing,
                )
            )

        if "coco" in datasets:
            results.update(
                evaluate_coco(
                    core,
                    root=_getattr_default(args, "coco_val_root", "./coco_val2017_feat"),
                    arch=arch,
                    device=eval_device,
                    batch_size=batch_size,
                    sim_chunk_size=sim_chunk_size,
                    skip_missing=skip_missing,
                )
            )

        if "winoground" in datasets:
            results.update(
                evaluate_winoground(
                    core,
                    root=_getattr_default(args, "winoground_root", "./winoground_feat"),
                    arch=arch,
                    device=eval_device,
                    batch_size=batch_size,
                    skip_missing=skip_missing,
                )
            )

        if "mmvp" in datasets:
            results.update(
                evaluate_mmvp(
                    core,
                    root=_getattr_default(args, "mmvp_root", "./mmvp_feat"),
                    arch=arch,
                    device=eval_device,
                    batch_size=batch_size,
                    skip_missing=skip_missing,
                )
            )

        results["eval/seconds"] = time.time() - start_time

        msg = " | ".join(f"{k}={v:.4f}" for k, v in sorted(results.items()))
        logging.info("Evaluation finished tag=%s: %s", tag, msg)

        if _getattr_default(args, "eval_write_json", True):
            output_jsonl = _getattr_default(args, "eval_output_jsonl", "")
            if output_jsonl:
                out_path = Path(output_jsonl)
            else:
                out_dir = Path(_getattr_default(args, "output_dir", "."))
                out_path = out_dir / "eval_metrics.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "tag": tag,
                "epoch": epoch,
                "global_step": global_step,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": results,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        core.train(was_training)

    _barrier()
    return results


# -----------------------------------------------------------------------------
# Standalone checkpoint evaluation
# -----------------------------------------------------------------------------


class StandaloneConnectorModel(nn.Module):
    """Same top-level state_dict names as train.py: vision, language, logit_scale."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.vision = VisionConnector.from_open_clip(
            arch=args.arch,
            width=args.vision_width,
            dropout=args.vision_dropout,
            eps=args.normalize_eps,
            force_fp32=True,
            device=None,
        )
        self.language = LanguageGuider(
            arch=args.arch,
            teacher_name=args.teacher_name,
            commonwords_root=args.commonwords_root,
            width=args.language_width,
            dropout=args.language_dropout,
            bias=args.language_bias,
            whitening_eps=args.whitening_eps,
            whitening_chunk_size=args.whitening_chunk_size,
            whitening_max_samples=args.whitening_max_samples,
            whitening_device=args.whitening_device,
            relation_temperature=args.relation_temperature,
            relation_weight=args.relation_weight,
            exclude_self_similarity=True,
            force_fp32=True,
            normalize_eps=args.normalize_eps,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / args.init_temperature))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="", help="Optional train.py checkpoint. If empty, evaluate initialization.")
    parser.add_argument("--arch", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--teacher-name", type=str, default="llm2vec_qwen3_4b")
    parser.add_argument("--commonwords-root", type=str, default="./commonwords79k_feat")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs")

    # Connector construction args must match training checkpoint shapes.
    parser.add_argument("--vision-width", type=float, default=2.0)
    parser.add_argument("--vision-dropout", type=float, default=0.0)
    parser.add_argument("--language-width", type=float, default=2.0)
    parser.add_argument("--language-dropout", type=float, default=0.1)
    parser.add_argument("--language-bias", action="store_true")
    parser.add_argument("--whitening-eps", type=float, default=1e-5)
    parser.add_argument("--whitening-chunk-size", type=int, default=8192)
    parser.add_argument("--whitening-max-samples", type=int, default=None)
    parser.add_argument("--whitening-device", type=str, default="cpu")
    parser.add_argument("--relation-temperature", type=float, default=0.1)
    parser.add_argument("--relation-weight", type=float, default=0.05)
    parser.add_argument("--normalize-eps", type=float, default=1e-6)
    parser.add_argument("--init-temperature", type=float, default=0.07)

    add_eval_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = _resolve_device(args, None)
    model = StandaloneConnectorModel(args)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing checkpoint keys: %s", missing)
        if unexpected:
            logging.warning("Unexpected checkpoint keys: %s", unexpected)
        logging.info("Loaded checkpoint: %s", args.checkpoint)

    model.to(device)
    evaluate_all(model, args, device=device, epoch=None, global_step=None, tag="standalone")


if __name__ == "__main__":
    main()
