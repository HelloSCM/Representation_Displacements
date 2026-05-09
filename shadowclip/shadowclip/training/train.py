#!/usr/bin/env python3
"""
Train ShadowCLIP on pre-extracted features.

Default setup:
    - arch: base
    - vision features: clip_base, dino_base_cls, dino_base_gap
    - text feature: clip_base_short
    - teacher feature: llm2vec_qwen3_4b
    - global batch size: 12288
    - DDP single-node multi-GPU

Example:
    CUDA_VISIBLE_DEVICES=6,7,8,9 torchrun \
      --standalone --nproc_per_node=4 \
      shadowclip train \
      --feature-root ./features \
      --commonwords-root ./commonwords79k_feat \
      --output-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP

from ..models.vision_connector import VisionConnector
from ..models.language_guider import LanguageGuider
from . import eval_module as eval_mod


# -----------------------------------------------------------------------------
# Distributed utilities
# -----------------------------------------------------------------------------


@dataclass
class DistInfo:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def init_distributed() -> DistInfo:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        dist.init_process_group(backend=backend, init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    return DistInfo(
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        x = x.detach().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def reduce_min_int(value: int, device: torch.device) -> int:
    x = torch.tensor([value], dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.MIN)
    return int(x.item())


def gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """All-gather without gradient."""
    if not (dist.is_available() and dist.is_initialized()):
        return x

    world_size = dist.get_world_size()
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.detach())
    return torch.cat(gathered, dim=0)


def gather_with_local_grad(x: torch.Tensor) -> torch.Tensor:
    """
    OpenCLIP-style all-gather.

    Features from other ranks are gathered without gradient. The local slice is
    replaced by the original tensor so local gradients are preserved.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.detach())
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


# -----------------------------------------------------------------------------
# Logging / reproducibility
# -----------------------------------------------------------------------------


def setup_logging(output_dir: Path, rank: int, log_level: str = "INFO") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, log_level.upper()))

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_dir / f"rank{rank}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_precision(allow_tf32: bool) -> None:
    # Keep full FP32 matmul unless explicitly enabling TF32.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Feature shard loading
# -----------------------------------------------------------------------------


def arch_feature_targets(arch: str, clip_text_group: str, teacher_name: str) -> Dict[str, str]:
    if arch == "base":
        return {
            "clip_vision": "clip_base",
            "dino_cls": "dino_base_cls",
            "dino_gap": "dino_base_gap",
            "clip_text": f"clip_base_{clip_text_group}",
            "teacher": teacher_name,
        }

    if arch == "large":
        return {
            "clip_vision": "clip_large",
            "dino_cls": "dino_large_cls",
            "dino_gap": "dino_large_gap",
            "clip_text": f"clip_large_{clip_text_group}",
            "teacher": teacher_name,
        }

    raise ValueError(f"Unsupported arch={arch!r}. Expected 'base' or 'large'.")


def list_shards(
    feature_root: Path,
    reference_target: str,
    shard_start: Optional[int],
    shard_end: Optional[int],
    max_shards: Optional[int],
) -> List[str]:
    ref_dir = feature_root / reference_target
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference feature directory not found: {ref_dir}")

    shard_ids = sorted(p.stem for p in ref_dir.glob("[0-9][0-9][0-9][0-9][0-9].safetensors"))

    def keep(shard_id: str) -> bool:
        try:
            idx = int(shard_id)
        except ValueError:
            return False
        if shard_start is not None and idx < shard_start:
            return False
        if shard_end is not None and idx > shard_end:
            return False
        return True

    shard_ids = [s for s in shard_ids if keep(s)]

    if max_shards is not None:
        shard_ids = shard_ids[:max_shards]

    if not shard_ids:
        raise RuntimeError(f"No feature shards found under {ref_dir}")

    return shard_ids


def read_num_samples(feature_root: Path, target: str, shard_id: str) -> int:
    meta_path = feature_root / target / f"{shard_id}.meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta["num_samples"])

    feat_path = feature_root / target / f"{shard_id}.safetensors"
    data = load_file(str(feat_path), device="cpu")
    return int(data["features"].shape[0])


def verify_shard_files(feature_root: Path, targets: Dict[str, str], shard_ids: Iterable[str]) -> None:
    missing: List[str] = []
    for shard_id in shard_ids:
        for logical_name, target in targets.items():
            path = feature_root / target / f"{shard_id}.safetensors"
            if not path.exists():
                missing.append(f"{logical_name}:{path}")

    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing feature files:\n{preview}{suffix}")


def stable_shard_seed(shard_id: str) -> int:
    try:
        return int(shard_id)
    except ValueError:
        value = 0
        for ch in shard_id:
            value = (value * 131 + ord(ch)) % 2_147_483_647
        return value


class FeatureShardBatcher:
    """
    Shard-streaming batcher.

    Each rank gets a deterministic subset of shard IDs. Within each rank, shards
    and rows can be shuffled every epoch. Each yielded batch contains aligned rows
    from all requested feature targets.
    """

    def __init__(
        self,
        *,
        feature_root: Path,
        targets: Dict[str, str],
        shard_ids: List[str],
        rank: int,
        world_size: int,
        local_batch_size: int,
        seed: int,
        shuffle_shards: bool = True,
        shuffle_samples: bool = True,
        drop_last: bool = True,
        pin_memory: bool = False,
    ) -> None:
        self.feature_root = feature_root
        self.targets = targets
        self.shard_ids = list(shard_ids)
        self.rank = rank
        self.world_size = world_size
        self.local_batch_size = local_batch_size
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.local_shard_ids = [
            shard_id for i, shard_id in enumerate(self.shard_ids) if i % self.world_size == self.rank
        ]

    def count_local_batches(self) -> int:
        total = 0
        ref_target = self.targets["clip_vision"]
        for shard_id in self.local_shard_ids:
            n = read_num_samples(self.feature_root, ref_target, shard_id)
            if self.drop_last:
                total += n // self.local_batch_size
            else:
                total += math.ceil(n / self.local_batch_size)
        return total

    def iter_epoch(self, epoch: int) -> Iterator[Dict[str, torch.Tensor]]:
        shard_ids = list(self.local_shard_ids)

        if self.shuffle_shards:
            rng = random.Random(self.seed + 10_000 * epoch + self.rank)
            rng.shuffle(shard_ids)

        for shard_id in shard_ids:
            shard_data = self._load_shard(shard_id)
            n = min(int(x.shape[0]) for x in shard_data.values())

            if n <= 0:
                continue

            if self.shuffle_samples:
                g = torch.Generator(device="cpu")
                g.manual_seed(self.seed + 1_000_003 * epoch + stable_shard_seed(shard_id))
                indices = torch.randperm(n, generator=g)
            else:
                indices = torch.arange(n)

            if self.drop_last:
                n_use = (n // self.local_batch_size) * self.local_batch_size
            else:
                n_use = n

            if n_use <= 0:
                continue

            indices = indices[:n_use]

            for start in range(0, n_use, self.local_batch_size):
                end = min(start + self.local_batch_size, n_use)

                if self.drop_last and end - start < self.local_batch_size:
                    continue

                sel = indices[start:end]
                batch = {name: tensor.index_select(0, sel) for name, tensor in shard_data.items()}

                if self.pin_memory:
                    batch = {name: tensor.pin_memory() for name, tensor in batch.items()}

                yield batch

            del shard_data

    def _load_shard(self, shard_id: str) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        for logical_name, target in self.targets.items():
            path = self.feature_root / target / f"{shard_id}.safetensors"
            data = load_file(str(path), device="cpu")
            if "features" not in data:
                raise KeyError(f"{path} does not contain tensor named 'features'.")
            out[logical_name] = data["features"]

        row_counts = {name: int(t.shape[0]) for name, t in out.items()}
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"Shard {shard_id} row count mismatch: {row_counts}")

        return out


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device=device, non_blocking=True) for name, tensor in batch.items()}


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class ConnectorTrainModel(nn.Module):
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
            exclude_self_similarity=args.exclude_self_similarity,
            force_fp32=True,
            normalize_eps=args.normalize_eps,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / args.init_temperature))
        self.max_logit_scale = float(args.max_logit_scale)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        relation_scope: str,
        relation_chunk_size: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        image_z = self.vision(
            clip_feat=batch["clip_vision"],
            dino_cls=batch["dino_cls"],
            dino_gap=batch["dino_gap"],
        )

        compute_local_relation = relation_scope == "local"

        lang_out = self.language(
            clip_text_feat=batch["clip_text"],
            llm_guidance=batch["teacher"] if compute_local_relation else None,
            compute_relation_loss=compute_local_relation,
            relation_chunk_size=relation_chunk_size,
            return_dict=True,
        )

        out: Dict[str, torch.Tensor] = {
            "image_z": image_z,
            "text_z": lang_out["text_z"],
            "language_fused": lang_out["language_fused"],
            "logit_scale": self.logit_scale.exp().clamp(max=self.max_logit_scale),
        }

        if compute_local_relation:
            out["relation_kd_loss"] = lang_out["relation_kd_loss"]

        elif relation_scope == "global":
            student_white = self.language.whiten_student(lang_out["language_fused"])
            with torch.no_grad():
                teacher_white = self.language.whiten_teacher(batch["teacher"])

            out["student_white"] = student_white
            out["teacher_white"] = teacher_white

        return out


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


def clip_contrastive_loss(
    image_z: torch.Tensor,
    text_z: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symmetric image-text contrastive loss, OpenCLIP-style.

    In DDP, each rank computes local rows against globally gathered columns.
    This gives global negatives while keeping local gradients.
    """
    local_batch = image_z.shape[0]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        all_image_z = gather_with_local_grad(image_z)
        all_text_z = gather_with_local_grad(text_z)
        labels = rank * local_batch + torch.arange(local_batch, device=image_z.device)
    else:
        all_image_z = image_z
        all_text_z = text_z
        labels = torch.arange(local_batch, device=image_z.device)

    logits_per_image = logit_scale * image_z @ all_text_z.T
    logits_per_text = logit_scale * text_z @ all_image_z.T

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = 0.5 * (loss_i + loss_t)

    return loss, loss_i.detach(), loss_t.detach()


def relation_kd_loss_global_local_rows(
    *,
    student_white: torch.Tensor,
    teacher_white: torch.Tensor,
    temperature: float,
    exclude_self_similarity: bool,
    normalize_eps: float,
    chunk_size: Optional[int],
    scale_by_temperature_squared: bool = True,
) -> torch.Tensor:
    """
    Global relation KD with local rows.

    Each rank computes KL for its local rows against globally gathered columns:

        teacher: softmax(teacher_cosine / T)
        student: log_softmax(student_cosine / T)
        KL(teacher || student)

    NaN/autograd-safe masking:
        We never modify the output of log_softmax in-place. Instead, we compute
        elementwise KL and use a boolean mask to remove diagonal self-similarity.
    """
    if student_white.ndim != 2 or teacher_white.ndim != 2:
        raise ValueError("student_white and teacher_white must both be 2D.")

    local_batch = student_white.shape[0]
    if local_batch < 1:
        return student_white.new_zeros(())

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    student_local = F.normalize(student_white.float(), p=2, dim=-1, eps=normalize_eps)
    with torch.no_grad():
        teacher_local = F.normalize(teacher_white.float(), p=2, dim=-1, eps=normalize_eps)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        global_student = gather_with_local_grad(student_local)
        global_teacher = gather_no_grad(teacher_local)
        global_offset = rank * local_batch
    else:
        global_student = student_local
        global_teacher = teacher_local
        global_offset = 0

    global_batch = global_student.shape[0]
    if global_batch < 2:
        return student_white.new_zeros(())

    if chunk_size is None or chunk_size <= 0:
        chunk_size = local_batch

    total = student_white.new_zeros(())
    total_rows = 0

    # Finite mask avoids softmax/log_softmax all -inf edge cases.
    mask_value = -1e4

    for start in range(0, local_batch, chunk_size):
        end = min(start + chunk_size, local_batch)
        rows = end - start

        student_logits = student_local[start:end] @ global_student.T
        teacher_logits = teacher_local[start:end] @ global_teacher.T

        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        valid_mask = None

        if exclude_self_similarity:
            diag_row_idx = torch.arange(rows, device=student_logits.device)
            diag_col_idx = torch.arange(global_offset + start, global_offset + end, device=student_logits.device)

            # Clone before indexed assignment to avoid any view/version surprises.
            student_logits = student_logits.clone()
            teacher_logits = teacher_logits.clone()

            student_logits[diag_row_idx, diag_col_idx] = mask_value
            teacher_logits[diag_row_idx, diag_col_idx] = mask_value

            valid_mask = torch.ones_like(student_logits, dtype=torch.bool)
            valid_mask[diag_row_idx, diag_col_idx] = False

        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=-1)

        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # Manual KL:
        #   sum_j p_t(j) * (log p_t(j) - log p_s(j))
        #
        # This is equivalent to F.kl_div(log_p_s, p_t), but lets us mask
        # diagonal entries without in-place editing log_softmax output.
        teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
        elem_kl = teacher_probs * (teacher_log_probs - student_log_probs)

        if valid_mask is not None:
            elem_kl = elem_kl.masked_fill(~valid_mask, 0.0)

        chunk_loss = elem_kl.sum(dim=-1).mean()

        total = total + chunk_loss * rows
        total_rows += rows

    loss = total / float(total_rows)

    if scale_by_temperature_squared:
        loss = loss * (temperature ** 2)

    return loss


# -----------------------------------------------------------------------------
# Optimizer / schedule
# -----------------------------------------------------------------------------


def create_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    logit_scale_params = []

    no_decay_keywords = (
        "bias",
        "ln",
        "norm",
        "residual_gate",
        "whiten",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_l = name.lower()

        if name == "logit_scale":
            logit_scale_params.append(param)
        elif param.ndim < 2 or any(k in name_l for k in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []

    if decay_params:
        groups.append({"params": decay_params, "weight_decay": args.weight_decay, "lr": args.lr, "base_lr": args.lr})

    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": args.lr, "base_lr": args.lr})

    if logit_scale_params:
        logit_lr = args.logit_scale_lr if args.logit_scale_lr is not None else args.lr
        groups.append({"params": logit_scale_params, "weight_decay": 0.0, "lr": logit_lr, "base_lr": logit_lr})

    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)


def set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr_reference: float,
    min_lr: float,
) -> None:
    if total_steps <= 0:
        return

    if warmup_steps > 0 and step < warmup_steps:
        warmup_factor = float(step + 1) / float(warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * warmup_factor
        return

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        base_lr = group["base_lr"]
        if base_lr_reference > 0:
            group_min_lr = min_lr * (base_lr / base_lr_reference)
        else:
            group_min_lr = min_lr
        group["lr"] = group_min_lr + (base_lr - group_min_lr) * cosine


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(state, path)


def load_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    epoch = int(ckpt.get("epoch", -1))
    global_step = int(ckpt.get("global_step", 0))
    return epoch + 1, global_step


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    dist_info = init_distributed()
    output_dir = Path(args.output_dir)

    setup_logging(output_dir, dist_info.rank, args.log_level)
    seed_everything(args.seed + dist_info.rank)
    configure_precision(args.allow_tf32)

    logging.info(
        "distributed=%s rank=%d world_size=%d local_rank=%d device=%s",
        dist_info.distributed,
        dist_info.rank,
        dist_info.world_size,
        dist_info.local_rank,
        dist_info.device,
    )

    if args.global_batch_size % dist_info.world_size != 0:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must be divisible by world_size={dist_info.world_size}."
        )

    local_batch_size = args.global_batch_size // dist_info.world_size
    logging.info("global_batch_size=%d local_batch_size=%d", args.global_batch_size, local_batch_size)

    feature_root = Path(args.feature_root)
    targets = arch_feature_targets(args.arch, args.clip_text_group, args.teacher_name)
    shard_ids = list_shards(
        feature_root=feature_root,
        reference_target=targets["clip_vision"],
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        max_shards=args.max_shards,
    )

    if args.verify_feature_files:
        verify_shard_files(feature_root, targets, shard_ids)

    batcher = FeatureShardBatcher(
        feature_root=feature_root,
        targets=targets,
        shard_ids=shard_ids,
        rank=dist_info.rank,
        world_size=dist_info.world_size,
        local_batch_size=local_batch_size,
        seed=args.seed,
        shuffle_shards=not args.no_shuffle_shards,
        shuffle_samples=not args.no_shuffle_samples,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    local_steps_per_epoch = batcher.count_local_batches()
    steps_per_epoch = reduce_min_int(local_steps_per_epoch, dist_info.device)

    if args.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)

    if steps_per_epoch <= 0:
        raise RuntimeError("No training steps available. Try reducing --global-batch-size or increasing data range.")

    if local_steps_per_epoch != steps_per_epoch:
        logging.warning(
            "local_steps_per_epoch=%d but synchronized steps_per_epoch=%d; extra local batches will be skipped to keep DDP ranks aligned.",
            local_steps_per_epoch,
            steps_per_epoch,
        )

    total_steps = steps_per_epoch * args.epochs

    if args.warmup_steps is None or args.warmup_steps < 0:
        warmup_steps = int(round(args.warmup_ratio * total_steps))
    else:
        warmup_steps = int(args.warmup_steps)

    logging.info("targets=%s", targets)
    logging.info("num_shards=%d local_shards=%d", len(shard_ids), len(batcher.local_shard_ids))
    logging.info(
        "steps_per_epoch=%d epochs=%d total_steps=%d warmup_steps=%d",
        steps_per_epoch,
        args.epochs,
        total_steps,
        warmup_steps,
    )

    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

    barrier()

    model = ConnectorTrainModel(args).to(dist_info.device)

    effective_relation_scope = args.relation_scope
    if args.relation_weight <= 0:
        effective_relation_scope = "none"

    find_unused = args.ddp_find_unused_parameters or effective_relation_scope == "none"

    if dist_info.distributed:
        model = DDP(
            model,
            device_ids=[dist_info.local_rank] if dist_info.device.type == "cuda" else None,
            output_device=dist_info.local_rank if dist_info.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=find_unused,
        )

    optimizer = create_optimizer(unwrap_model(model), args)

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            path=Path(args.resume),
            model=model,
            optimizer=optimizer if not args.resume_model_only else None,
            device=dist_info.device,
        )
        logging.info("resumed from %s at epoch=%d global_step=%d", args.resume, start_epoch, global_step)

    if getattr(args, "eval_before_train", True):
        eval_mod.evaluate_all(
            model,
            args,
            device=dist_info.device,
            epoch=-1,
            global_step=global_step,
            tag="before_train",
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_iter = batcher.iter_epoch(epoch)

        running = {"loss": 0.0, "contrastive": 0.0, "relation": 0.0, "loss_i": 0.0, "loss_t": 0.0}
        running_count = 0

        for step_in_epoch in range(steps_per_epoch):
            try:
                batch_cpu = next(epoch_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Rank {dist_info.rank} iterator ended early at step {step_in_epoch}; expected {steps_per_epoch} steps."
                ) from exc

            batch = move_batch_to_device(batch_cpu, dist_info.device)

            set_cosine_lr(
                optimizer,
                step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                base_lr_reference=args.lr,
                min_lr=args.min_lr,
            )

            optimizer.zero_grad(set_to_none=True)

            out = model(batch, relation_scope=effective_relation_scope, relation_chunk_size=args.relation_chunk_size)

            contrastive_loss, loss_i, loss_t = clip_contrastive_loss(
                image_z=out["image_z"],
                text_z=out["text_z"],
                logit_scale=out["logit_scale"],
            )

            if effective_relation_scope == "none":
                relation_loss = contrastive_loss.new_zeros(())
            elif effective_relation_scope == "local":
                relation_loss = out["relation_kd_loss"]
            elif effective_relation_scope == "global":
                relation_loss = relation_kd_loss_global_local_rows(
                    student_white=out["student_white"],
                    teacher_white=out["teacher_white"],
                    temperature=args.relation_temperature,
                    exclude_self_similarity=args.exclude_self_similarity,
                    normalize_eps=args.normalize_eps,
                    chunk_size=args.relation_chunk_size,
                    scale_by_temperature_squared=args.relation_scale_by_t2,
                )
            else:
                raise ValueError(f"Unsupported relation_scope={effective_relation_scope!r}")

            loss = args.contrastive_weight * contrastive_loss + args.relation_weight * relation_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch} step={step_in_epoch}: "
                    f"loss={loss.item()} contrastive={contrastive_loss.item()} relation={relation_loss.item()}"
                )

            loss.backward()

            if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()

            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0.0, math.log(args.max_logit_scale))

            global_step += 1

            if (
                args.eval_every_steps > 0
                and global_step % args.eval_every_steps == 0
            ):
                eval_mod.evaluate_all(
                    model,
                    args,
                    device=dist_info.device,
                    epoch=epoch + 1,
                    global_step=global_step,
                    tag=f"step_{global_step}",
                )

                model.train()

            metrics = torch.stack([loss.detach(), contrastive_loss.detach(), relation_loss.detach(), loss_i.detach(), loss_t.detach()])
            metrics = reduce_mean(metrics)

            running["loss"] += float(metrics[0].item())
            running["contrastive"] += float(metrics[1].item())
            running["relation"] += float(metrics[2].item())
            running["loss_i"] += float(metrics[3].item())
            running["loss_t"] += float(metrics[4].item())
            running_count += 1

            if is_rank0() and (global_step % args.log_every == 0 or step_in_epoch == 0):
                denom = max(1, running_count)
                logit_scale_value = float(unwrap_model(model).logit_scale.exp().detach().cpu().item())
                logging.info(
                    "epoch=%d/%d step=%d/%d global_step=%d lr=%.3e logit_scale=%.3f "
                    "loss=%.5f contrastive=%.5f relation=%.5f loss_i=%.5f loss_t=%.5f",
                    epoch + 1,
                    args.epochs,
                    step_in_epoch + 1,
                    steps_per_epoch,
                    global_step,
                    get_lr(optimizer),
                    logit_scale_value,
                    running["loss"] / denom,
                    running["contrastive"] / denom,
                    running["relation"] / denom,
                    running["loss_i"] / denom,
                    running["loss_t"] / denom,
                )
                for k in running:
                    running[k] = 0.0
                running_count = 0

        barrier()

        if is_rank0():
            save_checkpoint(path=output_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args)

            if (epoch + 1) % args.save_every_epoch == 0:
                save_checkpoint(
                    path=output_dir / f"epoch_{epoch + 1:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                )

        barrier()

        '''
        if args.eval_every_epoch > 0 and (epoch + 1) % args.eval_every_epoch == 0:
            eval_mod.evaluate_all(
                model,
                args,
                device=dist_info.device,
                epoch=epoch + 1,
                global_step=global_step,
                tag=f"epoch_{epoch + 1}",
            )

        barrier()
        '''

    if is_rank0():
        save_checkpoint(path=output_dir / "final.pt", model=model, optimizer=optimizer, epoch=args.epochs - 1, global_step=global_step, args=args)
        logging.info("training finished. final checkpoint saved to %s", output_dir / "final.pt")

    cleanup_distributed()


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--feature-root", type=str, default="./features")
    parser.add_argument("--commonwords-root", type=str, default="./commonwords79k_feat")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--arch", type=str, default="large", choices=["base", "large"])
    parser.add_argument("--clip-text-group", type=str, default="long", choices=["short", "long"])
    parser.add_argument("--teacher-name", type=str, default="llm2vec_qwen3_4b")
    parser.add_argument("--shard-start", type=int, default=None)
    parser.add_argument("--shard-end", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--verify-feature-files", action="store_true")
    parser.add_argument("--no-shuffle-shards", action="store_true")
    parser.add_argument("--no-shuffle-samples", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")

    # Batch / training length
    parser.add_argument("--global-batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)

    # VisionConnector
    parser.add_argument("--vision-width", type=float, default=2.0)
    parser.add_argument("--vision-dropout", type=float, default=0.1)

    # LanguageGuider connector branch
    parser.add_argument("--language-width", type=float, default=2.0)
    parser.add_argument("--language-dropout", type=float, default=0.1)
    parser.add_argument("--language-bias", action="store_true")

    # Whitening init
    parser.add_argument("--whitening-eps", type=float, default=1e-5)
    parser.add_argument("--whitening-chunk-size", type=int, default=8192)
    parser.add_argument("--whitening-max-samples", type=int, default=None)
    parser.add_argument("--whitening-device", type=str, default="cpu")

    # Contrastive loss
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--init-temperature", type=float, default=0.07)
    parser.add_argument("--max-logit-scale", type=float, default=100.0)
    parser.add_argument("--normalize-eps", type=float, default=1e-6)

    # Relation distillation
    parser.add_argument("--relation-scope", type=str, default="global", choices=["none", "local", "global"])
    parser.add_argument("--relation-temperature", type=float, default=0.1)
    parser.add_argument("--relation-weight", type=float, default=10.0)
    parser.add_argument("--relation-chunk-size", type=int, default=1024)
    parser.add_argument("--relation-scale-by-t2", dest="relation_scale_by_t2", action="store_true")
    parser.add_argument("--no-relation-scale-by-t2", dest="relation_scale_by_t2", action="store_false")
    parser.set_defaults(relation_scale_by_t2=True)
    parser.add_argument("--no-exclude-self-similarity", dest="exclude_self_similarity", action="store_false")
    parser.set_defaults(exclude_self_similarity=True)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--logit-scale-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    # Runtime / logging / checkpoint
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-epoch", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")

    eval_mod.add_eval_args(parser)

    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
