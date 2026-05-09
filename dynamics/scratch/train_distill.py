"""Train ViT-Small from random initialization with optional ViT-Base distillation.

Experiment covered by this script
---------------------------------
Teacher:
    timm ViT-Base ImageNet1k pretrained model
    default: vit_base_patch16_224.augreg_in1k

Student:
    timm ViT-Small randomly initialized model
    default: vit_small_patch16_224 with pretrained=False

Training modes:
    ce           : classification loss only
    feature_ce   : classification loss + feature distillation MSE
    relation_ce  : classification loss + relation distillation KL

Metrics:
    epoch 0 is evaluated before any update, then every epoch after training.
    Since teacher/student hidden dimensions differ (ViT-B 768 vs ViT-S 384),
    --metric_proj_dim defaults to 384 so PCA/whitening brings both feature spaces
    to a common dimensionality before OPA metrics are computed.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from data_utils import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_VAL_DIR,
    create_loader,
    make_dual_imagefolder,
    make_imagefolder,
)
from feature_utils import pool_tokens
from losses import relation_kl_loss
from metrics import compute_metric_suite
from utils import (
    append_csv_row,
    barrier,
    cleanup_distributed,
    is_main_process,
    reduce_mean,
    save_checkpoint,
    save_json,
    seed_everything,
    setup_distributed,
    unwrap_model,
)


TEACHER_MODEL = "vit_base_patch16_224.augreg_in1k"
STUDENT_MODEL = "vit_small_patch16_224"
DEFAULT_PROJECT_DIR = "."
DEFAULT_CKPT_DIR = "./checkpoints"


class ViTClassifierWithLastFeature(nn.Module):
    """timm ViT classifier that also exposes the last block image-level feature.

    This avoids torchvision FX tracing and works with newer timm/PyTorch attention
    implementations. The hook captures model.blocks[-1] output, which is a token
    tensor [B, T, D] for ViT models. We then pool it the same way as the static
    representation code: CLS token for standard ViTs, mean pooling for SigLIP/I-JEPA.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.model_name = model_name
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        if not hasattr(self.base_model, "blocks") or len(self.base_model.blocks) == 0:
            raise ValueError(f"{model_name} does not look like a timm ViT with blocks.")
        self._last_block_tokens: Optional[torch.Tensor] = None
        self._hook = self.base_model.blocks[-1].register_forward_hook(self._capture_last_block)

    @property
    def feat_dim(self) -> int:
        if hasattr(self.base_model, "num_features"):
            return int(self.base_model.num_features)
        if hasattr(self.base_model, "embed_dim"):
            return int(self.base_model.embed_dim)
        raise ValueError("Cannot infer feature dim from timm model.")

    def _capture_last_block(self, _module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self._last_block_tokens = output

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._last_block_tokens = None
        logits = self.base_model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if self._last_block_tokens is None:
            raise RuntimeError("Last-block hook did not fire; cannot obtain feature.")
        feat = pool_tokens(self._last_block_tokens, self.model_name)
        return logits, feat


class StudentToTeacherProjector(nn.Module):
    """Learnable projection used when ViT-S and ViT-B feature dims differ."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def parse_args():
    parser = argparse.ArgumentParser(
        "ViT-B ImageNet teacher -> random ViT-S student training dynamics"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        required=True,
        choices=["ce", "feature_ce", "relation_ce"],
        help="ce: classification only; feature_ce/relation_ce: CE plus distillation.",
    )
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--teacher_pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--student_pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep False for random initialization; set only for debugging.",
    )
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output_dir", type=str, default=os.path.join(DEFAULT_PROJECT_DIR, "results"))
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help=(
            "Per-GPU batch size under torchrun. With 8 GPUs, default 2048 gives "
            "global batch size 16384. Set 1024 for global batch size 8192."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="enable CUDA AMP")

    # Loss weights.
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1, help="relation distillation temperature")

    # Optimizer and LR schedule. If --lr is negative, LR is computed by linear scaling.
    parser.add_argument("--lr", type=float, default=-1.0,
                        help="Actual LR. If <0, use base_lr * global_batch_size / base_batch_size.")
    parser.add_argument("--base_lr", type=float, default=3e-4,
                        help="Base LR at --base_batch_size for linear scaling.")
    parser.add_argument("--base_batch_size", type=int, default=1024)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Metrics after epoch 0 and each training epoch.
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--metric_batch_size", type=int, default=256)
    parser.add_argument(
        "--metric_proj_dim",
        type=int,
        default=384,
        help="Default 384 because ViT-B and ViT-S hidden dims differ; set >0 for PCA/whiten.",
    )
    parser.add_argument("--metric_pair_sample_size", type=int, default=1000)
    parser.add_argument("--metric_max_samples", type=int, default=-1, help="-1 means all ImageNet val samples")
    parser.add_argument(
        "--include_initial_metric_reference",
        action="store_true",
        help=(
            "Also compute metrics against the initial random ViT-Small. "
            "By default, metrics are computed only against the teacher model."
        ),
    )

    # Validation accuracy for the supervised student classifier.
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_max_samples", type=int, default=-1)

    return parser.parse_args()


def build_experiment_dirs(args):
    exp_name = args.loss_type
    result_dir = os.path.join(args.output_dir, "distill_vitbase_to_vitsmall", exp_name)
    ckpt_dir = os.path.join(args.checkpoint_dir, "distill_vitbase_to_vitsmall", exp_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return result_dir, ckpt_dir


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=(enabled and device.type == "cuda"))


def get_base_model(model: nn.Module) -> nn.Module:
    model = unwrap_model(model)
    return model.base_model if hasattr(model, "base_model") else model


def effective_global_batch_size(args, world_size: int) -> int:
    return int(args.batch_size * max(world_size, 1))


def resolve_base_lr(args, global_batch: int) -> float:
    if args.lr is not None and args.lr > 0:
        return float(args.lr)
    return float(args.base_lr) * float(global_batch) / float(args.base_batch_size)


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate_student_classifier(
    model: nn.Module,
    loader,
    device: torch.device,
    args,
) -> float:
    model.eval()
    correct = 0.0
    total = 0
    iterator = tqdm(loader, desc="student val", disable=not is_main_process(), leave=False)
    for batch in iterator:
        images, target = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with amp_context(device, args.amp):
            logits, _ = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == target).float().sum().item()
        total += target.numel()
    return correct / max(total, 1)


def train_one_epoch(
    epoch: int,
    args,
    student_model: nn.Module,
    teacher_model: Optional[nn.Module],
    projector: Optional[nn.Module],
    loader,
    sampler,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    steps_per_epoch: int,
):
    if sampler is not None:
        sampler.set_epoch(epoch)

    student_model.train()
    if teacher_model is not None:
        teacher_model.eval()
    if projector is not None:
        projector.train()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_ce = 0.0
    total_distill = 0.0
    total_batches = 0

    iterator = tqdm(loader, desc=f"train epoch {epoch}", disable=not is_main_process())
    for step, batch in enumerate(iterator):
        global_step = (epoch - 1) * steps_per_epoch + step
        lr = cosine_warmup_lr(
            step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            min_lr=args.min_lr,
        )
        set_optimizer_lr(optimizer, lr)

        if args.loss_type == "ce":
            student_img, target = batch[0], batch[1]
            teacher_img = None
        else:
            student_img, teacher_img, target = batch[0], batch[1], batch[2]

        student_img = student_img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if teacher_img is not None:
            teacher_img = teacher_img.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        teacher_feat = None
        if teacher_model is not None and teacher_img is not None:
            with torch.no_grad():
                with amp_context(device, args.amp):
                    _, teacher_feat = teacher_model(teacher_img)

        with amp_context(device, args.amp):
            logits, student_feat = student_model(student_img)
            ce_loss = criterion(logits.float(), target)

            if args.loss_type == "ce":
                distill_loss = student_feat.float().sum() * 0.0
            elif args.loss_type == "feature_ce":
                if projector is None:
                    raise RuntimeError("feature_ce requires a student->teacher projector.")
                student_proj = projector(student_feat)
                distill_loss = F.mse_loss(student_proj.float(), teacher_feat.detach().float())
            elif args.loss_type == "relation_ce":
                distill_loss = relation_kl_loss(
                    student_feat,
                    teacher_feat,
                    temperature=args.temperature,
                    mask_diagonal=True,
                )
            else:
                raise ValueError(args.loss_type)

            loss = args.ce_weight * ce_loss + args.distill_weight * distill_loss

        scaler.scale(loss).backward()
        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                args.grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        loss_mean = reduce_mean(loss.detach())
        ce_mean = reduce_mean(ce_loss.detach())
        distill_mean = reduce_mean(distill_loss.detach())

        total_loss += loss_mean.item()
        total_ce += ce_mean.item()
        total_distill += distill_mean.item()
        total_batches += 1

        if is_main_process():
            iterator.set_postfix(
                loss=f"{loss_mean.item():.5f}",
                ce=f"{ce_mean.item():.5f}",
                dist=f"{distill_mean.item():.5f}",
                lr=f"{lr:.2e}",
            )

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce / denom,
        "distill_loss": total_distill / denom,
        "lr_last": lr if total_batches > 0 else base_lr,
    }


def run_epoch_metrics(
    args,
    epoch: int,
    ckpt_path: str,
    initial_ckpt_path: str,
    result_dir: str,
    device: torch.device,
):
    if args.skip_metrics:
        return
    metric_dir = os.path.join(result_dir, "metrics")
    cache_dir = os.path.join(result_dir, "feature_cache")
    reference_specs = [
        {
            "label": "teacher_vit_base_imagenet1k",
            "model_name": args.teacher_model,
            "pretrained": bool(args.teacher_pretrained),
        }
    ]
    if args.include_initial_metric_reference:
        reference_specs.append(
            {
                "label": "initial_student_random_vit_small",
                "model_name": args.student_model,
                "checkpoint_path": initial_ckpt_path,
                "pretrained": False,
            }
        )

    compute_metric_suite(
        current_spec={
            "label": f"student_epoch_{epoch:03d}",
            "model_name": args.student_model,
            "checkpoint_path": ckpt_path,
            "pretrained": False,
        },
        reference_specs=reference_specs,
        val_dir=args.val_dir,
        out_dir=metric_dir,
        epoch=epoch,
        batch_size=args.metric_batch_size,
        num_workers=args.num_workers,
        proj_dim=args.metric_proj_dim,
        pair_sample_size=args.metric_pair_sample_size,
        seed=args.seed,
        device=device,
        max_samples=args.metric_max_samples,
        cache_dir=cache_dir,
        summary_csv=os.path.join(result_dir, "metrics_summary.csv"),
    )


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    seed_everything(args.seed + rank)

    result_dir, ckpt_dir = build_experiment_dirs(args)
    global_batch = effective_global_batch_size(args, world_size)
    base_lr = resolve_base_lr(args, global_batch)

    if is_main_process():
        save_json(os.path.join(result_dir, "args.json"), vars(args))
        print(f"Result dir: {result_dir}")
        print(f"Checkpoint dir: {ckpt_dir}")
        print(f"World size: {world_size}; device: {device}")
        print(f"Per-GPU batch size: {args.batch_size}; global batch size: {global_batch}")
        print(f"Resolved base LR: {base_lr:.6g}; warmup epochs: {args.warmup_epochs}; min LR: {args.min_lr}")
        if args.metric_proj_dim <= 0:
            print("WARNING: metric_proj_dim <= 0. ViT-B and ViT-S dims differ, so metrics will fail.")

    # For CE-only we do not need teacher transforms/forward. For distillation modes,
    # return student and teacher views of the same image.
    if args.loss_type == "ce":
        train_ds = make_imagefolder(
            args.train_dir,
            model_name=args.student_model,
            is_training=True,
        )
    else:
        train_ds = make_dual_imagefolder(
            args.train_dir,
            model_a_name=args.student_model,
            model_b_name=args.teacher_model,
            is_training=True,
        )
    train_loader, train_sampler = create_loader(
        train_ds,
        batch_size=args.batch_size,
        is_training=True,
        distributed=distributed,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_loader = None
    if is_main_process() and not args.skip_eval:
        val_ds = make_imagefolder(
            args.val_dir,
            model_name=args.student_model,
            is_training=False,
            max_samples=args.eval_max_samples,
        )
        val_loader, _ = create_loader(
            val_ds,
            batch_size=args.eval_batch_size,
            is_training=False,
            distributed=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    teacher_model = None
    if args.loss_type in {"feature_ce", "relation_ce"}:
        teacher_model = ViTClassifierWithLastFeature(
            args.teacher_model,
            pretrained=bool(args.teacher_pretrained),
            num_classes=0,
        ).to(device)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()

    student_model = ViTClassifierWithLastFeature(
        args.student_model,
        pretrained=bool(args.student_pretrained),
        num_classes=1000,
    ).to(device)
    student_feat_dim = student_model.feat_dim
    teacher_feat_dim = teacher_model.feat_dim if teacher_model is not None else None

    projector = None
    if args.loss_type == "feature_ce":
        projector = StudentToTeacherProjector(student_feat_dim, teacher_feat_dim).to(device)
        if is_main_process():
            print(f"Using learnable feature projector: {student_feat_dim} -> {teacher_feat_dim}")

    # Save initial random student before any update, then compute epoch-0 metrics.
    initial_ckpt_path = os.path.join(ckpt_dir, "epoch_000.pth")
    if is_main_process():
        extra = {}
        if projector is not None:
            extra["feature_projector"] = projector.state_dict()
        save_checkpoint(initial_ckpt_path, student_model.base_model, epoch=0, args=args, extra=extra)
        print(f"Saved initial student checkpoint: {initial_ckpt_path}")
        run_epoch_metrics(
            args=args,
            epoch=0,
            ckpt_path=initial_ckpt_path,
            initial_ckpt_path=initial_ckpt_path,
            result_dir=result_dir,
            device=device,
        )
        if val_loader is not None:
            acc0 = evaluate_student_classifier(student_model, val_loader, device, args)
            append_csv_row(os.path.join(result_dir, "val_accuracy.csv"), {
                "epoch": 0,
                "val_top1": acc0,
                "checkpoint": initial_ckpt_path,
            })
            print(f"[epoch 0] val top1={acc0:.4f}")
    barrier()

    if distributed:
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)

    params = [p for p in student_model.parameters() if p.requires_grad]
    if projector is not None:
        params += [p for p in projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=base_lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(device, args.amp)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    train_log = os.path.join(result_dir, "train_loss.csv")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        stats = train_one_epoch(
            epoch=epoch,
            args=args,
            student_model=student_model,
            teacher_model=teacher_model,
            projector=projector,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            base_lr=base_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            steps_per_epoch=steps_per_epoch,
        )

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
        if is_main_process():
            extra = {
                "global_batch_size": global_batch,
                "resolved_base_lr": base_lr,
            }
            if projector is not None:
                extra["feature_projector"] = projector.state_dict()
            # Save raw timm base_model weights so metric code can load the checkpoint
            # directly into timm.create_model(..., num_classes=0).
            save_checkpoint(ckpt_path, get_base_model(student_model), optimizer=optimizer, epoch=epoch, args=args, extra=extra)
            append_csv_row(train_log, {
                "epoch": epoch,
                "loss": stats["loss"],
                "ce_loss": stats["ce_loss"],
                "distill_loss": stats["distill_loss"],
                "lr_last": stats["lr_last"],
                "seconds": time.time() - start,
                "checkpoint": ckpt_path,
            })
            print(
                f"[epoch {epoch}] loss={stats['loss']:.6f} "
                f"ce={stats['ce_loss']:.6f} distill={stats['distill_loss']:.6f} "
                f"saved={ckpt_path}"
            )
        barrier()

        if is_main_process():
            run_epoch_metrics(
                args=args,
                epoch=epoch,
                ckpt_path=ckpt_path,
                initial_ckpt_path=initial_ckpt_path,
                result_dir=result_dir,
                device=device,
            )
            if val_loader is not None:
                # Evaluate the current in-memory model, avoiding a reload.
                eval_model = unwrap_model(student_model)
                acc = evaluate_student_classifier(eval_model, val_loader, device, args)
                append_csv_row(os.path.join(result_dir, "val_accuracy.csv"), {
                    "epoch": epoch,
                    "val_top1": acc,
                    "checkpoint": ckpt_path,
                })
                print(f"[epoch {epoch}] val top1={acc:.4f}")
        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
