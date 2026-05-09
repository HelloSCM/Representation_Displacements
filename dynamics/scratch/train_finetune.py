import argparse
import os
import time

import timm
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data_utils import DEFAULT_TRAIN_DIR, DEFAULT_VAL_DIR, build_transform_for_model, create_loader
from linear_probe import evaluate_timm_classifier
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

from torchvision.datasets import ImageFolder


CLIP_MODEL = "vit_base_patch16_clip_224.laion2b"
AUGREG_MODEL = "vit_base_patch16_224.augreg_in1k"
DEFAULT_PROJECT_DIR = "."
DEFAULT_CKPT_DIR = "./checkpoints"


def parse_args():
    parser = argparse.ArgumentParser("Fine-tuning dynamics for CLIP ViT on ImageNet1k")
    parser.add_argument("--model_name", type=str, default=CLIP_MODEL)
    parser.add_argument("--reference_model", type=str, default=AUGREG_MODEL)
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output_dir", type=str, default=os.path.join(DEFAULT_PROJECT_DIR, "results"))
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)

    parser.add_argument("--epochs", type=int, default=20, help="full fine-tuning epochs; warmup LP is not counted")
    parser.add_argument("--batch_size", type=int, default=128, help="per-GPU batch size under torchrun")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--linear_probe_lr", type=float, default=1e-3)
    parser.add_argument("--linear_probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--metric_batch_size", type=int, default=64)
    parser.add_argument("--metric_proj_dim", type=int, default=-1)
    parser.add_argument("--metric_pair_sample_size", type=int, default=1000)
    parser.add_argument("--metric_max_samples", type=int, default=-1)

    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_max_val_samples", type=int, default=-1)
    return parser.parse_args()


def build_experiment_dirs(args):
    result_dir = os.path.join(args.output_dir, "finetune", "clip_to_imagenet")
    ckpt_dir = os.path.join(args.checkpoint_dir, "finetune", "clip_to_imagenet")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return result_dir, ckpt_dir


def set_linear_probe_mode(model: torch.nn.Module):
    base = unwrap_model(model)
    for p in base.parameters():
        p.requires_grad_(False)
    # timm ViT classifier is usually named `head`.
    if hasattr(base, "head"):
        for p in base.head.parameters():
            p.requires_grad_(True)
    else:
        # Fallback: unfreeze classifier module returned by get_classifier if possible.
        classifier = base.get_classifier() if hasattr(base, "get_classifier") else None
        if classifier is None:
            raise ValueError("Cannot find classifier head for linear probing.")
        for p in classifier.parameters():
            p.requires_grad_(True)


def set_full_finetune_mode(model: torch.nn.Module):
    base = unwrap_model(model)
    for p in base.parameters():
        p.requires_grad_(True)


def train_classification_epoch(
    model,
    loader,
    sampler,
    optimizer,
    criterion,
    scaler,
    device,
    epoch_label,
    amp=False,
):
    if sampler is not None:
        # epoch_label may be str for LP; only set when int-like.
        try:
            sampler.set_epoch(int(epoch_label))
        except Exception:
            sampler.set_epoch(0)
    model.train()
    total_loss = 0.0
    total_batches = 0
    iterator = tqdm(loader, desc=f"train {epoch_label}", disable=not is_main_process())
    for images, target in iterator:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_mean = reduce_mean(loss.detach())
        total_loss += loss_mean.item()
        total_batches += 1
        if is_main_process():
            iterator.set_postfix(loss=f"{loss_mean.item():.5f}")
    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    seed_everything(args.seed + rank)

    result_dir, ckpt_dir = build_experiment_dirs(args)
    if is_main_process():
        save_json(os.path.join(result_dir, "args.json"), vars(args))
        print(f"Result dir: {result_dir}")
        print(f"Checkpoint dir: {ckpt_dir}")
        print(f"World size: {world_size}; device: {device}")

    train_tf, _ = build_transform_for_model(args.model_name, is_training=True)
    train_ds = ImageFolder(args.train_dir, transform=train_tf)
    train_loader, train_sampler = create_loader(
        train_ds,
        batch_size=args.batch_size,
        is_training=True,
        distributed=distributed,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()

    # 1 epoch linear probing warmup; not counted in args.epochs.
    if is_main_process():
        print("Starting 1-epoch linear-probing warmup. This epoch is not counted as full fine-tuning.")
    set_linear_probe_mode(model)
    lp_model = DDP(model, device_ids=[local_rank], output_device=local_rank) if distributed else model
    lp_params = [p for p in lp_model.parameters() if p.requires_grad]
    lp_optimizer = torch.optim.AdamW(lp_params, lr=args.linear_probe_lr, weight_decay=args.linear_probe_weight_decay)
    scaler = GradScaler(enabled=args.amp)
    lp_loss = train_classification_epoch(
        lp_model,
        train_loader,
        train_sampler,
        lp_optimizer,
        criterion,
        scaler,
        device,
        epoch_label="linear_probe_warmup",
        amp=args.amp,
    )
    if distributed:
        model = lp_model.module
        del lp_model
    if is_main_process():
        warmup_ckpt = os.path.join(ckpt_dir, "linear_probe_warmup.pth")
        save_checkpoint(warmup_ckpt, model, optimizer=lp_optimizer, epoch=0, args=args)
        append_csv_row(os.path.join(result_dir, "train_loss.csv"), {
            "epoch": 0,
            "stage": "linear_probe_warmup",
            "loss": lp_loss,
            "checkpoint": warmup_ckpt,
        })
        print(f"Linear-probing warmup loss={lp_loss:.6f}; saved={warmup_ckpt}")
    barrier()

    # Full fine-tuning.
    set_full_finetune_mode(model)
    ft_model = DDP(model, device_ids=[local_rank], output_device=local_rank) if distributed else model
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = train_classification_epoch(
            ft_model,
            train_loader,
            train_sampler,
            optimizer,
            criterion,
            scaler,
            device,
            epoch_label=epoch,
            amp=args.amp,
        )
        if is_main_process():
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            save_checkpoint(ckpt_path, ft_model, optimizer=optimizer, epoch=epoch, args=args)
            append_csv_row(os.path.join(result_dir, "train_loss.csv"), {
                "epoch": epoch,
                "stage": "full_finetune",
                "loss": avg_loss,
                "seconds": time.time() - start,
                "checkpoint": ckpt_path,
            })
            print(f"[epoch {epoch}] loss={avg_loss:.6f} saved={ckpt_path}")
        barrier()

        if is_main_process():
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            acc = evaluate_timm_classifier(
                model_name=args.model_name,
                checkpoint_path=ckpt_path,
                val_dir=args.val_dir,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                num_classes=1000,
                max_val_samples=args.eval_max_val_samples,
            )
            append_csv_row(os.path.join(result_dir, "val_accuracy.csv"), {
                "epoch": epoch,
                "val_top1": acc,
                "checkpoint": ckpt_path,
            })
            print(f"[epoch {epoch}] ImageNet val top1={acc:.4f}")

            if not args.skip_metrics:
                metric_dir = os.path.join(result_dir, "metrics")
                cache_dir = os.path.join(result_dir, "feature_cache")
                compute_metric_suite(
                    current_spec={
                        "label": f"finetuned_clip_epoch_{epoch:03d}",
                        "model_name": args.model_name,
                        "checkpoint_path": ckpt_path,
                        "pretrained": False,
                    },
                    reference_specs=[
                        {"label": "original_clip", "model_name": args.model_name, "pretrained": True},
                        {"label": "augreg_student", "model_name": args.reference_model, "pretrained": True},
                    ],
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
        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
