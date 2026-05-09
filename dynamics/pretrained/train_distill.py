import argparse
import os
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data_utils import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_VAL_DIR,
    create_loader,
    make_dual_imagefolder,
)
from feature_utils import (
    build_feature_extractor_model,
    extract_last_layer_batch,
    freeze_unused_tail_for_block_feature_training,
)
from linear_probe import train_linear_probe
from losses import (
    LearnableWhitening,
    broadcast_parallel_state,
    feature_mse_loss,
    fit_parallel_distillation_state,
    parallel_distillation_loss,
    relation_kl_loss,
)
from metrics import compute_metric_suite
from utils import (
    append_csv_row,
    barrier,
    cleanup_distributed,
    is_main_process,
    reduce_mean,
    safe_name,
    save_checkpoint,
    save_json,
    seed_everything,
    setup_distributed,
)


TEACHER_MODEL = "vit_base_patch16_clip_224.laion2b"
STUDENT_MODEL = "vit_base_patch16_224.augreg_in1k"
DEFAULT_PROJECT_DIR = "."
DEFAULT_CKPT_DIR = "./checkpoints"
DEFAULT_IMAGENETTE_DIR = "./imagenette"
DEFAULT_IMAGEWOOF_DIR = "./imagewoof"


def parse_args():
    parser = argparse.ArgumentParser("Training-dynamics distillation for ViT representations")
    parser.add_argument("--loss_type", type=str, required=True, choices=["feature", "relation", "parallel"])
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output_dir", type=str, default=os.path.join(DEFAULT_PROJECT_DIR, "results"))
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128, help="per-GPU batch size under torchrun")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1, help="relation distillation temperature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="enable CUDA AMP")

    # Parallel distillation setup
    parser.add_argument("--parallel_fit_samples", type=int, default=-1, help="-1 means use all val samples")
    parser.add_argument("--whiten_eps", type=float, default=1e-5)

    # Metrics after each epoch
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--metric_batch_size", type=int, default=64)
    parser.add_argument("--metric_proj_dim", type=int, default=-1, help="-1 disables PCA; >0 enables PCA/whiten")
    parser.add_argument("--metric_pair_sample_size", type=int, default=1000)
    parser.add_argument("--metric_max_samples", type=int, default=-1, help="-1 means all ImageNet val samples")

    # Linear probe after each epoch
    parser.add_argument("--skip_linear_probe", action="store_true")
    parser.add_argument(
        "--probe_datasets",
        nargs="+",
        default=["imagenette", "imagewoof"],
        choices=["imagenette", "imagewoof"],
        help="Small datasets used for per-epoch distillation linear probing.",
    )
    parser.add_argument("--imagenette_dir", type=str, default=DEFAULT_IMAGENETTE_DIR)
    parser.add_argument("--imagewoof_dir", type=str, default=DEFAULT_IMAGEWOOF_DIR)
    parser.add_argument("--probe_epochs", type=int, default=1)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--probe_lr", type=float, default=0.1)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--probe_max_train_samples", type=int, default=-1)
    parser.add_argument("--probe_max_val_samples", type=int, default=-1)
    return parser.parse_args()


def build_experiment_dirs(args):
    exp_name = args.loss_type
    result_dir = os.path.join(args.output_dir, "distill", exp_name)
    ckpt_dir = os.path.join(args.checkpoint_dir, "distill", exp_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return result_dir, ckpt_dir


def get_probe_dataset_specs(args):
    """Return small 10-class linear-probe datasets requested for distillation."""
    roots = {
        "imagenette": args.imagenette_dir,
        "imagewoof": args.imagewoof_dir,
    }
    specs = []
    for name in args.probe_datasets:
        root = roots[name]
        specs.append({
            "name": name,
            "train_dir": os.path.join(root, "train"),
            "val_dir": os.path.join(root, "val"),
            "num_classes": 10,
        })
    return specs


def train_one_epoch(
    epoch,
    args,
    student_model,
    teacher_model,
    loader,
    sampler,
    optimizer,
    scaler,
    device,
    parallel_state=None,
    student_whitener=None,
):
    if sampler is not None:
        sampler.set_epoch(epoch)

    student_model.train()
    teacher_model.eval()
    if student_whitener is not None:
        student_whitener.train()

    total_loss = 0.0
    total_batches = 0
    iterator = tqdm(loader, desc=f"distill epoch {epoch}", disable=not is_main_process())
    for batch in iterator:
        student_img = batch[0].to(device, non_blocking=True)
        teacher_img = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_feat = extract_last_layer_batch(teacher_model, teacher_img, args.teacher_model)

        with autocast(enabled=args.amp):
            student_feat = extract_last_layer_batch(student_model, student_img, args.student_model)
            if args.loss_type == "feature":
                loss = feature_mse_loss(student_feat, teacher_feat)
            elif args.loss_type == "relation":
                loss = relation_kl_loss(
                    student_feat,
                    teacher_feat,
                    temperature=args.temperature,
                    mask_diagonal=True,
                )
            elif args.loss_type == "parallel":
                loss = parallel_distillation_loss(
                    student_feat,
                    teacher_feat,
                    parallel_state=parallel_state,
                    student_whitener=student_whitener,
                    mask_diagonal=True,
                )
            else:
                raise ValueError(args.loss_type)

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

    # Train data: student transform at index 0, teacher transform at index 1.
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

    teacher_model, _, _ = build_feature_extractor_model(
        args.teacher_model,
        pretrained=True,
        checkpoint_path=None,
        all_layers=False,
        num_classes=0,
        device=device,
    )
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.eval()

    student_model, _, feat_dim = build_feature_extractor_model(
        args.student_model,
        pretrained=True,
        checkpoint_path=None,
        all_layers=False,
        num_classes=0,
        device=device,
    )
    frozen_tail = freeze_unused_tail_for_block_feature_training(student_model)
    if is_main_process() and frozen_tail:
        print(
            "Frozen downstream parameters not used by block-feature distillation: "
            + ", ".join(frozen_tail)
        )

    parallel_state = None
    student_whitener = None
    if args.loss_type == "parallel":
        if is_main_process():
            print("Fitting teacher/student whitening matrices and teacher->student OPA on val set...")
        # Eval dual loader: student image index 0, teacher image index 1.
        val_ds = make_dual_imagefolder(
            args.val_dir,
            model_a_name=args.student_model,
            model_b_name=args.teacher_model,
            is_training=False,
            max_samples=args.parallel_fit_samples,
        )
        # This loader is deliberately not distributed; rank 0 fits once, then broadcasts.
        val_loader, _ = create_loader(
            val_ds,
            batch_size=args.metric_batch_size,
            is_training=False,
            distributed=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        state = None
        if is_main_process():
            state = fit_parallel_distillation_state(
                teacher_model=teacher_model,
                student_model=student_model,
                val_loader=val_loader,
                teacher_model_name=args.teacher_model,
                student_model_name=args.student_model,
                device=device,
                max_samples=args.parallel_fit_samples,
                eps=args.whiten_eps,
            )
        parallel_state = broadcast_parallel_state(state, feat_dim, device)
        student_whitener = LearnableWhitening(
            parallel_state["student_mean"],
            parallel_state["student_W"],
        ).to(device)
        barrier()

    if distributed:
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)

    params = [p for p in student_model.parameters() if p.requires_grad]
    if student_whitener is not None:
        params += [p for p in student_whitener.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    if is_main_process() and not args.skip_metrics:
        metric_dir = os.path.join(result_dir, "metrics")
        cache_dir = os.path.join(result_dir, "feature_cache")
        compute_metric_suite(
            current_spec={
                "label": "student_epoch_000",
                "model_name": args.student_model,
                "pretrained": True,
            },
            reference_specs=[
                {"label": "teacher_clip", "model_name": args.teacher_model, "pretrained": True},
                {"label": "initial_student_augreg", "model_name": args.student_model, "pretrained": True},
            ],
            val_dir=args.val_dir,
            out_dir=metric_dir,
            epoch=0,
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

    train_log = os.path.join(result_dir, "train_loss.csv")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = train_one_epoch(
            epoch=epoch,
            args=args,
            student_model=student_model,
            teacher_model=teacher_model,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            parallel_state=parallel_state,
            student_whitener=student_whitener,
        )
        if is_main_process():
            extra = {}
            if student_whitener is not None:
                extra["student_whitener"] = student_whitener.state_dict()
                extra["parallel_state"] = {k: v.detach().cpu() for k, v in parallel_state.items()}
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            save_checkpoint(ckpt_path, student_model, optimizer=optimizer, epoch=epoch, args=args, extra=extra)
            append_csv_row(train_log, {
                "epoch": epoch,
                "loss": avg_loss,
                "seconds": time.time() - start,
                "checkpoint": ckpt_path,
            })
            print(f"[epoch {epoch}] loss={avg_loss:.6f} saved={ckpt_path}")
        barrier()

        if is_main_process():
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            if not args.skip_metrics:
                metric_dir = os.path.join(result_dir, "metrics")
                cache_dir = os.path.join(result_dir, "feature_cache")
                compute_metric_suite(
                    current_spec={
                        "label": f"student_epoch_{epoch:03d}",
                        "model_name": args.student_model,
                        "checkpoint_path": ckpt_path,
                        "pretrained": False,
                    },
                    reference_specs=[
                        {"label": "teacher_clip", "model_name": args.teacher_model, "pretrained": True},
                        {"label": "initial_student_augreg", "model_name": args.student_model, "pretrained": True},
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
            if not args.skip_linear_probe:
                for probe_spec in get_probe_dataset_specs(args):
                    dataset_name = probe_spec["name"]
                    probe_dir = os.path.join(
                        result_dir,
                        "linear_probe",
                        dataset_name,
                        f"epoch_{epoch:03d}",
                    )
                    acc = train_linear_probe(
                        model_name=args.student_model,
                        checkpoint_path=ckpt_path,
                        train_dir=probe_spec["train_dir"],
                        val_dir=probe_spec["val_dir"],
                        output_dir=probe_dir,
                        epoch_id=epoch,
                        device=device,
                        batch_size=args.probe_batch_size,
                        num_workers=args.num_workers,
                        probe_epochs=args.probe_epochs,
                        lr=args.probe_lr,
                        weight_decay=args.probe_weight_decay,
                        max_train_samples=args.probe_max_train_samples,
                        max_val_samples=args.probe_max_val_samples,
                        num_classes=probe_spec["num_classes"],
                        log_csv=None,
                    )
                    append_csv_row(os.path.join(result_dir, "linear_probe_accuracy.csv"), {
                        "epoch": epoch,
                        "dataset": dataset_name,
                        "num_classes": probe_spec["num_classes"],
                        "probe_epochs": args.probe_epochs,
                        "val_top1": acc,
                        "train_dir": probe_spec["train_dir"],
                        "val_dir": probe_spec["val_dir"],
                        "checkpoint": os.path.join(probe_dir, f"linear_probe_epoch_{epoch:03d}.pth"),
                    })
                    print(f"[epoch {epoch}] {dataset_name} linear probe top1={acc:.4f}")
        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
