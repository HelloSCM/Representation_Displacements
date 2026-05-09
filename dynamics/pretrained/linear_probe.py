import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from data_utils import build_transform_for_model
from feature_utils import (
    build_feature_extractor_model,
    extract_last_layer_batch,
    get_feature_dim,
    load_checkpoint_into_model,
)
from utils import append_csv_row, save_checkpoint


class LinearProbeModel(nn.Module):
    def __init__(self, backbone: nn.Module, model_name: str, feat_dim: int, num_classes: int = 1000):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.head = nn.Linear(feat_dim, num_classes)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = extract_last_layer_batch(self.backbone, x, self.model_name)
        return self.head(feat.float())


def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().sum().item()


@torch.no_grad()
def evaluate_probe(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0.0
    total = 0
    for images, target in tqdm(loader, desc="probe val", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        correct += accuracy_top1(logits, target)
        total += target.numel()
    return correct / max(total, 1)


def train_linear_probe(
    model_name: str,
    checkpoint_path: Optional[str],
    train_dir: str,
    val_dir: str,
    output_dir: str,
    epoch_id: int,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
    probe_epochs: int = 1,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    max_train_samples: int = -1,
    max_val_samples: int = -1,
    num_classes: int = 1000,
    log_csv: Optional[str] = None,
):
    """Train a fresh linear probe on a frozen checkpoint and return val accuracy."""
    os.makedirs(output_dir, exist_ok=True)

    train_tf, _ = build_transform_for_model(model_name, is_training=True)
    val_tf, _ = build_transform_for_model(model_name, is_training=False)
    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=val_tf)
    if max_train_samples is not None and max_train_samples > 0:
        train_ds = Subset(train_ds, list(range(min(max_train_samples, len(train_ds)))))
    if max_val_samples is not None and max_val_samples > 0:
        val_ds = Subset(val_ds, list(range(min(max_val_samples, len(val_ds)))))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    backbone, _, feat_dim = build_feature_extractor_model(
        model_name=model_name,
        pretrained=checkpoint_path is None,
        checkpoint_path=checkpoint_path,
        all_layers=False,
        num_classes=0,
        device=device,
        strict_load=False,
    )
    probe = LinearProbeModel(backbone, model_name, feat_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(probe.head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for probe_epoch in range(1, probe_epochs + 1):
        probe.train()
        total_loss = 0.0
        total = 0
        for images, target in tqdm(train_loader, desc=f"probe train {probe_epoch}/{probe_epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = probe(images)
            loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.numel()
            total += target.numel()

    acc = evaluate_probe(probe, val_loader, device)
    ckpt_path = os.path.join(output_dir, f"linear_probe_epoch_{epoch_id:03d}.pth")
    save_checkpoint(ckpt_path, probe.head, optimizer=optimizer, epoch=probe_epochs)

    if log_csv is not None:
        append_csv_row(log_csv, {
            "epoch": epoch_id,
            "probe_epochs": probe_epochs,
            "val_top1": acc,
            "checkpoint": ckpt_path,
        })
    return acc


@torch.no_grad()
def evaluate_timm_classifier(
    model_name: str,
    checkpoint_path: str,
    val_dir: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
    num_classes: int = 1000,
    max_val_samples: int = -1,
):
    transform, _ = build_transform_for_model(model_name, is_training=False)
    val_ds = ImageFolder(val_dir, transform=transform)
    if max_val_samples is not None and max_val_samples > 0:
        val_ds = Subset(val_ds, list(range(min(max_val_samples, len(val_ds)))))
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    load_checkpoint_into_model(model, checkpoint_path, device=torch.device("cpu"), strict=False)
    model = model.to(device).eval()

    correct = 0.0
    total = 0
    for images, target in tqdm(loader, desc="val classifier", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        correct += accuracy_top1(logits, target)
        total += target.numel()
    return correct / max(total, 1)
