import os
from typing import Optional, Tuple

import timm
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, Subset
from torchvision.datasets import ImageFolder
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


DEFAULT_TRAIN_DIR = "./data/train"
DEFAULT_VAL_DIR = "./data/val"


class DualTransformImageFolder(ImageFolder):
    """ImageFolder returning two differently transformed views of the same image.

    This is useful for teacher/student distillation when each timm model has its own
    pretrained input normalization/crop configuration.
    """

    def __init__(self, root: str, transform_a=None, transform_b=None):
        super().__init__(root=root, transform=None)
        self.transform_a = transform_a
        self.transform_b = transform_b

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path)
        img_a = self.transform_a(img) if self.transform_a is not None else img
        img_b = self.transform_b(img) if self.transform_b is not None else img
        return img_a, img_b, target


def _make_temp_model(model_name: str):
    # pretrained=False still exposes timm's default/pretrained cfg for transforms.
    return timm.create_model(model_name, pretrained=False)


def build_transform_for_model(
    model_name: str,
    is_training: bool,
    model: Optional[torch.nn.Module] = None,
):
    """Build the timm preprocessing transform corresponding to a model config."""
    temp_model = None
    if model is None:
        temp_model = _make_temp_model(model_name)
        model = temp_model
    cfg_source = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
    data_config = resolve_data_config(cfg_source, model=model)
    transform = create_transform(**data_config, is_training=is_training)
    if temp_model is not None:
        del temp_model
    return transform, data_config


def make_imagefolder(
    root: str,
    model_name: str,
    is_training: bool,
    max_samples: int = -1,
) -> ImageFolder:
    transform, _ = build_transform_for_model(model_name, is_training=is_training)
    ds = ImageFolder(root=root, transform=transform)
    if max_samples is not None and max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return ds


def make_dual_imagefolder(
    root: str,
    model_a_name: str,
    model_b_name: str,
    is_training: bool,
    max_samples: int = -1,
) -> DualTransformImageFolder:
    transform_a, _ = build_transform_for_model(model_a_name, is_training=is_training)
    transform_b, _ = build_transform_for_model(model_b_name, is_training=is_training)
    ds = DualTransformImageFolder(root=root, transform_a=transform_a, transform_b=transform_b)
    if max_samples is not None and max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return ds


def create_loader(
    dataset,
    batch_size: int,
    is_training: bool,
    distributed: bool = False,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: Optional[bool] = None,
):
    if drop_last is None:
        drop_last = is_training
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=is_training, drop_last=drop_last)
        shuffle = False
    else:
        sampler = None
        shuffle = is_training
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def imagenet_num_classes() -> int:
    return 1000
