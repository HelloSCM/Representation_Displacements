import csv
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize torch.distributed when launched with torchrun.

    Returns:
        distributed, rank, local_rank, world_size, device
    """
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
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
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common prefixes introduced by DDP / torch.compile."""
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def load_state_dict_from_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return strip_state_dict_prefix(ckpt[key])
    if isinstance(ckpt, dict):
        return strip_state_dict_prefix(ckpt)
    raise ValueError(f"Cannot parse checkpoint: {path}")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    args: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "model": unwrap_model(model).state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if args is not None:
        try:
            payload["args"] = vars(args)
        except TypeError:
            payload["args"] = str(args)
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if get_world_size() < 2:
        return tensor
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )
