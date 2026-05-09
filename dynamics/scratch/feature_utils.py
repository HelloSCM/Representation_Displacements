import gc
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from data_utils import build_transform_for_model
from utils import load_state_dict_from_checkpoint


class ViTBlockFeatureExtractor(torch.nn.Module):
    """Forward-hook based ViT block feature extractor.

    Why hooks instead of torchvision.create_feature_extractor?
    -------------------------------------------------------
    Newer timm ViT blocks call PyTorch scaled_dot_product_attention with an
    ``is_causal`` boolean. torchvision's FX tracer can turn that value into a
    Proxy object, which raises:
        TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Proxy

    This wrapper avoids FX tracing completely. It registers forward hooks on
    ``model.blocks[i]`` and returns the block outputs. For standard timm ViT
    blocks, the block output is the final residual-stream tensor of that block,
    corresponding to the FX node typically named ``blocks.i.add_1``.
    """

    def __init__(self, base_model: torch.nn.Module, all_layers: bool = True):
        super().__init__()
        if not hasattr(base_model, "blocks"):
            raise ValueError("Only timm ViT-like models with `model.blocks` are supported.")
        self.base_model = base_model
        self.all_layers = all_layers
        self.num_blocks = len(base_model.blocks)
        self.layer_indices = list(range(self.num_blocks)) if all_layers else [self.num_blocks - 1]
        self._features: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.layer_indices:
            block = self.base_model.blocks[layer_idx]

            def hook(_module, _inputs, output, layer_idx=layer_idx):
                # output is usually [B, tokens, dim]. Preserve gradient for training losses.
                if isinstance(output, (tuple, list)):
                    output = output[0]
                self._features[f"layer{layer_idx}"] = output

            self._hooks.append(block.register_forward_hook(hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        self._features = OrderedDict()
        _ = self.base_model(x)
        return self._features

    def train(self, mode: bool = True):
        self.base_model.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)


def make_timm_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: Optional[int] = None,
) -> torch.nn.Module:
    kwargs = {}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    return model


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Make checkpoints saved from DDP/wrappers easier to load into raw timm models."""
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "base_model.", "module.base_model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    device: torch.device,
    strict: bool = False,
) -> torch.nn.Module:
    if checkpoint_path is None:
        return model
    state = load_state_dict_from_checkpoint(checkpoint_path, map_location="cpu")
    state = _strip_known_prefixes(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if len(unexpected) > 0:
        print(f"[load_checkpoint] Unexpected keys ignored: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[load_checkpoint] Missing keys ignored: {len(missing)}")
    return model


def get_num_blocks(model: torch.nn.Module) -> int:
    base = model.base_model if isinstance(model, ViTBlockFeatureExtractor) else model
    if not hasattr(base, "blocks"):
        raise ValueError("Only timm ViT-like models with `model.blocks` are supported.")
    return len(base.blocks)


def get_feature_dim(model: torch.nn.Module) -> int:
    base = model.base_model if isinstance(model, ViTBlockFeatureExtractor) else model
    if hasattr(base, "num_features"):
        return int(base.num_features)
    if hasattr(base, "embed_dim"):
        return int(base.embed_dim)
    raise ValueError("Cannot infer feature dimension from model.")


def make_return_nodes(model: torch.nn.Module, all_layers: bool = True) -> Dict[str, str]:
    """Compatibility helper retained for old imports/logging.

    The new implementation uses hooks on blocks directly instead of FX node names.
    """
    num_blocks = get_num_blocks(model)
    if all_layers:
        return {f"blocks.{i}": f"layer{i}" for i in range(num_blocks)}
    return {f"blocks.{num_blocks - 1}": "last"}


def build_feature_extractor_model(
    model_name: str,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    all_layers: bool = True,
    num_classes: Optional[int] = 0,
    device: Optional[torch.device] = None,
    strict_load: bool = False,
):
    """Create a timm ViT and wrap it with hook-based block feature extraction.

    Returned model.forward(images) -> OrderedDict[str, Tensor].
    For all_layers=True, keys are layer0..layer{L-1}; for all_layers=False,
    only layer{L-1} is returned.
    """
    use_pretrained = pretrained and checkpoint_path is None
    base = make_timm_model(model_name, pretrained=use_pretrained, num_classes=num_classes)
    if checkpoint_path is not None:
        base = load_checkpoint_into_model(base, checkpoint_path, device=torch.device("cpu"), strict=strict_load)
    num_blocks = get_num_blocks(base)
    feat_dim = get_feature_dim(base)
    extractor = ViTBlockFeatureExtractor(base, all_layers=all_layers)
    if device is not None:
        extractor = extractor.to(device)
    return extractor, num_blocks, feat_dim


def pool_tokens(x: torch.Tensor, model_name: str) -> torch.Tensor:
    """Convert token-level activations [B, T, D] to image-level features [B, D]."""
    lower = model_name.lower()
    if "siglip" in lower or "ijepa" in lower:
        return x.mean(dim=1)
    return x[:, 0, :]


def outputs_to_layer_tensor(outputs: Dict[str, torch.Tensor], model_name: str) -> torch.Tensor:
    """Dict of layer activations -> [B, L, D]."""
    # OrderedDict insertion order follows ascending layer order from hooks.
    feats = [pool_tokens(v, model_name) for v in outputs.values()]
    return torch.stack(feats, dim=1)


def extract_last_layer_batch(model: torch.nn.Module, images: torch.Tensor, model_name: str) -> torch.Tensor:
    outputs = model(images)
    if isinstance(outputs, dict):
        tensor = list(outputs.values())[-1]
    elif isinstance(outputs, (list, tuple)):
        tensor = outputs[-1]
    else:
        tensor = outputs
    if tensor.ndim == 3:
        return pool_tokens(tensor, model_name)
    return tensor


@torch.no_grad()
def extract_features_to_numpy(
    model_name: str,
    data_dir: str,
    checkpoint_path: Optional[str] = None,
    pretrained: bool = True,
    batch_size: int = 64,
    num_workers: int = 8,
    device: Optional[torch.device] = None,
    max_samples: int = -1,
    all_layers: bool = True,
    desc: Optional[str] = None,
) -> np.ndarray:
    """Extract [N, L, D] features from all requested ViT blocks."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform, _ = build_transform_for_model(model_name, is_training=False)
    dataset = ImageFolder(root=data_dir, transform=transform)
    if max_samples is not None and max_samples > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    model, _, _ = build_feature_extractor_model(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        all_layers=all_layers,
        num_classes=0,
        device=device,
        strict_load=False,
    )
    model.eval()

    chunks: List[torch.Tensor] = []
    iterator = tqdm(loader, desc=desc or f"extract {model_name}", leave=False)
    for batch in iterator:
        images = batch[0].to(device, non_blocking=True)
        outputs = model(images)
        feats = outputs_to_layer_tensor(outputs, model_name)  # [B, L, D]
        chunks.append(feats.float().cpu())

    arr = torch.cat(chunks, dim=0).numpy()
    if isinstance(model, ViTBlockFeatureExtractor):
        model.remove_hooks()
    del model, chunks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return arr


@torch.no_grad()
def collect_last_layer_features_from_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    device: torch.device,
    input_index: int = 0,
    max_samples: int = -1,
    desc: str = "collect last-layer features",
) -> torch.Tensor:
    """Collect last-layer [N, D] features from a loader.

    For DualTransformImageFolder batches, input_index selects which image tensor to use.
    """
    model.eval()
    feats = []
    seen = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch[input_index].to(device, non_blocking=True)
        feat = extract_last_layer_batch(model, images, model_name)
        feats.append(feat.float().cpu())
        seen += feat.shape[0]
        if max_samples is not None and max_samples > 0 and seen >= max_samples:
            break
    out = torch.cat(feats, dim=0)
    if max_samples is not None and max_samples > 0:
        out = out[:max_samples]
    return out
