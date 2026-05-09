from pathlib import Path
import h5py
import torch


def read_vision_features_imagenet22k(input_root: Path, class_feat_root: Path, model: str, device: torch.device):
    """Read ImageNet22k vision features for one model."""
    class_h5 = class_feat_root / f"{model}.h5"
    vision_h5 = input_root / f"{model}.h5"
    if not class_h5.exists():
        raise FileNotFoundError(f"ImageNet22k class feature file not found: {class_h5}")
    if not vision_h5.exists():
        raise FileNotFoundError(f"ImageNet22k vision feature file not found: {vision_h5}")

    class_features = {}
    with h5py.File(class_h5, "r") as f:
        for wnid in f.keys():
            feat = torch.tensor(f[wnid][...], dtype=torch.float32, device=device)
            class_features[wnid] = feat
    return class_features


def read_language_features_commonwords79k(input_root: Path, model: str, device: torch.device):
    pt_path = input_root / f"{model}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"CommonWords79k feature file not found: {pt_path}")
    features = torch.load(pt_path, map_location="cpu")
    return {k: v.to(device=device, dtype=torch.float32) for k, v in features.items()}
