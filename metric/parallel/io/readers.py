from pathlib import Path
import re
import numpy as np
import torch


VISION_PATTERN = re.compile(r"imagenet22k-(.+)-layer(\d+)-(\d+)\.pt$")
LANG_PATTERN = re.compile(r"common_words_79k-(.+)-layer(\d+)-(\d+)\.pt$")
SAMPLE_PATTERN = re.compile(r"flickr30k-(.+)-layer(\d+)-(\d+)\.npy$")


def _parse(name: str, pattern):
    m = pattern.search(name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_layered_pt(directory: Path, pattern, model: str):
    out = {}
    for p in directory.glob("*.pt"):
        info = _parse(p.name, pattern)
        if info and info[0] == model:
            _, layer = info
            out[layer] = torch.load(p, map_location="cpu")
    if not out:
        raise ValueError(f"no files found for model: {model}")
    return out


def load_layered_npy(directory: Path, model: str):
    out = {}
    for p in directory.glob("*.npy"):
        info = _parse(p.name, SAMPLE_PATTERN)
        if info and info[0] == model:
            _, layer = info
            out[layer] = np.load(p)
    if not out:
        raise ValueError(f"no flickr30k files found for model: {model}")
    return out
