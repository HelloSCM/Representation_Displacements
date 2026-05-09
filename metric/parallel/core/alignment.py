import numpy as np
import torch
import nltk
from nltk.corpus import wordnet as wn


try:
    wn.ensure_loaded()
except Exception:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_concept_intersection(vision_dict: dict, language_dict: dict):
    v_keys = list(vision_dict.keys())
    l_keys = set(language_dict.keys())
    v_out, l_out = [], []
    for v_id in v_keys:
        try:
            offset = int(v_id[1:])
            synset = wn.synset_from_pos_and_offset("n", offset)
            lemmas = [l.name().lower().replace("_", " ") for l in synset.lemmas()]
            for lemma in lemmas:
                if lemma in l_keys:
                    v_out.append(v_id)
                    l_out.append(lemma)
                    break
        except Exception:
            continue
    if not v_out:
        raise ValueError("no overlap found between ImageNet22k concepts and CommonWords79k")
    return v_out, l_out


def split_train_val(total: int, train_size: int, seed: int):
    if train_size >= total:
        raise ValueError(f"train_size must be smaller than total samples; got {train_size} >= {total}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    return train_idx, val_idx
