from collections import defaultdict
import torch
import torch.nn.functional as F


def process_metrics(class_features: dict[str, torch.Tensor], parent_fn):
    parent_to_children = defaultdict(list)
    for item in class_features.keys():
        parent_to_children[parent_fn(item)].append(item)

    selected = []
    unselected = []
    for _, children in parent_to_children.items():
        if len(children) >= 3:
            selected.extend(children)
        else:
            unselected.extend(children)

    if not selected:
        return None, []

    parent_features = {
        p: torch.stack([class_features[c] for c in children]).mean(dim=0)
        for p, children in parent_to_children.items()
    }

    u_matrix = torch.stack([class_features[u] for u in unselected]) if unselected else None
    acc = defaultdict(list)
    raw = []

    for c_id in selected:
        parent_id = parent_fn(c_id)
        siblings = [x for x in parent_to_children[parent_id] if x != c_id]
        c = class_features[c_id]
        p = parent_features[parent_id]
        s_matrix = torch.stack([class_features[s] for s in siblings])

        sim = F.cosine_similarity(c.unsqueeze(0), p.unsqueeze(0)).item()
        acc["child_parent"].append(sim)
        raw.append({"metric": "child_parent", "item": c_id, "target": parent_id, "cosine_similarity": sim})

        b = F.cosine_similarity(c.unsqueeze(0), s_matrix)
        acc["child_brothers_mean"].append(b.mean().item())
        acc["child_brothers_std"].append(b.std(unbiased=False).item() if len(b) > 1 else 0.0)

        if u_matrix is not None:
            o = F.cosine_similarity(c.unsqueeze(0), u_matrix)
            acc["child_others_mean"].append(o.mean().item())
            acc["child_others_std"].append(o.std(unbiased=False).item() if len(o) > 1 else 0.0)

        c_minus_p = c - p
        d = F.cosine_similarity(c_minus_p.unsqueeze(0), p.unsqueeze(0)).item()
        acc["child-parent_parent"].append(d)

        s_minus_p = s_matrix - p
        e = F.cosine_similarity(c_minus_p.unsqueeze(0), s_minus_p)
        acc["child-parent_brothers-parent_mean"].append(e.mean().item())
        acc["child-parent_brothers-parent_std"].append(e.std(unbiased=False).item() if len(e) > 1 else 0.0)

        c_minus_s = c.unsqueeze(0) - s_matrix
        f = F.cosine_similarity(c_minus_s, p.unsqueeze(0))
        acc["child-brothers_parent_mean"].append(f.mean().item())
        acc["child-brothers_parent_std"].append(f.std(unbiased=False).item() if len(f) > 1 else 0.0)

    final_metrics = {k: float(torch.tensor(v).mean().item()) for k, v in acc.items() if v}
    return final_metrics, raw
