import numpy as np
import torch
from sklearn.decomposition import PCA


def pca_whiten_fit_transform(x: np.ndarray, proj_dim: int, seed: int):
    if proj_dim >= x.shape[1]:
        raise ValueError(f"proj_dim must be smaller than feature dimension; got {proj_dim} >= {x.shape[1]}")
    pca = PCA(n_components=proj_dim, random_state=seed, whiten=True)
    x_proj = pca.fit_transform(x)
    scale = np.sqrt(pca.explained_variance_)
    mu_proj = np.dot(pca.mean_, pca.components_.T) / scale
    return x_proj, mu_proj


@torch.no_grad()
def opa_matrix(a_train: torch.Tensor, b_train: torch.Tensor) -> torch.Tensor:
    m = torch.matmul(b_train.T, a_train)
    u, _, vh = torch.linalg.svd(m)
    return torch.matmul(u, vh)


@torch.no_grad()
def metrics_with_fixed_rotation(
    a_val_centered: torch.Tensor,
    a_mu_proj: torch.Tensor,
    b_val_centered: torch.Tensor,
    b_mu_proj: torch.Tensor,
    rotation: torch.Tensor,
):
    b_final = torch.matmul(b_val_centered, rotation) + b_mu_proj
    a_final = a_val_centered + a_mu_proj

    def cos_sim(x, y):
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

    direct = cos_sim(a_final, b_final).mean().item()

    n = a_final.shape[0]
    idx = torch.arange(n, device=a_final.device)
    idx2 = torch.roll(idx, shifts=1)
    mod = cos_sim(a_final - a_final[idx2], b_final - b_final[idx2]).mean().item()
    obj = cos_sim(a_final - b_final, a_final[idx2] - b_final[idx2]).mean().item()
    return {"cross_modality": mod, "cross_object": obj, "direct_similarity": direct}
