import torch


def compute_pca_projection(x: torch.Tensor, n_components: int, eps: float = 1e-5):
    feature_dim = x.shape[1]
    if n_components >= feature_dim:
        raise ValueError(
            f"pca_components must be smaller than feature dimension; got {n_components} >= {feature_dim}"
        )
    mu = torch.mean(x, dim=0)
    x_centered = x - mu
    cov = torch.mm(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
    u, s, _ = torch.linalg.svd(cov)
    u_k = u[:, :n_components]
    s_k = s[:n_components]
    w_pca = u_k @ torch.diag(1.0 / torch.sqrt(s_k + eps))
    return mu, w_pca


def apply_pca(features_dict: dict[str, torch.Tensor], mu: torch.Tensor, w_pca: torch.Tensor):
    out: dict[str, torch.Tensor] = {}
    for k, feat in features_dict.items():
        centered = feat.to(mu.device) - mu
        out[k] = torch.matmul(centered.unsqueeze(0), w_pca).squeeze(0)
    return out
