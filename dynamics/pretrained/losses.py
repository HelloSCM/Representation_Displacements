from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from feature_utils import extract_last_layer_batch
from utils import is_dist_avail_and_initialized, is_main_process


def feature_mse_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_feat, teacher_feat.detach())


def pairwise_cosine_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, dim=-1, eps=eps)
    return x @ x.t()


def relation_kl_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    temperature: float = 0.1,
    mask_diagonal: bool = True,
) -> torch.Tensor:
    """KL between teacher and student batch-wise cosine-similarity distributions.

    Robust implementation for AMP/DDP training:
    - compute the relation logits in fp32, even under autocast;
    - remove diagonal entries before softmax instead of filling them with a large
      negative value. Filling with -1e4 and then dividing by a small temperature
      can become -inf in fp16, and KLDivLoss may produce NaN on 0 * inf terms;
    - if a local batch has fewer than 2 samples, return a differentiable zero.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Force fp32 for numerical stability. This is especially important when the
    # surrounding training loop uses torch.cuda.amp.autocast.
    student_feat_f = student_feat.float()
    teacher_feat_f = teacher_feat.detach().float()

    sim_s = pairwise_cosine_matrix(student_feat_f)
    sim_t = pairwise_cosine_matrix(teacher_feat_f)

    bsz = sim_s.shape[0]
    if mask_diagonal:
        if bsz < 2:
            return student_feat_f.sum() * 0.0
        offdiag = ~torch.eye(bsz, dtype=torch.bool, device=sim_s.device)
        sim_s = sim_s[offdiag].view(bsz, bsz - 1)
        sim_t = sim_t[offdiag].view(bsz, bsz - 1)

    log_p_s = F.log_softmax(sim_s / temperature, dim=-1)
    p_t = F.softmax(sim_t / temperature, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean")# * (temperature ** 2)


def fit_whitening_matrix(features: torch.Tensor, eps: float = 1e-5):
    """Fit ZCA-style whitening: (x - mean) @ W."""
    x = features.float()
    mean = x.mean(dim=0)
    xc = x - mean
    denom = max(xc.shape[0] - 1, 1)
    cov = (xc.t() @ xc) / denom
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = torch.rsqrt(eigvals)
    W = eigvecs @ torch.diag(inv_sqrt) @ eigvecs.t()
    return mean, W


class LearnableWhitening(nn.Module):
    """Fixed mean plus learnable whitening/projection matrix."""

    def __init__(self, mean: torch.Tensor, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone().float())
        self.weight = nn.Parameter(weight.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(dtype=x.dtype, device=x.device)) @ self.weight.to(dtype=x.dtype, device=x.device)


def compute_opa_teacher_to_student(teacher_white: torch.Tensor, student_white: torch.Tensor) -> torch.Tensor:
    """Return R such that teacher_white @ R is closest to student_white."""
    M = teacher_white.t() @ student_white
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    return R


@torch.no_grad()
def fit_parallel_distillation_state(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    val_loader,
    teacher_model_name: str,
    student_model_name: str,
    device: torch.device,
    max_samples: int = -1,
    eps: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """Precompute fixed teacher whitening, initial student whitening, and OPA.

    The val_loader must return (student_image, teacher_image, target). Student images
    are at index 0; teacher images are at index 1.
    """
    teacher_feats = []
    student_feats = []
    seen = 0
    teacher_model.eval()
    student_model.eval()

    for batch in tqdm(val_loader, desc="fit parallel transforms", leave=False):
        student_img = batch[0].to(device, non_blocking=True)
        teacher_img = batch[1].to(device, non_blocking=True)
        t_feat = extract_last_layer_batch(teacher_model, teacher_img, teacher_model_name).float()
        s_feat = extract_last_layer_batch(student_model, student_img, student_model_name).float()
        teacher_feats.append(t_feat.cpu())
        student_feats.append(s_feat.cpu())
        seen += t_feat.shape[0]
        if max_samples is not None and max_samples > 0 and seen >= max_samples:
            break

    teacher_feats = torch.cat(teacher_feats, dim=0)
    student_feats = torch.cat(student_feats, dim=0)
    if max_samples is not None and max_samples > 0:
        teacher_feats = teacher_feats[:max_samples]
        student_feats = student_feats[:max_samples]

    t_mean, t_W = fit_whitening_matrix(teacher_feats, eps=eps)
    s_mean, s_W = fit_whitening_matrix(student_feats, eps=eps)

    t_white = (teacher_feats - t_mean) @ t_W
    s_white = (student_feats - s_mean) @ s_W
    R = compute_opa_teacher_to_student(t_white, s_white)

    return {
        "teacher_mean": t_mean,
        "teacher_W": t_W,
        "student_mean": s_mean,
        "student_W": s_W,
        "teacher_to_student_R": R,
    }


def broadcast_parallel_state(state: Optional[Dict[str, torch.Tensor]], dim: int, device: torch.device):
    """Broadcast precomputed parallel-distillation tensors from rank 0."""
    keys = ["teacher_mean", "teacher_W", "student_mean", "student_W", "teacher_to_student_R"]
    shapes = {
        "teacher_mean": (dim,),
        "student_mean": (dim,),
        "teacher_W": (dim, dim),
        "student_W": (dim, dim),
        "teacher_to_student_R": (dim, dim),
    }
    if not is_dist_avail_and_initialized():
        return {k: v.to(device) for k, v in state.items()}

    out = {}
    for k in keys:
        if is_main_process():
            tensor = state[k].to(device)
        else:
            tensor = torch.empty(shapes[k], device=device, dtype=torch.float32)
        dist.broadcast(tensor, src=0)
        out[k] = tensor
    return out


def apply_fixed_teacher_transform(
    teacher_feat: torch.Tensor,
    state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    mean = state["teacher_mean"].to(device=teacher_feat.device, dtype=teacher_feat.dtype)
    W = state["teacher_W"].to(device=teacher_feat.device, dtype=teacher_feat.dtype)
    R = state["teacher_to_student_R"].to(device=teacher_feat.device, dtype=teacher_feat.dtype)
    return ((teacher_feat - mean) @ W) @ R


def parallel_distillation_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    parallel_state: Dict[str, torch.Tensor],
    student_whitener: LearnableWhitening,
    mask_diagonal: bool = True,
) -> torch.Tensor:
    """1 - mean cosine between corresponding pairwise difference vectors."""
    with torch.no_grad():
        teacher_z = apply_fixed_teacher_transform(teacher_feat.detach(), parallel_state)
    student_z = student_whitener(student_feat)

    ds = student_z[:, None, :] - student_z[None, :, :]
    dt = teacher_z[:, None, :] - teacher_z[None, :, :]
    cos = F.cosine_similarity(ds, dt, dim=-1, eps=1e-8)

    if mask_diagonal:
        bsz = cos.shape[0]
        mask = ~torch.eye(bsz, dtype=torch.bool, device=cos.device)
        cos = cos[mask]

    return 1.0 - cos.mean()
