from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class LanguageGuider(nn.Module):
    """
    CLIP text connector + teacher relation distillation guider.

    Student:
        CLIP text feature -> residual MLP adapter -> language_fused

    Teacher:
        LLM2Vec or decoder-only text feature, used only for relation distillation.

    CommonWords79k features are used to initialize whitening transforms.

    Expected CLIP text feature:
        base  : [B, 512]
        large : [B, 768]

    Expected teacher feature:
        [B, teacher_dim]

    Outputs:
        text_z:
            L2-normalized CLIP-space text representation for image-text contrastive learning.

        relation_kd_loss:
            KL loss aligning pairwise similarity distributions between
            whitened student features and whitened teacher features.
    """

    ARCH_CONFIGS: Dict[str, Dict[str, Union[int, str]]] = {
        "base": {
            "clip_dim": 512,
            "commonwords_clip_file": "clip_base.safetensors",
        },
        "large": {
            "clip_dim": 768,
            "commonwords_clip_file": "clip_large.safetensors",
        },
    }

    def __init__(
        self,
        arch: str = "base",
        *,
        teacher_name: str = "llm2vec_qwen3_06b",
        commonwords_root: str = "./commonwords79k_feat",
        width: float = 2.0,
        dropout: float = 0.1,
        bias: bool = False,
        whitening_eps: float = 1e-5,
        whitening_chunk_size: int = 8192,
        whitening_max_samples: Optional[int] = None,
        whitening_device: str = "cpu",
        relation_temperature: float = 0.1,
        relation_weight: float = 0.05,
        exclude_self_similarity: bool = True,
        force_fp32: bool = True,
        normalize_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if arch not in self.ARCH_CONFIGS:
            raise ValueError(
                f"Unsupported arch={arch!r}. "
                f"Expected one of {list(self.ARCH_CONFIGS.keys())}."
            )

        self.arch = arch
        self.teacher_name = teacher_name
        self.clip_dim = int(self.ARCH_CONFIGS[arch]["clip_dim"])

        self.relation_temperature = relation_temperature
        self.relation_weight = relation_weight
        self.exclude_self_similarity = exclude_self_similarity
        self.force_fp32 = force_fp32
        self.normalize_eps = normalize_eps

        commonwords_root = Path(commonwords_root)

        clip_path = commonwords_root / str(self.ARCH_CONFIGS[arch]["commonwords_clip_file"])
        teacher_path = commonwords_root / f"{teacher_name}.safetensors"

        clip_commonwords = self._load_features(clip_path)
        teacher_commonwords = self._load_features(teacher_path)

        if clip_commonwords.ndim != 2:
            raise ValueError(f"CLIP CommonWords features must be 2D, got {clip_commonwords.shape}.")
        if teacher_commonwords.ndim != 2:
            raise ValueError(f"Teacher CommonWords features must be 2D, got {teacher_commonwords.shape}.")

        if clip_commonwords.shape[1] != self.clip_dim:
            raise ValueError(
                f"Expected CLIP dim {self.clip_dim} for arch={arch!r}, "
                f"but got CommonWords feature shape {tuple(clip_commonwords.shape)}."
            )

        if clip_commonwords.shape[0] != teacher_commonwords.shape[0]:
            raise ValueError(
                "CommonWords CLIP and teacher features should have the same number of samples. "
                f"Got {clip_commonwords.shape[0]} and {teacher_commonwords.shape[0]}."
            )

        self.teacher_dim = int(teacher_commonwords.shape[1])

        # ------------------------------------------------------------
        # Student MLP branch
        # ------------------------------------------------------------
        hidden_dim = int(round(width * self.teacher_dim))
        self.hidden_dim = hidden_dim

        self.mlp_ln = nn.LayerNorm(self.clip_dim)

        self.mlp_fc1 = nn.Linear(self.clip_dim, hidden_dim, bias=bias)
        self.mlp_fc2 = nn.Linear(hidden_dim, self.clip_dim, bias=bias)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Zero-initialized vector gate.
        # At init:
        #     language_fused == clip_text_feat
        self.residual_gate = nn.Parameter(torch.zeros(self.clip_dim))

        # ------------------------------------------------------------
        # Whitening initialization from CommonWords79k
        # ------------------------------------------------------------
        student_mean, student_whiten = self._compute_whitening(
            clip_commonwords,
            eps=whitening_eps,
            chunk_size=whitening_chunk_size,
            max_samples=whitening_max_samples,
            device=whitening_device,
        )

        teacher_mean, teacher_whiten = self._compute_whitening(
            teacher_commonwords,
            eps=whitening_eps,
            chunk_size=whitening_chunk_size,
            max_samples=whitening_max_samples,
            device=whitening_device,
        )

        # Student whitening matrix is trainable.
        self.register_buffer("student_whiten_mean", student_mean.float())
        self.student_whiten_matrix = nn.Parameter(student_whiten.float())

        # Teacher whitening matrix is frozen.
        self.register_buffer("teacher_whiten_mean", teacher_mean.float())
        self.register_buffer("teacher_whiten_matrix", teacher_whiten.float())

        # Keep this module in float32 by default.
        self.float()

    @staticmethod
    def _load_features(path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        data = load_file(str(path), device="cpu")

        if "features" not in data:
            raise KeyError(f"{path} does not contain tensor named 'features'.")

        return data["features"]

    @staticmethod
    @torch.no_grad()
    def _compute_whitening(
        features: torch.Tensor,
        *,
        eps: float = 1e-5,
        chunk_size: int = 8192,
        max_samples: Optional[int] = None,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ZCA whitening:

            x_white = (x - mean) @ W

        where:

            W = Q diag(1 / sqrt(lambda + eps)) Q^T

        The computation is chunked to avoid materializing a full float32 copy
        of large teacher features.
        """

        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got {tuple(features.shape)}.")

        n, dim = features.shape

        if max_samples is not None and n > max_samples:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(0)
            indices = torch.randperm(n, generator=generator)[:max_samples]
            indices = indices.sort().values
            features = features.index_select(0, indices)
            n = max_samples

        if n < 2:
            raise ValueError("Need at least 2 samples to compute whitening.")

        device_obj = torch.device(device)

        # First pass: mean.
        mean = torch.zeros(dim, dtype=torch.float32, device=device_obj)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = features[start:end].to(device=device_obj, dtype=torch.float32)
            mean += chunk.sum(dim=0)

        mean /= float(n)

        # Second pass: covariance.
        cov = torch.zeros(dim, dim, dtype=torch.float32, device=device_obj)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = features[start:end].to(device=device_obj, dtype=torch.float32)
            chunk = chunk - mean
            cov += chunk.T @ chunk

        cov /= float(n - 1)
        cov = 0.5 * (cov + cov.T)

        eigvals, eigvecs = torch.linalg.eigh(cov)

        eigvals = eigvals.clamp_min(eps)
        inv_sqrt = torch.rsqrt(eigvals)

        whiten = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T

        return mean.cpu().contiguous(), whiten.cpu().contiguous()

    def build_language_fused(self, clip_text_feat: torch.Tensor) -> torch.Tensor:
        """
        Build language_fused:

            language_fused = clip_text_feat + gate * MLP(LN(clip_text_feat))

        At initialization, gate is all zeros, so:

            language_fused == clip_text_feat
        """

        self._check_clip_input(clip_text_feat)

        clip_text_feat = clip_text_feat.float()

        mlp_x = self.mlp_ln(clip_text_feat)
        mlp_x = self.mlp_fc1(mlp_x)
        mlp_x = self.activation(mlp_x)
        mlp_x = self.dropout(mlp_x)
        mlp_x = self.mlp_fc2(mlp_x)

        mlp_x = mlp_x * self.residual_gate.float()

        language_fused = clip_text_feat + mlp_x
        return language_fused

    def encode_text(self, clip_text_feat: torch.Tensor) -> torch.Tensor:
        """
        Return L2-normalized text representation for image-text contrastive learning.
        """

        language_fused = self.build_language_fused(clip_text_feat)

        text_z = F.normalize(
            language_fused,
            p=2,
            dim=-1,
            eps=self.normalize_eps,
        )

        return text_z

    def whiten_student(self, language_fused: torch.Tensor) -> torch.Tensor:
        """
        Student whitening.

        This path is trainable because student_whiten_matrix is an nn.Parameter.
        """

        language_fused = language_fused.float()

        return (
            language_fused - self.student_whiten_mean.float()
        ) @ self.student_whiten_matrix.float()

    def whiten_teacher(self, llm_guidance: torch.Tensor) -> torch.Tensor:
        """
        Teacher whitening.

        Teacher features may be stored as float16, but are converted to float32.
        Teacher whitening matrix is frozen.
        """

        self._check_teacher_input(llm_guidance)

        llm_guidance = llm_guidance.float()

        return (
            llm_guidance - self.teacher_whiten_mean.float()
        ) @ self.teacher_whiten_matrix.float()

    def relation_kd_loss(
        self,
        student_white: torch.Tensor,
        teacher_white: torch.Tensor,
        *,
        temperature: Optional[float] = None,
        exclude_self_similarity: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        scale_by_temperature_squared: bool = True,
    ) -> torch.Tensor:
        """
        Compute relation distillation KL loss.

        Teacher:
            softmax(teacher_sim / temperature)

        Student:
            log_softmax(student_sim / temperature)

        Loss:
            KL(teacher_distribution || student_distribution)

        Implemented with:
            F.kl_div(student_log_probs, teacher_probs)

        because PyTorch expects:
            input  = log probability
            target = probability
        """

        if student_white.ndim != 2 or teacher_white.ndim != 2:
            raise ValueError("student_white and teacher_white must both be 2D.")

        if student_white.shape[0] != teacher_white.shape[0]:
            raise ValueError(
                "Student and teacher batch size must match. "
                f"Got {student_white.shape[0]} and {teacher_white.shape[0]}."
            )

        batch_size = student_white.shape[0]

        if batch_size < 2:
            return student_white.new_zeros(())

        if temperature is None:
            temperature = self.relation_temperature

        if exclude_self_similarity is None:
            exclude_self_similarity = self.exclude_self_similarity

        student_norm = F.normalize(
            student_white.float(),
            p=2,
            dim=-1,
            eps=self.normalize_eps,
        )

        with torch.no_grad():
            teacher_norm = F.normalize(
                teacher_white.float(),
                p=2,
                dim=-1,
                eps=self.normalize_eps,
            )

        if chunk_size is None:
            loss = self._relation_kd_loss_full(
                student_norm=student_norm,
                teacher_norm=teacher_norm,
                temperature=temperature,
                exclude_self_similarity=exclude_self_similarity,
            )
        else:
            loss = self._relation_kd_loss_chunked(
                student_norm=student_norm,
                teacher_norm=teacher_norm,
                temperature=temperature,
                exclude_self_similarity=exclude_self_similarity,
                chunk_size=chunk_size,
            )

        if scale_by_temperature_squared:
            loss = loss * (temperature ** 2)

        return loss

    def _relation_kd_loss_full(
        self,
        *,
        student_norm: torch.Tensor,
        teacher_norm: torch.Tensor,
        temperature: float,
        exclude_self_similarity: bool,
    ) -> torch.Tensor:
        batch_size = student_norm.shape[0]

        student_logits = student_norm @ student_norm.T
        teacher_logits = teacher_norm @ teacher_norm.T

        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if exclude_self_similarity:
            eye = torch.eye(
                batch_size,
                dtype=torch.bool,
                device=student_logits.device,
            )
            student_logits = student_logits.masked_fill(eye, -torch.inf)
            teacher_logits = teacher_logits.masked_fill(eye, -torch.inf)

        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=-1)

        student_log_probs = F.log_softmax(student_logits, dim=-1)

        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        return loss

    def _relation_kd_loss_chunked(
        self,
        *,
        student_norm: torch.Tensor,
        teacher_norm: torch.Tensor,
        temperature: float,
        exclude_self_similarity: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Memory-saving version for very large batches.

        Instead of materializing [B, B] at once, it computes [chunk, B].
        """

        batch_size = student_norm.shape[0]
        total_loss = student_norm.new_zeros(())
        total_rows = 0

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            rows = end - start

            student_logits = student_norm[start:end] @ student_norm.T
            teacher_logits = teacher_norm[start:end] @ teacher_norm.T

            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            if exclude_self_similarity:
                row_indices = torch.arange(rows, device=student_logits.device)
                col_indices = torch.arange(start, end, device=student_logits.device)

                student_logits[row_indices, col_indices] = -torch.inf
                teacher_logits[row_indices, col_indices] = -torch.inf

            with torch.no_grad():
                teacher_probs = F.softmax(teacher_logits, dim=-1)

            student_log_probs = F.log_softmax(student_logits, dim=-1)

            chunk_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            )

            total_loss = total_loss + chunk_loss * rows
            total_rows += rows

        return total_loss / float(total_rows)

    def forward(
        self,
        clip_text_feat: torch.Tensor,
        llm_guidance: Optional[torch.Tensor] = None,
        *,
        compute_relation_loss: bool = True,
        relation_chunk_size: Optional[int] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            clip_text_feat:
                CLIP text feature after encode_text.
                base  : [B, 512]
                large : [B, 768]

            llm_guidance:
                Teacher feature from LLM2Vec or decoder-only model.
                Shape: [B, teacher_dim]

            compute_relation_loss:
                Whether to compute relation KD loss.

            relation_chunk_size:
                Use chunked relation loss for very large batch sizes.
                For batch size > 10000, values like 1024, 2048, or 4096
                are safer than full [B, B] materialization.

        Returns:
            If return_dict=True:
                {
                    "text_z": L2-normalized text embedding,
                    "language_fused": unnormalized fused CLIP-space feature,
                    "relation_kd_loss": unweighted KL loss,
                    "weighted_relation_kd_loss": relation_weight * KL loss,
                    ...
                }

            If return_dict=False:
                text_z
        """

        self._check_clip_input(clip_text_feat)

        if self.force_fp32:
            ctx = (
                torch.autocast(device_type="cuda", enabled=False)
                if clip_text_feat.is_cuda
                else contextlib.nullcontext()
            )
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            clip_text_feat = clip_text_feat.float()

            language_fused = self.build_language_fused(clip_text_feat)

            text_z = F.normalize(
                language_fused,
                p=2,
                dim=-1,
                eps=self.normalize_eps,
            )

            outputs = {
                "text_z": text_z,
                "language_fused": language_fused,
            }

            if compute_relation_loss:
                if llm_guidance is None:
                    raise ValueError(
                        "llm_guidance must be provided when compute_relation_loss=True."
                    )

                self._check_teacher_input(llm_guidance)

                student_white = self.whiten_student(language_fused)

                with torch.no_grad():
                    teacher_white = self.whiten_teacher(llm_guidance)

                relation_kd_loss = self.relation_kd_loss(
                    student_white=student_white,
                    teacher_white=teacher_white,
                    temperature=self.relation_temperature,
                    exclude_self_similarity=self.exclude_self_similarity,
                    chunk_size=relation_chunk_size,
                )

                outputs.update(
                    {
                        "student_white": student_white,
                        "teacher_white": teacher_white,
                        "relation_kd_loss": relation_kd_loss,
                        "weighted_relation_kd_loss": self.relation_weight * relation_kd_loss,
                    }
                )

        if not return_dict:
            return text_z

        return outputs

    def _check_clip_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"clip_text_feat must be 2D [B, {self.clip_dim}], "
                f"got shape={tuple(x.shape)}."
            )

        if x.shape[-1] != self.clip_dim:
            raise ValueError(
                f"clip_text_feat last dim must be {self.clip_dim} for arch={self.arch!r}, "
                f"got shape={tuple(x.shape)}."
            )

    def _check_teacher_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"llm_guidance must be 2D [B, {self.teacher_dim}], "
                f"got shape={tuple(x.shape)}."
            )

        if x.shape[-1] != self.teacher_dim:
            raise ValueError(
                f"llm_guidance last dim must be {self.teacher_dim} "
                f"for teacher={self.teacher_name!r}, got shape={tuple(x.shape)}."
            )