from __future__ import annotations

import contextlib
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionConnector(nn.Module):
    """
    CLIP-gated DINO fusion connector.

    Inputs:
        base:
            clip_feat : [B, 768]
            dino_cls  : [B, 768]
            dino_gap  : [B, 768]

        large:
            clip_feat : [B, 1024]
            dino_cls  : [B, 1024]
            dino_gap  : [B, 1024]

    Output:
        base:
            [B, 512]

        large:
            [B, 768]

    Design:
        1. dino_cls / dino_gap:
              LayerNorm -> projection to CLIP hidden space
              concat -> dino_fused -> LayerNorm

        2. clip:
              LayerNorm

        3. gated fusion:
              gate  = swish(clip_ln @ W)
              value = dino_fused_ln @ V
              hidden = gate * value

        4. hidden -> zero-initialized down projection

        5. add frozen original CLIP branch:
              final = normalize(clip_feat @ frozen_visual_proj + adapter_out)
    """

    ARCH_CONFIGS: Dict[str, Dict[str, Union[str, int]]] = {
        "base": {
            "input_dim": 768,
            "clip_embed_dim": 512,
            "clip_model_name": "ViT-B-16",
            "clip_pretrained": "datacomp_xl_s13b_b90k",
        },
        "large": {
            "input_dim": 1024,
            "clip_embed_dim": 768,
            "clip_model_name": "ViT-L-14",
            "clip_pretrained": "datacomp_xl_s13b_b90k",
        },
    }

    def __init__(
        self,
        arch: str = "base",
        *,
        clip_visual_proj: Union[torch.Tensor, nn.Parameter, nn.Linear],
        width: float = 2.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
        force_fp32: bool = True,
    ) -> None:
        super().__init__()

        if arch not in self.ARCH_CONFIGS:
            raise ValueError(
                f"Unsupported arch={arch!r}. "
                f"Expected one of {list(self.ARCH_CONFIGS.keys())}."
            )

        cfg = self.ARCH_CONFIGS[arch]

        self.arch = arch
        self.input_dim = int(cfg["input_dim"])
        self.clip_embed_dim = int(cfg["clip_embed_dim"])
        self.fused_dim = 2 * self.input_dim
        self.hidden_dim = int(round(width * self.fused_dim))

        self.width = width
        self.dropout_p = dropout
        self.eps = eps
        self.force_fp32 = force_fp32

        proj = self._normalize_clip_proj(
            clip_visual_proj,
            expected_input_dim=self.input_dim,
            expected_output_dim=self.clip_embed_dim,
        )

        # Frozen original CLIP visual.proj.
        # Shape: [input_dim, clip_embed_dim]
        self.register_buffer("clip_visual_proj_frozen", proj.clone().float())

        # ------------------------------------------------------------
        # DINO branch:
        # dino_cls / dino_gap -> CLIP hidden space
        # ------------------------------------------------------------
        self.dino_cls_ln = nn.LayerNorm(self.input_dim)
        self.dino_gap_ln = nn.LayerNorm(self.input_dim)

        self.dino_cls_proj = nn.Linear(
            self.input_dim,
            self.input_dim,
            bias=False,
        )
        self.dino_gap_proj = nn.Linear(
            self.input_dim,
            self.input_dim,
            bias=False,
        )

        # After concat: [B, 2 * input_dim]
        self.dino_fused_ln = nn.LayerNorm(self.fused_dim)

        # ------------------------------------------------------------
        # CLIP gate branch:
        # clip -> LN -> W -> swish
        # ------------------------------------------------------------
        self.clip_ln = nn.LayerNorm(self.input_dim)

        self.gate_proj = nn.Linear(
            self.input_dim,
            self.hidden_dim,
            bias=False,
        )

        # ------------------------------------------------------------
        # DINO value branch:
        # dino_fused -> V
        # ------------------------------------------------------------
        self.value_proj = nn.Linear(
            self.fused_dim,
            self.hidden_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------
        # Zero-init down projection:
        # hidden_dim -> CLIP text-aligned embedding dim
        #
        # Because this is initialized to all zeros:
        #     adapter_out == 0 at initialization
        #     output == normalize(original CLIP branch)
        # ------------------------------------------------------------
        self.down_proj = nn.Linear(
            self.hidden_dim,
            self.clip_embed_dim,
            bias=False,
        )
        nn.init.zeros_(self.down_proj.weight)

    @staticmethod
    def _normalize_clip_proj(
        clip_visual_proj: Union[torch.Tensor, nn.Parameter, nn.Linear],
        *,
        expected_input_dim: int,
        expected_output_dim: int,
    ) -> torch.Tensor:
        """
        Normalize possible projection formats into:

            [input_dim, output_dim]

        OpenCLIP model.visual.proj is usually:
            [input_dim, output_dim]

        nn.Linear.weight is:
            [output_dim, input_dim]
        """

        if isinstance(clip_visual_proj, nn.Linear):
            proj = clip_visual_proj.weight.detach().T
        else:
            proj = torch.as_tensor(clip_visual_proj).detach()

        if proj.ndim != 2:
            raise ValueError(
                f"clip_visual_proj must be 2D, got shape={tuple(proj.shape)}."
            )

        if proj.shape == (expected_input_dim, expected_output_dim):
            return proj.float().contiguous()

        if proj.shape == (expected_output_dim, expected_input_dim):
            return proj.T.float().contiguous()

        raise ValueError(
            "Unexpected clip_visual_proj shape. "
            f"Got {tuple(proj.shape)}, expected either "
            f"{(expected_input_dim, expected_output_dim)} or "
            f"{(expected_output_dim, expected_input_dim)}."
        )

    @classmethod
    def from_open_clip(
        cls,
        arch: str = "base",
        *,
        width: float = 2.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
        force_fp32: bool = True,
        device: Optional[torch.device] = None,
    ) -> "VisionConnector":
        """
        Load OpenCLIP only to fetch model.visual.proj.
        """

        import open_clip

        if arch not in cls.ARCH_CONFIGS:
            raise ValueError(
                f"Unsupported arch={arch!r}. "
                f"Expected one of {list(cls.ARCH_CONFIGS.keys())}."
            )

        cfg = cls.ARCH_CONFIGS[arch]

        model, _, _ = open_clip.create_model_and_transforms(
            str(cfg["clip_model_name"]),
            pretrained=str(cfg["clip_pretrained"]),
        )

        visual_proj = getattr(model.visual, "proj", None)
        if visual_proj is None:
            raise ValueError("OpenCLIP model.visual.proj is None.")

        connector = cls(
            arch=arch,
            clip_visual_proj=visual_proj,
            width=width,
            dropout=dropout,
            eps=eps,
            force_fp32=force_fp32,
        )

        if device is not None:
            connector = connector.to(device)

        # Keep all trainable parameters in float32.
        connector.float()

        return connector

    def forward(
        self,
        clip_feat: torch.Tensor,
        dino_cls: torch.Tensor,
        dino_gap: torch.Tensor,
        *,
        return_dict: bool = False,
    ):
        self._check_input_shape("clip_feat", clip_feat)
        self._check_input_shape("dino_cls", dino_cls)
        self._check_input_shape("dino_gap", dino_gap)

        if self.force_fp32:
            ctx = (
                torch.autocast(device_type="cuda", enabled=False)
                if clip_feat.is_cuda
                else contextlib.nullcontext()
            )
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            clip_feat = clip_feat.float()
            dino_cls = dino_cls.float()
            dino_gap = dino_gap.float()

            # --------------------------------------------------------
            # Frozen original CLIP branch.
            # [B, input_dim] -> [B, clip_embed_dim]
            # --------------------------------------------------------
            clip_original = clip_feat @ self.clip_visual_proj_frozen.float()

            # --------------------------------------------------------
            # DINO value input.
            # dino_cls: [B, input_dim] -> [B, input_dim]
            # dino_gap: [B, input_dim] -> [B, input_dim]
            # concat  : [B, 2 * input_dim]
            # --------------------------------------------------------
            dino_cls_x = self.dino_cls_proj(self.dino_cls_ln(dino_cls))
            dino_gap_x = self.dino_gap_proj(self.dino_gap_ln(dino_gap))

            dino_fused = torch.cat([dino_cls_x, dino_gap_x], dim=-1)
            dino_fused = self.dino_fused_ln(dino_fused)

            # --------------------------------------------------------
            # CLIP gate.
            # clip_gate: [B, input_dim] -> [B, hidden_dim]
            # --------------------------------------------------------
            clip_gate_input = self.clip_ln(clip_feat)
            gate = self.gate_proj(clip_gate_input)
            gate = F.silu(gate)

            # --------------------------------------------------------
            # DINO value.
            # value: [B, 2 * input_dim] -> [B, hidden_dim]
            # --------------------------------------------------------
            value = self.value_proj(dino_fused)

            # --------------------------------------------------------
            # Cross-gated fusion.
            # hidden: [B, hidden_dim]
            # --------------------------------------------------------
            hidden = gate * value
            hidden = self.dropout(hidden)

            # --------------------------------------------------------
            # Zero-init down projection.
            # adapter_out: [B, clip_embed_dim]
            # --------------------------------------------------------
            adapter_out = self.down_proj(hidden)

            # --------------------------------------------------------
            # Final output.
            # --------------------------------------------------------
            pre_norm = clip_original + adapter_out
            z = F.normalize(pre_norm, p=2, dim=-1, eps=self.eps)

        if not return_dict:
            return z

        return {
            "z": z,
            "clip_original": clip_original,
            "dino_cls_x": dino_cls_x,
            "dino_gap_x": dino_gap_x,
            "dino_fused": dino_fused,
            "gate": gate,
            "value": value,
            "hidden": hidden,
            "adapter_out": adapter_out,
            "pre_norm": pre_norm,
        }

    def _check_input_shape(self, name: str, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"{name} must be 2D [B, {self.input_dim}], "
                f"got shape={tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"{name} last dim must be {self.input_dim} for arch={self.arch!r}, "
                f"got shape={tuple(x.shape)}."
            )