"""Frozen DINOv2 ViT backbone with an optional LoRA adapter on the last blocks.

The forward pass is split in two so the router and the detector can share work:

* ``forward_prefix`` runs patch embedding and the first ``n_blocks - adapt_blocks`` blocks. These
  blocks never carry an adapter, so their output is shared.
* ``forward_suffix`` runs the last ``adapt_blocks`` blocks. With LoRA disabled it gives the frozen
  CLS token for routing; with the selected product's LoRA enabled it gives adapted patch tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

from fscad.models.lora import inject_lora, set_lora_enabled

# Friendly names used in configs -> timm model names.
TIMM_NAMES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
}


@dataclass
class Features:
    cls: torch.Tensor  # (B, D)
    patches: torch.Tensor  # (B, N, D)
    grid: tuple[int, int]  # (h, w) with h * w = N


class DinoBackbone(nn.Module):
    def __init__(
        self,
        name: str = "dinov2_vits14",
        pretrained: bool = True,
        img_size: int = 448,
        adapt_blocks: int = 4,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_lora: bool = True,
    ):
        super().__init__()
        timm_name = TIMM_NAMES.get(name, name)
        self.vit = timm.create_model(
            timm_name, pretrained=pretrained, img_size=img_size, dynamic_img_size=True, num_classes=0
        )
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.n_prefix = self.vit.num_prefix_tokens  # CLS (+ registers)
        self.n_blocks = len(self.vit.blocks)
        self.adapt_blocks = adapt_blocks
        self.use_lora = use_lora
        if use_lora:
            inject_lora(self.vit, adapt_blocks, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    # The backbone always stays in eval mode (no dropout / drop-path); only LoRA weights train.
    def train(self, mode: bool = True):  # noqa: D102
        super().train(False)
        return self

    @property
    def split_index(self) -> int:
        return self.n_blocks - self.adapt_blocks

    def forward_prefix(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        v = self.vit
        h = v.patch_embed(x)
        if h.ndim == 4:  # dynamic_img_size -> (B, H, W, C)
            grid = (h.shape[1], h.shape[2])
        else:
            g = v.patch_embed.grid_size
            grid = (g[0], g[1])
        h = v._pos_embed(h)
        h = v.patch_drop(h)
        h = v.norm_pre(h)
        for blk in v.blocks[: self.split_index]:
            h = blk(h)
        return h, grid

    def forward_suffix(self, h: torch.Tensor, adapted: bool) -> torch.Tensor:
        set_lora_enabled(self.vit, adapted and self.use_lora)
        for blk in self.vit.blocks[self.split_index :]:
            h = blk(h)
        return self.vit.norm(h)

    def _split(self, tokens: torch.Tensor, grid: tuple[int, int]) -> Features:
        return Features(cls=tokens[:, 0], patches=tokens[:, self.n_prefix :], grid=grid)

    def forward(self, x: torch.Tensor, adapted: bool = True) -> Features:
        h, grid = self.forward_prefix(x)
        return self._split(self.forward_suffix(h, adapted), grid)

    def forward_both(self, x: torch.Tensor) -> tuple[Features, Features]:
        """Frozen features (for routing) and adapted features (for scoring) with a shared prefix."""
        h, grid = self.forward_prefix(x)
        frozen = self._split(self.forward_suffix(h, adapted=False), grid)
        adapted = self._split(self.forward_suffix(h, adapted=True), grid)
        return frozen, adapted


def build_backbone(cfg) -> DinoBackbone:
    """Build from a Hydra/OmegaConf ``model`` + ``lora`` config."""
    return DinoBackbone(
        name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        img_size=cfg.model.img_size,
        adapt_blocks=cfg.lora.blocks,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        use_lora=cfg.lora.enabled,
    )
