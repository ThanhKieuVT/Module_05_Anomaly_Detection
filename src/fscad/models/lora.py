"""LoRA for the fused qkv projection of a timm ViT.

timm (and the official DINOv2 code) compute Q, K, V with a single ``Linear(D, 3D)``. peft can only
wrap that layer as a whole, so it cannot adapt Q and V while leaving K untouched. This module wraps
the fused layer and adds a low-rank update to the Q and V slices only:

    q = x W_Q + (alpha / r) * x A_q^T B_q^T,   v = x W_V + (alpha / r) * x A_v^T B_v^T

B is initialised to zero, so a freshly injected adapter leaves the backbone output unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


class LoRAQKV(nn.Module):
    """Wraps a fused qkv ``nn.Linear`` and adds LoRA updates to Q and V."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError("expected a fused qkv projection with out_features = 3 * dim")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.dim = base.in_features
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

        self.lora_A_q = nn.Parameter(torch.empty(r, self.dim))
        self.lora_B_q = nn.Parameter(torch.zeros(self.dim, r))
        self.lora_A_v = nn.Parameter(torch.empty(r, self.dim))
        self.lora_B_v = nn.Parameter(torch.zeros(self.dim, r))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.zeros_(self.lora_B_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        h = self.dropout(x)
        dq = (h @ self.lora_A_q.T) @ self.lora_B_q.T
        dv = (h @ self.lora_A_v.T) @ self.lora_B_v.T
        d = self.dim
        # out = [q | k | v]; only q and v receive an update.
        return torch.cat(
            [out[..., :d] + self.scale * dq, out[..., d : 2 * d], out[..., 2 * d :] + self.scale * dv],
            dim=-1,
        )


def inject_lora(
    vit: nn.Module, last_n_blocks: int, r: int = 8, alpha: float = 16.0, dropout: float = 0.0
) -> list[LoRAQKV]:
    """Replace ``attn.qkv`` in the last ``last_n_blocks`` blocks with :class:`LoRAQKV`."""
    blocks = list(vit.blocks)
    if not 0 < last_n_blocks <= len(blocks):
        raise ValueError(f"last_n_blocks must be in [1, {len(blocks)}]")
    modules = []
    for blk in blocks[len(blocks) - last_n_blocks :]:
        qkv = blk.attn.qkv
        if isinstance(qkv, LoRAQKV):
            modules.append(qkv)
            continue
        lora = LoRAQKV(qkv, r=r, alpha=alpha, dropout=dropout).to(qkv.weight.device)
        blk.attn.qkv = lora
        modules.append(lora)
    return modules


def lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRAQKV]]:
    for name, m in model.named_modules():
        if isinstance(m, LoRAQKV):
            yield name, m


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for _, m in lora_modules(model) for n, p in m.named_parameters() if n.startswith("lora_")]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter weights: this is what gets stored per product."""
    return {
        f"{name}.{n}": p.detach().cpu().clone()
        for name, m in lora_modules(model)
        for n, p in m.named_parameters()
        if n.startswith("lora_")
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = {
        f"{name}.{n}": p
        for name, m in lora_modules(model)
        for n, p in m.named_parameters()
        if n.startswith("lora_")
    }
    missing = set(params) - set(state)
    unexpected = set(state) - set(params)
    if missing or unexpected:
        raise KeyError(f"LoRA state mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))


def reset_lora(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.reset_parameters()


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for _, m in lora_modules(model):
        m.enabled = enabled


def expected_lora_param_count(dim: int, r: int, n_blocks: int) -> int:
    """L * 2 (Q and V) * r * (d_in + d_out), the formula used in the report."""
    return n_blocks * 2 * r * (dim + dim)
