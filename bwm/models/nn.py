"""Shared PyTorch building blocks.

Kept deliberately small and explicit: every system in the comparison is built
from the same primitives, so architectural differences between (say) the
Transformer baseline and the Transformer world model are differences of
*structure*, not of engineering quality.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MLP", "SinusoidalPositions", "TemporalTransformer", "GraphBlock",
           "GraphEncoder", "count_params", "symlog", "symexp"]


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    ps = module.parameters()
    return int(sum(p.numel() for p in ps if p.requires_grad or not trainable_only))


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving log compression (Dreamer-v3).

    Financial targets are heavy tailed; symlog keeps a single loss scale usable
    across five orders of magnitude of reward without clipping information away.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class MLP(nn.Module):
    def __init__(self, sizes: Sequence[int], act: str = "silu", out_act: bool = False,
                 norm: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        acts = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        A = acts[act]
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            last = i == len(sizes) - 2
            if not last or out_act:
                if norm:
                    layers.append(nn.LayerNorm(sizes[i + 1]))
                layers.append(A())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositions(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class TemporalTransformer(nn.Module):
    """Causal Transformer encoder over a short observation/action window."""

    def __init__(self, d_in: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ff_mult: int = 2, dropout: float = 0.1,
                 max_len: int = 512) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = SinusoidalPositions(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, d_in) -> (B, T, d_model) with causal masking."""
        T = x.shape[1]
        h = self.pos(self.proj(x))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        return self.norm(self.enc(h, mask=mask))


class GraphBlock(nn.Module):
    """One round of message passing on a fixed heterogeneous graph."""

    def __init__(self, d_node: int, d_edge: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        h = hidden or d_node
        self.msg = MLP([2 * d_node + d_edge, h, d_node], norm=True)
        self.upd = MLP([2 * d_node, h, d_node], norm=True)
        self.norm = nn.LayerNorm(d_node)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        """``h``: (B, N, d);  ``edge_index``: (2, E);  ``edge_feat``: (E, d_edge)."""
        src, dst = edge_index[0], edge_index[1]
        B, N, D = h.shape
        hs, hd = h[:, src], h[:, dst]                       # (B, E, D)
        ef = edge_feat.unsqueeze(0).expand(B, -1, -1)
        m = self.msg(torch.cat([hs, hd, ef], dim=-1))       # (B, E, D)
        agg = torch.zeros_like(h).index_add_(1, dst, m)
        deg = torch.zeros(N, device=h.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=h.dtype)).clamp(min=1.0)
        agg = agg / deg.view(1, N, 1)
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))


class GraphEncoder(nn.Module):
    """Encodes one graph snapshot into a pooled vector plus node states."""

    def __init__(self, d_in: int, d_model: int = 64, d_edge: int = 7,
                 n_layers: int = 2, n_node_types: int = 5) -> None:
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.type_emb = nn.Embedding(n_node_types, d_model)
        self.blocks = nn.ModuleList(
            [GraphBlock(d_model, d_edge) for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """``x``: (B, N, d_in) -> (B, N, d_model)."""
        h = self.inp(x) + self.type_emb(node_type).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h, edge_index, edge_feat)
        return h
