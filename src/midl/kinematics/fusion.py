from __future__ import annotations

import math

import gin
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@gin.configurable
class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim: int, context_dims: tuple[int, ...], embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.ModuleList(nn.Linear(c, embed_dim) for c in context_dims)
        self.v_proj = nn.ModuleList(nn.Linear(c, embed_dim) for c in context_dims)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, query: Tensor, contexts: list[Tensor]) -> Tensor:
        q = self.q_proj(query).unsqueeze(1)
        keys = torch.stack([proj(c) for proj, c in zip(self.k_proj, contexts, strict=True)], dim=1)
        vals = torch.stack([proj(c) for proj, c in zip(self.v_proj, contexts, strict=True)], dim=1)
        scores = torch.matmul(q, keys.transpose(1, 2)) / math.sqrt(self.embed_dim)
        weights = F.softmax(scores, dim=-1)
        fused = torch.matmul(weights, vals).squeeze(1)
        normed: Tensor = self.out_norm(fused)
        return normed
