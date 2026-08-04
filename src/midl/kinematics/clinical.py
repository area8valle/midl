from __future__ import annotations

import gin
from torch import Tensor, nn


@gin.configurable
class ClinicalEncoder(nn.Module):
    def __init__(
        self, n_features: int = 14, hidden: tuple[int, ...] = (128, 64, 64), out_dim: int = 64
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.GELU())
            prev = width
        self.body = nn.Sequential(*layers)
        self.out_dim = out_dim
        self.head = nn.Identity() if prev == out_dim else nn.Linear(prev, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        encoded: Tensor = self.head(self.body(x))
        return encoded
