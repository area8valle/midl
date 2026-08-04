from __future__ import annotations

import gin
from torch import Tensor, nn


@gin.configurable
class CoxHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        risk: Tensor = self.net(x).squeeze(-1)
        return risk
