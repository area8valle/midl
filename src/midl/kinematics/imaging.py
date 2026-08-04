from __future__ import annotations

import gin
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

EFFICIENTNET_B4_FEATURES = 1792


def clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)
) -> np.ndarray:
    h, w = image.shape
    lo, hi = float(image.min()), float(image.max())
    scaled = (
        np.zeros_like(image, dtype=np.uint8)
        if hi <= lo
        else (((image - lo) / (hi - lo) * 255.0).astype(np.uint8))
    )
    ty, tx = tile_grid
    th, tw = h // ty, w // tx
    maps = np.zeros((ty, tx, 256), dtype=np.float32)
    for i in range(ty):
        for j in range(tx):
            y0, x0 = i * th, j * tw
            y1 = h if i == ty - 1 else y0 + th
            x1 = w if j == tx - 1 else x0 + tw
            tile = scaled[y0:y1, x0:x1]
            hist = np.bincount(tile.reshape(-1), minlength=256).astype(np.float32)
            limit = max(1.0, clip_limit * hist.mean())
            excess = np.clip(hist - limit, 0, None).sum()
            hist = np.clip(hist, None, limit) + excess / 256.0
            cdf = np.cumsum(hist)
            cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-6)
            maps[i, j] = cdf * 255.0
    out = np.zeros_like(scaled, dtype=np.float32)
    for y in range(h):
        fy = min(max((y + 0.5) / th - 0.5, 0.0), ty - 1.0)
        i0 = int(np.floor(fy))
        i1 = min(i0 + 1, ty - 1)
        wy = fy - i0
        for x in range(w):
            fx = min(max((x + 0.5) / tw - 0.5, 0.0), tx - 1.0)
            j0 = int(np.floor(fx))
            j1 = min(j0 + 1, tx - 1)
            wx = fx - j0
            v = scaled[y, x]
            top = maps[i0, j0, v] * (1 - wx) + maps[i0, j1, v] * wx
            bot = maps[i1, j0, v] * (1 - wx) + maps[i1, j1, v] * wx
            out[y, x] = top * (1 - wy) + bot * wy
    return out / 255.0


def preprocess_radiograph(
    raw: np.ndarray,
    roi_size: int = 280,
    out_size: int = 380,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Tensor:
    h, w = raw.shape
    cy, cx = h // 2, w // 2
    half = roi_size // 2
    y0, x0 = max(cy - half, 0), max(cx - half, 0)
    roi = raw[y0 : y0 + roi_size, x0 : x0 + roi_size]
    equalized = clahe(roi, clip_limit, tile_grid)
    tensor = torch.from_numpy(equalized.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(out_size, out_size), mode="bicubic", align_corners=False)
    return resized.squeeze(0).repeat(3, 1, 1)


class _LiteStem(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(96, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.body(x).flatten(1)
        projected: Tensor = self.proj(h)
        return projected


@gin.configurable
class RadiographEncoder(nn.Module):
    features: nn.Module
    pool: nn.Module
    proj: nn.Module

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = False,
        out_dim: int = EFFICIENTNET_B4_FEATURES,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        if backbone == "efficientnet_b4":
            self.features, feat_dim = _build_efficientnet(pretrained)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(feat_dim, out_dim)
            self._lite: _LiteStem | None = None
        elif backbone == "lite":
            self.features = nn.Identity()
            self.pool = nn.Identity()
            self.proj = nn.Identity()
            self._lite = _LiteStem(out_dim)
        else:
            raise ValueError(f"unknown backbone {backbone}")

    def forward(self, x: Tensor) -> Tensor:
        if self._lite is not None:
            lite_features: Tensor = self._lite(x)
            return lite_features
        h = self.pool(self.features(x)).flatten(1)
        projected: Tensor = self.proj(h)
        return projected


def _build_efficientnet(pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    net = efficientnet_b4(weights=weights)
    return net.features, EFFICIENTNET_B4_FEATURES
