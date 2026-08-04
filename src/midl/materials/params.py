from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import gin
import numpy as np
import torch

CLINICAL_FEATURES: tuple[str, ...] = (
    "age",
    "sex",
    "bmi",
    "baseline_kl",
    "jsw_medial",
    "jsw_lateral",
    "womac_pain",
    "womac_stiffness",
    "womac_function",
    "charlson_index",
    "prior_injury",
    "pase_score",
    "current_smoking",
    "prior_corticosteroid",
)

KL_GRADES: tuple[int, ...] = (0, 1, 2, 3, 4)

ER_BOUNDS_MPA: tuple[float, float] = (0.5, 50.0)
TAU_SIGMA_BOUNDS_S: tuple[float, float] = (1.0, 1000.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


@gin.configurable
@dataclass(frozen=True)
class RunSettings:
    seed: int = 42
    device: str = "auto"
    out_dir: str = "runs/midl"
    amp: bool = False
    num_workers: int = 0
    feature_names: tuple[str, ...] = field(default=CLINICAL_FEATURES)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


gin.external_configurable(torch.optim.AdamW, "AdamW")
gin.external_configurable(torch.optim.Adam, "Adam")
