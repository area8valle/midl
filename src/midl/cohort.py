from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from midl.materials.params import CLINICAL_FEATURES

OAI_KL_PROBS: tuple[float, ...] = (0.261, 0.178, 0.286, 0.177, 0.099)
MOST_KL_PROBS: tuple[float, ...] = (0.258, 0.199, 0.283, 0.171, 0.089)
F = TypeVar("F", bound=Callable[..., object])
configurable = cast(Callable[[F], F], gin.configurable)


@dataclass
class CohortArrays:
    images: np.ndarray
    clinical: np.ndarray
    progression: np.ndarray
    tkr_time: np.ndarray
    tkr_event: np.ndarray
    kl_grade: np.ndarray

    def __len__(self) -> int:
        return int(self.images.shape[0])


def _draw_clinical(
    rng: np.random.Generator, kl: np.ndarray, age_mu: float, bmi_mu: float, female_p: float
) -> np.ndarray:
    n = kl.shape[0]
    age = rng.normal(age_mu, 9.0, n)
    sex = (rng.random(n) < female_p).astype(np.float64)
    bmi = rng.normal(bmi_mu, 5.0, n) + 0.6 * kl
    jsw_medial = np.clip(5.0 - 0.7 * kl + rng.normal(0, 0.5, n), 0.5, 7.0)
    jsw_lateral = np.clip(5.4 - 0.5 * kl + rng.normal(0, 0.5, n), 0.5, 7.5)
    womac_pain = np.clip(rng.normal(3.2 + 0.8 * kl, 3.0, n), 0, 20)
    womac_stiff = np.clip(rng.normal(1.5 + 0.4 * kl, 1.6, n), 0, 8)
    womac_func = np.clip(rng.normal(10.8 + 2.5 * kl, 12.0, n), 0, 68)
    charlson = rng.poisson(0.6, n).astype(np.float64)
    prior_injury = (rng.random(n) < 0.33).astype(np.float64)
    pase = np.clip(rng.normal(161, 82, n), 0, 400)
    smoking = (rng.random(n) < 0.09).astype(np.float64)
    cortico = (rng.random(n) < 0.12).astype(np.float64)
    cols = [
        age,
        sex,
        bmi,
        kl.astype(np.float64),
        jsw_medial,
        jsw_lateral,
        womac_pain,
        womac_stiff,
        womac_func,
        charlson,
        prior_injury,
        pase,
        smoking,
        cortico,
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def _latent_risk(clinical: np.ndarray, image_signal: np.ndarray) -> np.ndarray:
    kl = clinical[:, 3]
    bmi = clinical[:, 2]
    jsw = clinical[:, 4]
    pain = clinical[:, 6]
    z = 0.55 * kl + 0.045 * (bmi - 28.0) - 0.30 * (jsw - 4.0) + 0.03 * pain + 1.1 * image_signal
    return z.astype(np.float64)


@configurable
def synthesize_cohort(
    n: int = 256,
    image_size: int = 380,
    channels: int = 3,
    kl_probs: tuple[float, ...] = OAI_KL_PROBS,
    age_mu: float = 61.3,
    bmi_mu: float = 28.6,
    female_p: float = 0.582,
    prog_rate: float = 0.247,
    tkr_rate: float = 0.105,
    follow_up_months: float = 96.0,
    seed: int = 0,
) -> CohortArrays:
    rng = np.random.default_rng(seed)
    kl = rng.choice(len(kl_probs), size=n, p=np.asarray(kl_probs) / np.sum(kl_probs))
    clinical = _draw_clinical(rng, kl, age_mu, bmi_mu, female_p)
    base = kl.astype(np.float64)[:, None, None, None] / 4.0 - 0.5
    noise = rng.normal(0, 1.0, size=(n, channels, image_size, image_size))
    images = (0.35 * base + 0.65 * noise).astype(np.float32)
    image_signal = images.reshape(n, -1).mean(axis=1) + 0.5 * base.reshape(n)
    risk = _latent_risk(clinical, image_signal)
    prog_logit = risk - np.quantile(risk, 1.0 - prog_rate)
    progression = (rng.random(n) < _sigmoid(prog_logit)).astype(np.float32)
    hazard = np.exp(0.8 * (risk - risk.mean()) / (risk.std() + 1e-6))
    raw_time = rng.exponential(follow_up_months / (tkr_rate * 6.0) / np.clip(hazard, 0.2, 5.0))
    tkr_event = (raw_time <= follow_up_months).astype(np.float32)
    keep = rng.random(n) < (tkr_rate / max(tkr_event.mean(), 1e-6))
    tkr_event = (tkr_event * keep).astype(np.float32)
    tkr_time = np.minimum(raw_time, follow_up_months).astype(np.float32)
    return CohortArrays(images, clinical, progression, tkr_time, tkr_event, kl.astype(np.int64))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def feature_medians(clinical: np.ndarray) -> np.ndarray:
    return np.nanmedian(clinical, axis=0).astype(np.float32)


def impute_median(clinical: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = clinical.copy()
    mask = np.isnan(out)
    idx = np.where(mask)
    out[idx] = np.take(medians, idx[1])
    return out


def standardize(clinical: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((clinical - mean) / np.clip(std, 1e-6, None)).astype(np.float32)


def _augment(image: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    if torch.rand(1, generator=rng).item() < 0.5:
        image = torch.flip(image, dims=[-1])
    angle = (torch.rand(1, generator=rng).item() * 2 - 1) * 10.0
    image = _rotate(image, angle)
    bright = 1.0 + (torch.rand(1, generator=rng).item() * 2 - 1) * 0.15
    contrast = 1.0 + (torch.rand(1, generator=rng).item() * 2 - 1) * 0.15
    mean = image.mean()
    return (image * bright - mean) * contrast + mean


def _rotate(image: torch.Tensor, angle_deg: float) -> torch.Tensor:
    try:
        from torchvision.transforms.functional import rotate

        return rotate(image, angle_deg)
    except Exception:
        return image


class KneeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        arrays: CohortArrays,
        train: bool = False,
        medians: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        clinical = arrays.clinical
        if medians is not None:
            clinical = impute_median(clinical, medians)
        if mean is not None and std is not None:
            clinical = standardize(clinical, mean, std)
        self._images = torch.from_numpy(arrays.images)
        self._clinical = torch.from_numpy(clinical)
        self._prog = torch.from_numpy(arrays.progression)
        self._time = torch.from_numpy(arrays.tkr_time)
        self._event = torch.from_numpy(arrays.tkr_event)
        self._kl = torch.from_numpy(arrays.kl_grade)
        self._train = train
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self._images[index]
        if self._train:
            image = _augment(image, self._gen)
        return {
            "image": image,
            "clinical": self._clinical[index],
            "progression": self._prog[index],
            "tkr_time": self._time[index],
            "tkr_event": self._event[index],
            "kl_grade": self._kl[index],
        }


def stratified_split(
    kl: np.ndarray, fractions: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for grade in np.unique(kl):
        members = np.where(kl == grade)[0]
        rng.shuffle(members)
        n = members.shape[0]
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        train_idx.extend(members[:n_train].tolist())
        val_idx.extend(members[n_train : n_train + n_val].tolist())
        test_idx.extend(members[n_train + n_val :].tolist())
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def _subset(arrays: CohortArrays, idx: np.ndarray) -> CohortArrays:
    return CohortArrays(
        arrays.images[idx],
        arrays.clinical[idx],
        arrays.progression[idx],
        arrays.tkr_time[idx],
        arrays.tkr_event[idx],
        arrays.kl_grade[idx],
    )


@configurable
def make_dataloaders(
    arrays: CohortArrays | None = None,
    batch_size: int = 32,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    if arrays is None:
        arrays = synthesize_cohort(seed=seed)
    if len(CLINICAL_FEATURES) != arrays.clinical.shape[1]:
        raise ValueError("clinical feature count mismatch")
    tr_idx, va_idx, te_idx = stratified_split(arrays.kl_grade, fractions, seed)
    train_arr = _subset(arrays, tr_idx)
    medians = feature_medians(train_arr.clinical)
    imputed = impute_median(train_arr.clinical, medians)
    mean = imputed.mean(axis=0).astype(np.float32)
    std = imputed.std(axis=0).astype(np.float32)
    splits = {
        "train": (train_arr, True),
        "val": (_subset(arrays, va_idx), False),
        "test": (_subset(arrays, te_idx), False),
    }
    loaders: dict[str, DataLoader[dict[str, torch.Tensor]]] = {}
    for name, (arr, is_train) in splits.items():
        ds = KneeDataset(arr, train=is_train, medians=medians, mean=mean, std=std, seed=seed)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, drop_last=False
        )
    return loaders
