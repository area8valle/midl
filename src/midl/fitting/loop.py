from __future__ import annotations

import math
import os
from collections.abc import Iterable

import gin
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from midl.fitting.loss import CompositeObjective
from midl.materials.params import resolve_device, set_seed
from midl.observables.gauges import auroc
from midl.solver import MIDL

Batch = dict[str, Tensor]


@gin.configurable
class Trainer:
    def __init__(
        self,
        model: nn.Module | None = None,
        objective: CompositeObjective | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        epochs: int = 100,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        grad_clip: float = 1.0,
        patience: int = 15,
        amp: bool = False,
        ema_decay: float = 0.0,
        seed: int = 42,
        out_dir: str = "runs/midl",
        device: str = "auto",
    ) -> None:
        set_seed(seed)
        self.device = resolve_device(device)
        self.model = (model if model is not None else MIDL()).to(self.device)
        self.objective = objective if objective is not None else CompositeObjective()
        self.base_lr = lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.seed = seed
        self.out_dir = out_dir
        self.amp = amp and self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.ema_decay = ema_decay
        self.ema: dict[str, Tensor] | None = None
        if ema_decay > 0:
            self.ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def lr_for_epoch(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        span = max(self.epochs - self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / span
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _to_device(self, batch: Batch) -> Batch:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _loss(self, batch: Batch) -> tuple[Tensor, dict[str, float]]:
        output = self.model(batch["image"], batch["clinical"])
        return self.objective(output, batch["progression"], batch["tkr_time"], batch["tkr_event"])

    def _update_ema(self) -> None:
        if self.ema is None:
            return
        for key, value in self.model.state_dict().items():
            self.ema[key].mul_(self.ema_decay).add_(value.detach(), alpha=1 - self.ema_decay)

    def train_steps(self, loader: Iterable[Batch], n_steps: int) -> list[float]:
        self.model.train()
        losses: list[float] = []
        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss, parts = self._loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._update_ema()
            losses.append(parts["loss"])
            if step + 1 >= n_steps:
                break
        return losses

    def _run_epoch(self, loader: Iterable[Batch], epoch: int) -> float:
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_for_epoch(epoch)
        running = 0.0
        count = 0
        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                loss, parts = self._loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()
            running += parts["loss"]
            count += 1
        return running / max(count, 1)

    @torch.no_grad()
    def predict(self, loader: Iterable[Batch]) -> dict[str, np.ndarray]:
        self.model.eval()
        keys = ["prog_score", "prog_label", "cox_risk", "tkr_time", "tkr_event", "kl_grade"]
        buffers: dict[str, list[np.ndarray]] = {k: [] for k in keys}
        for batch in loader:
            batch = self._to_device(batch)
            output = self.model(batch["image"], batch["clinical"])
            buffers["prog_score"].append(output.progression_logit.cpu().numpy())
            buffers["prog_label"].append(batch["progression"].cpu().numpy())
            buffers["cox_risk"].append(output.cox_risk.cpu().numpy())
            buffers["tkr_time"].append(batch["tkr_time"].cpu().numpy())
            buffers["tkr_event"].append(batch["tkr_event"].cpu().numpy())
            buffers["kl_grade"].append(batch["kl_grade"].cpu().numpy())
        return {k: np.concatenate(v) for k, v in buffers.items()}

    def validate(self, loader: Iterable[Batch]) -> float:
        preds = self.predict(loader)
        return auroc(preds["prog_score"], preds["prog_label"])

    def fit(self, loaders: dict[str, DataLoader[Batch]]) -> dict[str, float]:
        os.makedirs(self.out_dir, exist_ok=True)
        best = -1.0
        stale = 0
        for epoch in range(self.epochs):
            self._run_epoch(loaders["train"], epoch)
            score = self.validate(loaders["val"]) if "val" in loaders else 0.0
            if score > best:
                best = score
                stale = 0
                self.save_checkpoint(os.path.join(self.out_dir, "best.pt"), epoch, best)
            else:
                stale += 1
            self.save_checkpoint(os.path.join(self.out_dir, "last.pt"), epoch, best)
            if stale >= self.patience:
                break
        return {"best_val_auroc": best}

    def save_checkpoint(self, path: str, epoch: int, best: float) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "seed": self.seed,
            "ema": self.ema,
            "gin_config": gin.operative_config_str(),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str) -> dict[str, object]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.seed = int(payload["seed"])
        self.ema = payload.get("ema")
        set_seed(self.seed)
        return payload
