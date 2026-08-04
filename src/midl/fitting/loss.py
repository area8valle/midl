from __future__ import annotations

import gin
import torch
from torch import Tensor
from torch.nn import functional as F

from midl.dynamics.residual import physics_residual
from midl.solver import ModelOutput, build_dynamics


def weighted_bce(logit: Tensor, target: Tensor) -> Tensor:
    pos = target.sum()
    neg = target.numel() - pos
    weight = (neg / pos) if float(pos) > 0 else torch.ones((), device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=weight)


def cox_partial_nll(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order]
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
    contributions = (ordered_risk - log_risk_set) * ordered_event
    return -contributions.sum() / ordered_event.sum().clamp(min=1.0)


def bounds_penalty(x: Tensor, low: float, high: float) -> Tensor:
    return (F.relu(low - x) ** 2 + F.relu(x - high) ** 2).mean()


@gin.configurable
class CompositeObjective:
    def __init__(
        self,
        lambda_tkr: float = 0.5,
        lambda_phys: float = 0.1,
        lambda_reg: float = 0.01,
        er_bounds: tuple[float, float] = (0.5, 50.0),
        tau_sigma_bounds: tuple[float, float] = (1.0, 1000.0),
    ) -> None:
        self.lambda_tkr = lambda_tkr
        self.lambda_phys = lambda_phys
        self.lambda_reg = lambda_reg
        self.er_bounds = er_bounds
        self.tau_sigma_bounds = tau_sigma_bounds

    def __call__(
        self, output: ModelOutput, progression: Tensor, tkr_time: Tensor, tkr_event: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        l_prog = weighted_bce(output.progression_logit, progression)
        l_tkr = cox_partial_nll(output.cox_risk, tkr_time, tkr_event)
        l_phys = output.fused.new_zeros(())
        aux = output.aux
        if (
            self.lambda_phys > 0
            and aux.trajectory is not None
            and aux.params is not None
            and aux.t_nodes is not None
            and aux.name in ("sls", "polynomial", "windkessel")
        ):
            dynamics = build_dynamics(aux.name, aux.t_nodes)
            l_phys = physics_residual(aux.trajectory, aux.t_nodes, dynamics, aux.params).mean()
        l_reg = output.fused.new_zeros(())
        if aux.e_r is not None and aux.tau_sigma is not None:
            l_reg = bounds_penalty(aux.e_r, *self.er_bounds) + bounds_penalty(
                aux.tau_sigma, *self.tau_sigma_bounds
            )
        total = (
            l_prog + self.lambda_tkr * l_tkr + self.lambda_phys * l_phys + self.lambda_reg * l_reg
        )
        parts = {
            "loss": float(total.detach()),
            "progression": float(l_prog.detach()),
            "tkr": float(l_tkr.detach()),
            "physics": float(l_phys.detach()),
            "regularization": float(l_reg.detach()),
        }
        return total, parts
