from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

Params = tuple[Tensor, ...]
Dynamics = Callable[[float, Tensor, Params], Tensor]

_EPS = 1e-4


def constrain_sls(raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    e_r = F.softplus(raw[..., 0]) + _EPS
    tau_sigma = F.softplus(raw[..., 1]) + _EPS
    tau_eps = tau_sigma + F.softplus(raw[..., 2]) + _EPS
    return e_r, tau_sigma, tau_eps


def interp_value(t: float, t_nodes: Tensor, values: Tensor) -> Tensor:
    m = t_nodes.shape[0]
    span = float(t_nodes[-1] - t_nodes[0])
    h = span / (m - 1)
    clamped = min(max(t, float(t_nodes[0])), float(t_nodes[-1]))
    pos = (clamped - float(t_nodes[0])) / h
    k = int(pos)
    if k > m - 2:
        k = m - 2
    frac = pos - k
    return values[:, k] * (1.0 - frac) + values[:, k + 1] * frac


def central_difference(values: Tensor, h: float) -> Tensor:
    deriv = torch.empty_like(values)
    deriv[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * h)
    deriv[:, 0] = (values[:, 1] - values[:, 0]) / h
    deriv[:, -1] = (values[:, -1] - values[:, -2]) / h
    return deriv


def node_spacing(t_nodes: Tensor) -> float:
    return float(t_nodes[-1] - t_nodes[0]) / (t_nodes.shape[0] - 1)


def make_sls_dynamics(t_nodes: Tensor) -> Dynamics:
    def f(t: float, z: Tensor, params: Params) -> Tensor:
        sigma, dsigma, e_r, tau_sigma, tau_eps = params
        s = interp_value(t, t_nodes, sigma)
        ds = interp_value(t, t_nodes, dsigma)
        return (s + tau_sigma * ds - e_r * z) / (e_r * tau_eps)

    return f


def make_polynomial_dynamics() -> Dynamics:
    def f(t: float, z: Tensor, params: Params) -> Tensor:
        a, b, c, d = params
        return a * z**3 + b * z**2 + c * z + d

    return f


def make_windkessel_dynamics(t_nodes: Tensor) -> Dynamics:
    def f(t: float, z: Tensor, params: Params) -> Tensor:
        sigma, resistance, compliance = params
        s = interp_value(t, t_nodes, sigma)
        return (s - z / resistance) / compliance

    return f
