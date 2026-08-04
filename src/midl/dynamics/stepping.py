from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from midl.materials.constitutive import Dynamics, Params

State = tuple[Tensor, ...]
StateDeriv = Callable[[float, State], State]


def _rk4_state(dynamics: Dynamics, t: float, z: Tensor, h: float, params: Params) -> Tensor:
    k1 = dynamics(t, z, params)
    k2 = dynamics(t + h / 2, z + h / 2 * k1, params)
    k3 = dynamics(t + h / 2, z + h / 2 * k2, params)
    k4 = dynamics(t + h, z + h * k3, params)
    return z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint_unrolled(dynamics: Dynamics, z0: Tensor, t_nodes: Tensor, params: Params) -> Tensor:
    m = int(t_nodes.shape[0])
    h = float(t_nodes[-1] - t_nodes[0]) / (m - 1)
    z = z0
    nodes = [z]
    for k in range(m - 1):
        z = _rk4_state(dynamics, float(t_nodes[k]), z, h, params)
        nodes.append(z)
    return torch.stack(nodes, dim=1)


def _axpy(state: State, k: State, h: float) -> State:
    return tuple(s + h * ki for s, ki in zip(state, k, strict=True))


def _rk4_tuple(deriv: StateDeriv, t: float, state: State, h: float) -> State:
    k1 = deriv(t, state)
    k2 = deriv(t + h / 2, _axpy(state, k1, h / 2))
    k3 = deriv(t + h / 2, _axpy(state, k2, h / 2))
    k4 = deriv(t + h, _axpy(state, k3, h))
    return tuple(
        s + (h / 6) * (a + 2 * b + 2 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )


def _adjoint_segment(
    dynamics: Dynamics,
    t1: float,
    z1: Tensor,
    a1: Tensor,
    g1: list[Tensor],
    params: Params,
    step: float,
) -> tuple[Tensor, list[Tensor]]:
    def deriv(t: float, st: State) -> State:
        z, a = st[0], st[1]
        with torch.enable_grad():
            z_ = z.detach().requires_grad_(True)
            p_ = tuple(p.detach().requires_grad_(True) for p in params)
            fz = dynamics(t, z_, p_)
            grads = torch.autograd.grad(fz, (z_, *p_), grad_outputs=a, allow_unused=True)
        dz = fz.detach()
        da = -grads[0] if grads[0] is not None else torch.zeros_like(a)
        dg = [
            (-grads[i + 1] if grads[i + 1] is not None else torch.zeros_like(params[i]))
            for i in range(len(params))
        ]
        return (dz, da, *dg)

    with torch.no_grad():
        new = _rk4_tuple(deriv, t1, (z1, a1, *g1), step)
    return new[1], list(new[2:])


class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, dynamics: Dynamics, t_nodes: Tensor, z0: Tensor, *params: Tensor
    ) -> Tensor:
        m = int(t_nodes.shape[0])
        h = float(t_nodes[-1] - t_nodes[0]) / (m - 1)
        with torch.no_grad():
            z = z0
            nodes = [z]
            for k in range(m - 1):
                z = _rk4_state(dynamics, float(t_nodes[k]), z, h, params)
                nodes.append(z)
            traj = torch.stack(nodes, dim=1)
        ctx.dynamics = dynamics
        ctx.save_for_backward(t_nodes, traj, *params)
        return traj

    @staticmethod
    def backward(ctx: Any, grad_traj: Tensor) -> tuple[Any, ...]:
        dynamics = ctx.dynamics
        t_nodes, _traj, *params = ctx.saved_tensors
        m = int(t_nodes.shape[0])
        h = float(t_nodes[-1] - t_nodes[0]) / (m - 1)
        traj = _traj
        a = torch.zeros_like(grad_traj[:, 0])
        gparams = [torch.zeros_like(p) for p in params]
        for k in range(m - 1, 0, -1):
            a = a + grad_traj[:, k]
            a, gparams = _adjoint_segment(
                dynamics, float(t_nodes[k]), traj[:, k], a, gparams, tuple(params), -h
            )
        return (None, None, None, *gparams)


def odeint_adjoint(dynamics: Dynamics, z0: Tensor, t_nodes: Tensor, params: Params) -> Tensor:
    return ODEAdjoint.apply(dynamics, t_nodes, z0, *params)
