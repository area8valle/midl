from __future__ import annotations

from torch import Tensor

from midl.materials.constitutive import Dynamics, Params, central_difference, node_spacing


def physics_residual(traj: Tensor, t_nodes: Tensor, dynamics: Dynamics, params: Params) -> Tensor:
    m = t_nodes.shape[0]
    h = node_spacing(t_nodes)
    deriv = central_difference(traj, h)
    total = traj.new_zeros(traj.shape[0])
    for k in range(m):
        rhs = dynamics(float(t_nodes[k]), traj[:, k], params)
        total = total + (deriv[:, k] - rhs) ** 2
    return total / m
