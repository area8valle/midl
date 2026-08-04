from __future__ import annotations

from dataclasses import dataclass

import gin
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from midl.dynamics.stepping import odeint_adjoint, odeint_unrolled
from midl.kinematics.clinical import ClinicalEncoder
from midl.kinematics.fusion import CrossAttentionFusion
from midl.kinematics.imaging import RadiographEncoder
from midl.materials.constitutive import (
    Dynamics,
    Params,
    central_difference,
    constrain_sls,
    make_polynomial_dynamics,
    make_sls_dynamics,
    make_windkessel_dynamics,
    node_spacing,
)
from midl.observables.progression import ProgressionHead
from midl.observables.survival import CoxHead

MECHANISMS = ("sls", "polynomial", "windkessel", "fc", "none")
_EPS = 1e-4


def build_dynamics(name: str, t_nodes: Tensor) -> Dynamics:
    if name == "sls":
        return make_sls_dynamics(t_nodes)
    if name == "polynomial":
        return make_polynomial_dynamics()
    if name == "windkessel":
        return make_windkessel_dynamics(t_nodes)
    raise ValueError(f"no dynamics for mechanism {name}")


@dataclass
class MechanismAux:
    name: str
    t_nodes: Tensor | None = None
    trajectory: Tensor | None = None
    params: Params | None = None
    e_r: Tensor | None = None
    tau_sigma: Tensor | None = None
    tau_eps: Tensor | None = None


@dataclass
class ModelOutput:
    progression_logit: Tensor
    cox_risk: Tensor
    fused: Tensor
    z_bio: Tensor | None
    aux: MechanismAux


@gin.configurable
class MIDL(nn.Module):
    t_nodes: Tensor

    def __init__(
        self,
        mechanism: str = "sls",
        use_clinical: bool = True,
        pseudo_steps: int = 16,
        img_dim: int = 1792,
        clin_dim: int = 64,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        if mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {mechanism}")
        self.mechanism = mechanism
        self.use_clinical = use_clinical
        self.pseudo_steps = pseudo_steps
        self.use_adjoint = True
        self.register_buffer("t_nodes", torch.linspace(0.0, 1.0, pseudo_steps))

        self.radiograph = RadiographEncoder(out_dim=img_dim)
        self.clinical = ClinicalEncoder(out_dim=clin_dim) if use_clinical else None
        feat_dim = img_dim + (clin_dim if use_clinical else 0)

        raw_dim = {"sls": 3, "polynomial": 4, "windkessel": 2, "fc": 4, "none": 0}[mechanism]
        self.param_head: nn.Module | None = None
        self.loading: nn.Module | None = None
        self.fc_bottleneck: nn.Module | None = None
        if mechanism in ("sls", "polynomial", "windkessel"):
            self.param_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, raw_dim)
            )
            self._init_param_bias(mechanism)
            if mechanism in ("sls", "windkessel"):
                self.loading = nn.Linear(img_dim, pseudo_steps)
        elif mechanism == "fc":
            self.fc_bottleneck = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, 4)
            )

        query_dim = img_dim if mechanism == "none" else 4
        context_dims = (img_dim,) + ((clin_dim,) if use_clinical else ())
        self.fusion = CrossAttentionFusion(query_dim, context_dims, embed_dim=embed_dim)
        self.progression = ProgressionHead(embed_dim)
        self.cox = CoxHead(embed_dim)

    def _init_param_bias(self, mechanism: str) -> None:
        assert isinstance(self.param_head, nn.Sequential)
        final = self.param_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        if mechanism == "sls":
            final.bias.data = torch.tensor([10.0, 100.0, 100.0])
        elif mechanism == "windkessel":
            final.bias.data = torch.tensor([5.0, 5.0])
        else:
            nn.init.zeros_(final.bias)

    def encode(self, image: Tensor, clinical: Tensor) -> tuple[Tensor, Tensor | None, Tensor]:
        z_img = self.radiograph(image)
        z_clin = self.clinical(clinical) if self.clinical is not None else None
        feat = z_img if z_clin is None else torch.cat([z_img, z_clin], dim=1)
        return z_img, z_clin, feat

    def _integrate(self, dynamics: Dynamics, z0: Tensor, t_nodes: Tensor, params: Params) -> Tensor:
        if self.use_adjoint:
            return odeint_adjoint(dynamics, z0, t_nodes, params)
        return odeint_unrolled(dynamics, z0, t_nodes, params)

    def mechanism_forward(self, z_img: Tensor, feat: Tensor) -> tuple[Tensor | None, MechanismAux]:
        t_nodes = self.t_nodes
        z0 = feat.new_zeros(feat.shape[0])
        if self.mechanism == "sls":
            assert self.param_head is not None and self.loading is not None
            e_r, tau_sigma, tau_eps = constrain_sls(self.param_head(feat))
            sigma = self.loading(z_img)
            dsigma = central_difference(sigma, node_spacing(t_nodes))
            params: Params = (sigma, dsigma, e_r, tau_sigma, tau_eps)
            traj = self._integrate(make_sls_dynamics(t_nodes), z0, t_nodes, params)
            z_bio = torch.stack([traj[:, -1], e_r, tau_sigma, tau_eps], dim=1)
            return z_bio, MechanismAux("sls", t_nodes, traj, params, e_r, tau_sigma, tau_eps)
        if self.mechanism == "polynomial":
            assert self.param_head is not None
            raw = self.param_head(feat)
            params = (raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3])
            traj = self._integrate(make_polynomial_dynamics(), z0, t_nodes, params)
            z_bio = torch.stack([traj[:, -1], raw[:, 0], raw[:, 1], raw[:, 2]], dim=1)
            return z_bio, MechanismAux("polynomial", t_nodes, traj, params)
        if self.mechanism == "windkessel":
            assert self.param_head is not None and self.loading is not None
            raw = self.param_head(feat)
            resistance = F.softplus(raw[:, 0]) + _EPS
            compliance = F.softplus(raw[:, 1]) + _EPS
            load = self.loading(z_img)
            params = (load, resistance, compliance)
            traj = self._integrate(make_windkessel_dynamics(t_nodes), z0, t_nodes, params)
            z_bio = torch.stack(
                [traj[:, -1], resistance, compliance, torch.zeros_like(resistance)], dim=1
            )
            return z_bio, MechanismAux("windkessel", t_nodes, traj, params)
        if self.mechanism == "fc":
            assert self.fc_bottleneck is not None
            return self.fc_bottleneck(feat), MechanismAux("fc")
        return None, MechanismAux("none")

    def forward(self, image: Tensor, clinical: Tensor) -> ModelOutput:
        z_img, z_clin, feat = self.encode(image, clinical)
        z_bio, aux = self.mechanism_forward(z_img, feat)
        contexts = [z_img] + ([z_clin] if z_clin is not None else [])
        query = z_img if z_bio is None else z_bio
        fused = self.fusion(query, contexts)
        return ModelOutput(self.progression(fused), self.cox(fused), fused, z_bio, aux)
