# SPDX-License-Identifier: BSD-3-Clause
"""WF mixture of experts, implemented with plain PyTorch for Isaac Gym.

The experts share a feature trunk, followed by independent grouped linear
heads. The gate combines latent features; its indices do not name gaits.
Reference: Robot-Nav/legbot_lab, WF-CTS-MOE, daff9fb.
"""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def get_activation(name):
    activations = {
        "elu": nn.ELU, "selu": nn.SELU, "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError("Unsupported CTS activation: %s" % name)
    return activations[name]()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation="elu",
                 last_activation=None):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), get_activation(activation)])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        if last_activation is not None:
            layers.append(get_activation(last_activation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=-1)


class SimNorm(nn.Module):
    def forward(self, x):
        return F.softmax(x.reshape(*x.shape[:-1], -1, 8), dim=-1).reshape_as(x)


class Experts(nn.Module):
    def __init__(self, expert_num, input_dim, backbone_hidden_dims,
                 expert_hidden_dim, output_dim, activation="elu"):
        super().__init__()
        self.expert_num = expert_num
        self.output_dim = output_dim
        self.backbone = MLP(
            input_dim, expert_num * expert_hidden_dim, backbone_hidden_dims,
            activation, last_activation=activation,
        )
        self.experts = nn.Conv1d(
            expert_num * expert_hidden_dim, expert_num * output_dim,
            kernel_size=1, groups=expert_num,
        )

    def forward(self, x):
        features = self.backbone(x).unsqueeze(-1)
        return self.experts(features).squeeze(-1).reshape(
            -1, self.expert_num, self.output_dim
        )


class MoE(nn.Module):
    def __init__(self, expert_num, input_dim, hidden_dims, output_dim,
                 activation="elu"):
        super().__init__()
        if expert_num < 1 or not hidden_dims:
            raise ValueError("MoE requires experts and at least one hidden layer")
        self.experts = Experts(
            expert_num, input_dim, hidden_dims[:-1], hidden_dims[-1],
            output_dim, activation,
        )
        self.gating_network = nn.Sequential(
            MLP(input_dim, expert_num, hidden_dims, activation),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        weights = self.gating_network(x)
        expert_latents = self.experts(x)
        return (weights.unsqueeze(-1) * expert_latents).sum(dim=1), weights


class MoEActor(nn.Module):
    """Dense action-mean mixture with the same expert layout as the student.

    Experts share a feature trunk and have independent output heads. The gate
    has its own MLP; its softmax weights combine all expert action means.
    Unlike the student latent, actions are not L2-normalized. Exploration is
    applied once to the combined mean by ActorCriticMoECTS.
    """

    __constants__ = ["residual_enabled", "num_current_obs"]

    def __init__(self, expert_num, input_dim, hidden_dims, num_actions,
                 activation="elu", residual_enabled=False, num_current_obs=0,
                 residual_hidden_dims=(128, 64), residual_action_scales=None):
        super().__init__()
        self.residual_enabled = residual_enabled
        self.num_current_obs = int(num_current_obs)
        if residual_enabled:
            if not 0 < self.num_current_obs <= input_dim:
                raise ValueError("Residual Actor requires the current observation width")
            if not residual_hidden_dims or any(
                    isinstance(width, bool) or not isinstance(width, int) or width < 1
                    for width in residual_hidden_dims):
                raise ValueError("Residual hidden dimensions must be positive integers")
            if residual_action_scales is None or len(residual_action_scales) != num_actions:
                raise ValueError("Residual action scales must have one entry per action")
            if any(isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0
                   for scale in residual_action_scales):
                raise ValueError("Residual action scales must be finite and strictly positive")
        self.moe = MoE(expert_num, input_dim, hidden_dims, num_actions, activation)
        # Keep actor.moe.* intact so legacy checkpoints retain every original key.
        # The disabled branch creates no parameters and consumes no random draws.
        if residual_enabled:
            # Networks are constructed on CPU before policy.to(device). Extra
            # residual draws must not shift shared Critic initialization or the
            # first environment reset, whose curriculum state can persist even
            # after initialize_from later reseeds and resets the environment.
            with torch.random.fork_rng(devices=[]):
                self.residual = MLP(self.num_current_obs + num_actions, num_actions,
                                    residual_hidden_dims, activation)
            nn.init.zeros_(self.residual.network[-1].weight)
            nn.init.zeros_(self.residual.network[-1].bias)
            self.register_buffer("residual_action_scales", torch.tensor(
                residual_action_scales, dtype=torch.float32))
        else:
            self.residual = nn.Identity()
            self.register_buffer("residual_action_scales", torch.empty(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean, _, _, _ = self.forward_with_diagnostics(inputs)
        return mean

    def forward_with_diagnostics(
            self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final mean, Actor gate, coarse mean, and bounded correction."""
        coarse, weights = self.moe(inputs)
        if self.residual_enabled:
            current = inputs[:, -self.num_current_obs:]
            raw = self.residual(torch.cat((current, coarse), dim=-1))
            correction = self.residual_action_scales * torch.tanh(raw)
            return coarse + correction, weights, coarse, correction
        return coarse, weights, coarse, torch.zeros_like(coarse)

    def get_expert_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.moe.gating_network(inputs)


class MoECritic(nn.Module):
    """Dense mixture of per-task value estimates with a shared router."""

    def __init__(self, expert_num, input_dim, hidden_dims, activation="elu", task_count=1):
        super().__init__()
        self.moe = MoE(expert_num, input_dim, hidden_dims, task_count, activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value, _ = self.moe(inputs)
        return value

    def get_expert_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.moe.gating_network(inputs)
