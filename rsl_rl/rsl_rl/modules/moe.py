# SPDX-License-Identifier: BSD-3-Clause
"""WF mixture of experts, implemented with plain PyTorch for Isaac Gym.

The experts share a feature trunk, followed by independent grouped linear
heads. The gate combines latent features; its indices do not name gaits.
Reference: Robot-Nav/legbot_lab, WF-CTS-MOE, daff9fb.
"""

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

    def __init__(self, expert_num, input_dim, hidden_dims, num_actions,
                 activation="elu"):
        super().__init__()
        self.moe = MoE(expert_num, input_dim, hidden_dims, num_actions, activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.moe(inputs)
        return mean

    def get_expert_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.moe.gating_network(inputs)
