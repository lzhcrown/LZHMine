# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Concurrent teacher/student policy adapted from legbot_lab WF-CTS-MOE.

Only the student encoder and shared actor are required for deployment. Inputs use
term-major history: each observation term contains oldest-to-newest frames.
"""

import torch
from torch import nn
from torch.distributions import Normal

from .moe import L2Norm, MLP, MoE, MoEActor, MoECritic, SimNorm


class StudentMLPEncoder(nn.Module):
    """One normalized latent encoder, without experts or a routing network."""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation="elu", norm_type="l2norm"):
        super().__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation)
        self.norm_layer = L2Norm() if norm_type == "l2norm" else SimNorm()

    def forward(self, observations):
        latent = self.norm_layer(self.mlp(observations))
        return latent, self.get_expert_probabilities(observations)

    def get_expert_probabilities(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.new_empty((observations.shape[0], 0))


class StudentMoEEncoder(nn.Module):
    def __init__(self, expert_num, input_dim, hidden_dims, output_dim,
                 activation="elu", norm_type="l2norm"):
        super().__init__()
        self.moe = MoE(expert_num, input_dim, hidden_dims, output_dim, activation)
        self.norm_layer = L2Norm() if norm_type == "l2norm" else SimNorm()

    def forward(self, observations):
        latent, weights = self.moe(observations)
        return self.norm_layer(latent), weights

    def get_expert_probabilities(self, observations: torch.Tensor) -> torch.Tensor:
        return self.moe.gating_network(observations)


class ActorCriticMoECTS(nn.Module):
    is_recurrent = False

    def __init__(
        self, num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
        history_length=4, observation_dims=(3, 3, 3, 16, 16, 16),
        actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
        teacher_encoder_hidden_dims=(512, 256),
        student_encoder_hidden_dims=(512, 256, 256), expert_num=8,
        latent_dim=32, activation="elu", init_noise_std=1.0,
        norm_type="l2norm", noise_std_type="scalar", state_dependent_std=False,
        actor_obs_normalization=False, critic_obs_normalization=False,
        actor_type="mlp", actor_expert_num=8,
        student_encoder_type="moe", critic_type="mlp", critic_expert_num=8,
    ):
        super().__init__()
        if history_length < 1 or sum(observation_dims) != num_one_step_obs:
            raise ValueError("History length/observation term dimensions are inconsistent")
        if any(width < 1 for width in observation_dims):
            raise ValueError("Observation terms must have positive widths")
        if num_actor_obs != num_one_step_obs * history_length:
            raise ValueError("Actor input must contain exactly history_length complete frames")
        if norm_type not in ("l2norm", "simnorm"):
            raise ValueError("Unsupported latent normalization: %s" % norm_type)
        if norm_type == "simnorm" and latent_dim % 8:
            raise ValueError("SimNorm requires a latent dimension divisible by eight")
        if init_noise_std <= 0:
            raise ValueError("Initial action standard deviation must be positive")
        if noise_std_type != "scalar" or state_dependent_std:
            raise ValueError("This WF policy uses a shared scalar standard deviation per action")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("WF baseline disables empirical observation normalization")
        for name, network_type, count in (
                ("student_encoder", student_encoder_type, expert_num),
                ("actor", actor_type, actor_expert_num),
                ("critic", critic_type, critic_expert_num)):
            if network_type not in ("mlp", "moe"):
                raise ValueError("%s_type must be 'mlp' or 'moe'" % name)
            if (isinstance(count, bool) or int(count) != count or count < 0
                    or (network_type == "moe" and count < 1)):
                raise ValueError("%s expert count must be a positive integer for MoE" % name)

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_single_obs = num_one_step_obs
        self.num_actions = num_actions
        self.history_length = history_length
        self.observation_dims = list(observation_dims)
        self.student_encoder_type = student_encoder_type
        self.expert_num = int(expert_num) if student_encoder_type == "moe" else 0
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        # Missing actor_type in an older checkpoint retains the original MLP.
        self.actor_type = actor_type
        self.actor_expert_num = int(actor_expert_num) if actor_type == "moe" else 0
        self.critic_type = critic_type
        self.critic_expert_num = int(critic_expert_num) if critic_type == "moe" else 0

        indices = []
        offset = 0
        for width in observation_dims:
            indices.extend(range(offset + (history_length - 1) * width,
                                 offset + history_length * width))
            offset += history_length * width
        # Derived from metadata, so it need not be saved in the checkpoint.
        self.register_buffer("current_obs_indices", torch.tensor(indices, dtype=torch.long),
                             persistent=False)

        self.teacher_encoder = nn.Sequential(
            MLP(num_critic_obs, latent_dim, teacher_encoder_hidden_dims, activation),
            L2Norm() if norm_type == "l2norm" else SimNorm(),
        )
        # Keep this module name so existing MoE checkpoints retain their keys.
        if student_encoder_type == "moe":
            self.student_moe_encoder = StudentMoEEncoder(
                self.expert_num, num_actor_obs, student_encoder_hidden_dims, latent_dim,
                activation, norm_type,
            )
        else:
            self.student_moe_encoder = StudentMLPEncoder(
                num_actor_obs, student_encoder_hidden_dims, latent_dim, activation, norm_type,
            )
        if actor_type == "moe":
            self.actor = MoEActor(self.actor_expert_num, latent_dim + num_one_step_obs,
                                  actor_hidden_dims, num_actions, activation)
        else:
            self.actor = MLP(latent_dim + num_one_step_obs, num_actions,
                             actor_hidden_dims, activation)
        if critic_type == "moe":
            self.critic = MoECritic(self.critic_expert_num, latent_dim + num_critic_obs,
                                    critic_hidden_dims, activation)
        else:
            self.critic = MLP(latent_dim + num_critic_obs, 1, critic_hidden_dims, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.actor_gate_weights = torch.empty(0)
        self.critic_gate_weights = torch.empty(0)

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        return self.act_inference(observations)

    def get_current_observations(self, observations):
        return observations.index_select(-1, self.current_obs_indices)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, critic_obs=None, is_teacher=False):
        if is_teacher:
            if critic_obs is None:
                raise ValueError("Teacher actions require current privileged observations")
            latent = self.teacher_encoder(critic_obs)
        else:
            # PPO may train the shared actor on student trajectories, but must
            # never train the student encoder through its policy gradient.
            with torch.no_grad():
                latent, _ = self.student_moe_encoder(observations)
        inputs = torch.cat((latent, self.get_current_observations(observations)), dim=-1)
        if self.actor_type == "moe":
            mean, self.actor_gate_weights = self.actor.moe(inputs)
        else:
            mean = self.actor(inputs)
            self.actor_gate_weights = inputs.new_empty((inputs.shape[0], 0))
        self.distribution = Normal(mean, self.std.clamp_min(1.0e-6).expand_as(mean),
                                   validate_args=False)
        return self.distribution.sample()

    def act_inference(self, observations):
        latent, _ = self.student_moe_encoder(observations)
        return self.actor(torch.cat((latent, self.get_current_observations(observations)), dim=-1))

    def evaluate(self, observations, critic_obs, is_teacher=False):
        # The value function cannot train either latent encoder.
        with torch.no_grad():
            if is_teacher:
                latent = self.teacher_encoder(critic_obs)
            else:
                latent, _ = self.student_moe_encoder(observations)
        inputs = torch.cat((latent.detach(), critic_obs), dim=-1)
        if self.critic_type == "moe":
            value, self.critic_gate_weights = self.critic.moe(inputs)
        else:
            value = self.critic(inputs)
            self.critic_gate_weights = inputs.new_empty((inputs.shape[0], 0))
        return value

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_expert_probabilities(self, observations):
        return self.student_moe_encoder.get_expert_probabilities(observations)

    def get_actor_expert_probabilities(self, observations, critic_obs=None, is_teacher=False):
        """Actor routing, distinct from the student's representation routing."""
        if self.actor_type != "moe":
            return observations.new_empty((observations.shape[0], 0))
        if is_teacher:
            if critic_obs is None:
                raise ValueError("Teacher routing requires current privileged observations")
            latent = self.teacher_encoder(critic_obs)
        else:
            with torch.no_grad():
                latent, _ = self.student_moe_encoder(observations)
        inputs = torch.cat((latent, self.get_current_observations(observations)), dim=-1)
        return self.actor.get_expert_probabilities(inputs)
