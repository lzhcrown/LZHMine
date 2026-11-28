# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Concurrent teacher/student policy adapted from legbot_lab WF-CTS-MOE.

Only the student encoder and shared actor are required for deployment. Inputs use
term-major history: each observation term contains oldest-to-newest frames.
"""

from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal

from .frozen_terrain_encoder import (
    FrozenTerrainEncoder, FrozenTerrainTeacherEncoder, TERRAIN_LATENT_DIM, pretrained_scan_slice,
)
from .moe import L2Norm, MLP, MoE, MoEActor, MoECritic, SimNorm, TaskMLPCritic
from .teacher_map_attention import MapAttentionTeacherEncoder, TeacherMapAttention


class StudentEncoderInputs(nn.Module):
    """Append the current frozen map code to the term-major history when configured."""

    def __init__(self, terrain_encoder):
        super().__init__()
        self.terrain_encoder = terrain_encoder

    def fused_input(self, history: torch.Tensor, height_scan: Optional[torch.Tensor]):
        if self.terrain_encoder is None:
            return history
        if height_scan is None:
            raise ValueError("The map-conditioned student requires a current height_scan")
        if height_scan.dim() != 2 or height_scan.size(0) != history.size(0):
            raise ValueError("height_scan must contain one scan per observation history")
        geometry = self.terrain_encoder(height_scan.detach())
        return torch.cat((history, geometry), dim=-1)


class StudentMLPEncoder(StudentEncoderInputs):
    """One normalized latent encoder, without experts or a routing network."""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation="elu", norm_type="l2norm", terrain_encoder=None):
        super().__init__(terrain_encoder)
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation)
        self.norm_layer = L2Norm() if norm_type == "l2norm" else SimNorm()

    def forward(self, observations: torch.Tensor, height_scan: Optional[torch.Tensor] = None):
        latent = self.norm_layer(self.mlp(self.fused_input(observations, height_scan)))
        return latent, observations.new_empty((observations.shape[0], 0))

    def get_expert_probabilities(
            self, observations: torch.Tensor, height_scan: Optional[torch.Tensor] = None) -> torch.Tensor:
        return observations.new_empty((observations.shape[0], 0))


class StudentMoEEncoder(StudentEncoderInputs):
    def __init__(self, expert_num, input_dim, hidden_dims, output_dim,
                 activation="elu", norm_type="l2norm", terrain_encoder=None):
        super().__init__(terrain_encoder)
        self.moe = MoE(expert_num, input_dim, hidden_dims, output_dim, activation)
        self.norm_layer = L2Norm() if norm_type == "l2norm" else SimNorm()

    def forward(self, observations: torch.Tensor, height_scan: Optional[torch.Tensor] = None):
        latent, weights = self.moe(self.fused_input(observations, height_scan))
        return self.norm_layer(latent), weights

    def get_expert_probabilities(
            self, observations: torch.Tensor, height_scan: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.moe.gating_network(self.fused_input(observations, height_scan))


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
        residual_enabled=False, residual_hidden_dims=(128, 64),
        residual_action_scales=None, critic_task_count=1,
        teacher_map_attention=False, observation_config=None, teacher_frozen_terrain=False,
        student_frozen_terrain=False,
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
        if not isinstance(residual_enabled, bool):
            raise ValueError("residual_enabled must be a boolean")
        if not isinstance(teacher_map_attention, bool):
            raise ValueError("teacher_map_attention must be a boolean")
        if not isinstance(teacher_frozen_terrain, bool):
            raise ValueError("teacher_frozen_terrain must be a boolean")
        if not isinstance(student_frozen_terrain, bool):
            raise ValueError("student_frozen_terrain must be a boolean")
        if teacher_frozen_terrain and teacher_map_attention:
            raise ValueError("teacher_frozen_terrain and teacher_map_attention are mutually exclusive")
        if residual_enabled and actor_type != "moe":
            raise ValueError("Action residual experiments require a MoE Actor")
        if not residual_enabled and residual_action_scales is not None:
            raise ValueError("Disabled residual must not configure action scales")
        if not isinstance(residual_hidden_dims, (list, tuple)) or not residual_hidden_dims or any(
                isinstance(width, bool) or not isinstance(width, int) or width < 1
                for width in residual_hidden_dims):
            raise ValueError("Residual hidden dimensions must be positive integers")
        for name, network_type, count in (
                ("student_encoder", student_encoder_type, expert_num),
                ("actor", actor_type, actor_expert_num),
                ("critic", critic_type, critic_expert_num)):
            allowed = ("mlp", "moe", "task_mlp") if name == "critic" else ("mlp", "moe")
            if network_type not in allowed:
                raise ValueError("%s_type must be one of %s" % (name, allowed))
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
        if type(critic_task_count) is not int or critic_task_count not in (1, 2):
            raise ValueError("critic_task_count must be 1 or 2")
        if critic_type == "task_mlp" and (critic_task_count != 2 or critic_expert_num != 0):
            raise ValueError("task_mlp requires critic_task_count=2 and critic_expert_num=0")
        self.critic_task_count = critic_task_count
        self.teacher_map_attention = teacher_map_attention
        self.teacher_frozen_terrain = teacher_frozen_terrain
        self.student_frozen_terrain = student_frozen_terrain
        self.residual_enabled = residual_enabled
        self.residual_hidden_dims = list(residual_hidden_dims)
        self.residual_action_scales = (list(residual_action_scales)
                                       if residual_action_scales is not None else None)

        indices = []
        offset = 0
        for width in observation_dims:
            indices.extend(range(offset + (history_length - 1) * width,
                                 offset + history_length * width))
            offset += history_length * width
        # Derived from metadata, so it need not be saved in the checkpoint.
        self.register_buffer("current_obs_indices", torch.tensor(indices, dtype=torch.long),
                             persistent=False)

        if teacher_frozen_terrain:
            self.teacher_encoder = FrozenTerrainTeacherEncoder(
                observation_config, num_critic_obs, latent_dim, teacher_encoder_hidden_dims,
                activation, norm_type,
            )
        else:
            self.teacher_encoder = nn.Sequential(
                MLP(num_critic_obs, latent_dim, teacher_encoder_hidden_dims, activation),
                L2Norm() if norm_type == "l2norm" else SimNorm(),
            )
        terrain_encoder = None
        student_input_dim = num_actor_obs
        if student_frozen_terrain:
            self.student_scan_slice = pretrained_scan_slice(observation_config, num_critic_obs)
            terrain_encoder = (self.teacher_encoder.terrain_encoder if teacher_frozen_terrain
                               else FrozenTerrainEncoder())
            student_input_dim += TERRAIN_LATENT_DIM

        # Keep this module name so existing MoE checkpoints retain their keys.
        if student_encoder_type == "moe":
            self.student_moe_encoder = StudentMoEEncoder(
                self.expert_num, student_input_dim, student_encoder_hidden_dims, latent_dim,
                activation, norm_type, terrain_encoder,
            )
        else:
            self.student_moe_encoder = StudentMLPEncoder(
                student_input_dim, student_encoder_hidden_dims, latent_dim, activation, norm_type, terrain_encoder,
            )
        if actor_type == "moe":
            self.actor = MoEActor(self.actor_expert_num, latent_dim + num_one_step_obs,
                                  actor_hidden_dims, num_actions, activation,
                                  residual_enabled=residual_enabled,
                                  num_current_obs=num_one_step_obs,
                                  residual_hidden_dims=residual_hidden_dims,
                                  residual_action_scales=residual_action_scales)
        else:
            self.actor = MLP(latent_dim + num_one_step_obs, num_actions,
                             actor_hidden_dims, activation)
        if critic_type == "moe":
            self.critic = MoECritic(self.critic_expert_num, latent_dim + num_critic_obs,
                                    critic_hidden_dims, activation, critic_task_count)
        elif critic_type == "task_mlp":
            self.critic = TaskMLPCritic(latent_dim + num_critic_obs, critic_hidden_dims, activation)
        else:
            self.critic = MLP(latent_dim + num_critic_obs, critic_task_count, critic_hidden_dims, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        if teacher_map_attention:
            self.teacher_encoder = MapAttentionTeacherEncoder(
                self.teacher_encoder, TeacherMapAttention(observation_config, num_critic_obs, latent_dim),
            )
        self.distribution = None
        self.actor_gate_weights = torch.empty(0)
        self.critic_gate_weights = torch.empty(0)
        self.coarse_action_mean = torch.empty(0)
        self.action_residual = torch.empty(0)

    def reset(self, dones=None):
        pass

    def forward(self, observations, height_scan=None):
        return self.act_inference(observations, height_scan)

    def student_height_scan(self, critic_obs):
        """Select only the current preprocessed scan for the training student."""
        if not self.student_frozen_terrain:
            return None
        if critic_obs is None or critic_obs.ndim != 2 or critic_obs.shape[1] != self.num_critic_obs:
            raise ValueError("Training the map-conditioned student requires the privileged frame with height_scan")
        return critic_obs[:, self.student_scan_slice]

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

    def action_mean_with_diagnostics(self, observations, critic_obs=None, is_teacher=False):
        """PPO mean path; student latents retain their stop-gradient boundary."""
        if is_teacher:
            if critic_obs is None:
                raise ValueError("Teacher actions require current privileged observations")
            latent = self.teacher_encoder(critic_obs)
        else:
            # PPO may train the shared actor on student trajectories, but must
            # never train the student encoder through its policy gradient.
            with torch.no_grad():
                latent, _ = self.student_moe_encoder(observations, self.student_height_scan(critic_obs))
        inputs = torch.cat((latent, self.get_current_observations(observations)), dim=-1)
        if self.actor_type == "moe":
            return self.actor.forward_with_diagnostics(inputs)
        mean = self.actor(inputs)
        return mean, inputs.new_empty((inputs.shape[0], 0)), mean, torch.zeros_like(mean)

    def act(self, observations, critic_obs=None, is_teacher=False):
        mean, weights, coarse, correction = self.action_mean_with_diagnostics(
            observations, critic_obs, is_teacher)
        self.actor_gate_weights = weights
        self.coarse_action_mean = coarse
        self.action_residual = correction
        self.distribution = Normal(mean, self.std.clamp_min(1.0e-6).expand_as(mean),
                                   validate_args=False)
        return self.distribution.sample()

    def act_inference(self, observations, height_scan=None):
        mean, _, _, _, _ = self.act_inference_with_diagnostics(observations, height_scan)
        return mean

    def act_inference_with_diagnostics(self, observations, height_scan=None):
        """One deployment forward for actions and all routing/residual diagnostics."""
        latent, student_weights = self.student_moe_encoder(observations, height_scan)
        inputs = torch.cat((latent, self.get_current_observations(observations)), dim=-1)
        if self.actor_type == "moe":
            mean, actor_weights, coarse, correction = self.actor.forward_with_diagnostics(inputs)
        else:
            mean = self.actor(inputs)
            actor_weights = inputs.new_empty((inputs.shape[0], 0))
            coarse, correction = mean, torch.zeros_like(mean)
        return mean, student_weights, actor_weights, coarse, correction

    def evaluate(self, observations, critic_obs, is_teacher=False, platform_mask=None):
        # The value function cannot train either latent encoder.
        with torch.no_grad():
            if is_teacher:
                latent = self.teacher_encoder(critic_obs)
            else:
                latent, _ = self.student_moe_encoder(observations, self.student_height_scan(critic_obs))
        inputs = torch.cat((latent.detach(), critic_obs), dim=-1)
        if self.critic_type == "moe":
            value, self.critic_gate_weights = self.critic.moe(inputs)
        else:
            self.critic_gate_weights = inputs.new_empty((inputs.shape[0], 0))
            if self.critic_type == "task_mlp":
                return self.critic(inputs, platform_mask)
            value = self.critic(inputs)
        if self.critic_task_count == 2:
            if (platform_mask is None or platform_mask.dtype != torch.bool
                    or platform_mask.shape != (value.shape[0],)):
                raise ValueError("Two task values require one boolean platform label per sample")
            return torch.where(platform_mask[:, None], value[:, 1:2], value[:, :1])
        return value

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_expert_probabilities(self, observations, height_scan=None):
        return self.student_moe_encoder.get_expert_probabilities(observations, height_scan)

    def get_actor_expert_probabilities(self, observations, critic_obs=None, is_teacher=False, height_scan=None):
        """Actor routing, distinct from the student's representation routing."""
        if self.actor_type != "moe":
            return observations.new_empty((observations.shape[0], 0))
        if is_teacher:
            if critic_obs is None:
                raise ValueError("Teacher routing requires current privileged observations")
            latent = self.teacher_encoder(critic_obs)
        else:
            with torch.no_grad():
                if height_scan is None and critic_obs is not None:
                    height_scan = self.student_height_scan(critic_obs)
                latent, _ = self.student_moe_encoder(observations, height_scan)
        inputs = torch.cat((latent, self.get_current_observations(observations)), dim=-1)
        return self.actor.get_expert_probabilities(inputs)
