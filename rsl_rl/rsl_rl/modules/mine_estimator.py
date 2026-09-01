"""Gated modal dual-encoder used by the Nezha MINE policy.

The source encoder consumes proprioceptive history available on the robot.  The
target encoder is training-only and consumes a configured simulator target
view.  A soft modality gate blends learnable modal prototypes before the
source latent is passed to the policy.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def _mlp(input_dim: int, hidden_dims: List[int], output_dim: int, activation: str) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), _activation(activation)))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class MINEEstimator(nn.Module):
    """Modality-gated source/target estimator.

    Observation histories are flattened and ordered newest-first throughout
    training, play, export, MuJoCo and deployment.  The target encoder can
    consume either the full privileged frame or its actor-observation prefix;
    the latter reproduces wheel_gym_CQ's ``is_privileged_obs=False`` experiment.
    """

    WHEEL_MODE = 0
    LEG_MODE = 1
    HYBRID_MODE = 2

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_one_step_privileged_obs: int,
        estimation_target_indices: List[int],
        is_privileged_obs: bool = True,
        enc_hidden_dims: List[int] = [256, 128, 64],
        tar_hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 16,
        num_modes: int = 3,
        gate_hidden_dim: int = 64,
        num_prototypes: int = 32,
        temperature: float = 3.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        mode_loss_coef: float = 0.5,
        mode_semantic_cfg: Dict = None,
        activation: str = "elu",
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if temporal_steps < 1:
            raise ValueError("temporal_steps must be positive")
        if not estimation_target_indices:
            raise ValueError("estimation_target_indices must not be empty")
        if max(estimation_target_indices) >= num_one_step_privileged_obs:
            raise ValueError("An estimation target index is outside one privileged frame")

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_privileged_obs = num_one_step_privileged_obs
        self.is_privileged_obs = bool(is_privileged_obs)
        self.num_est_prob = len(estimation_target_indices)
        self.num_latent = latent_dim
        self.num_modes = num_modes
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        semantic_cfg = mode_semantic_cfg or {}
        self.mode_semantic_enabled = bool(semantic_cfg.get("enabled", False))
        self.mode_semantic_loss_coef = float(semantic_cfg.get("loss_coef", 0.0))
        self.mode_loss_coef = float(
            semantic_cfg.get("mode_loss_coef", mode_loss_coef)
        )
        self.mode_command_slice = list(semantic_cfg.get("command_slice", [6, 9]))
        self.mode_dof_vel_slice = list(semantic_cfg.get("dof_vel_slice", [21, 37]))
        self.mode_action_slice = list(semantic_cfg.get("action_slice", [49, 65]))
        self.mode_wheel_dof_indices = list(
            semantic_cfg.get("wheel_dof_indices", [3, 7, 11, 15])
        )
        self.mode_leg_dof_indices = list(
            semantic_cfg.get(
                "leg_dof_indices",
                [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
            )
        )
        self.wheel_activity_low = float(
            semantic_cfg.get("wheel_activity_low", 0.05)
        )
        self.wheel_activity_high = float(
            semantic_cfg.get("wheel_activity_high", 0.20)
        )
        self.leg_activity_low = float(semantic_cfg.get("leg_activity_low", 0.03))
        self.leg_activity_high = float(
            semantic_cfg.get("leg_activity_high", 0.12)
        )
        self.static_command_threshold = float(
            semantic_cfg.get("static_command_threshold", 0.05)
        )
        self.leg_action_delta_weight = float(
            semantic_cfg.get("leg_action_delta_weight", 0.70)
        )
        self.leg_velocity_weight = float(
            semantic_cfg.get("leg_velocity_weight", 0.30)
        )
        if num_modes != 3 and self.mode_semantic_enabled:
            raise ValueError("Semantic mode anchoring requires exactly three modes")
        if self.wheel_activity_high <= self.wheel_activity_low:
            raise ValueError("wheel_activity_high must exceed wheel_activity_low")
        if self.leg_activity_high <= self.leg_activity_low:
            raise ValueError("leg_activity_high must exceed leg_activity_low")
        for semantic_slice in (
            self.mode_command_slice,
            self.mode_dof_vel_slice,
            self.mode_action_slice,
        ):
            if (
                len(semantic_slice) != 2
                or semantic_slice[0] < 0
                or semantic_slice[0] >= semantic_slice[1]
                or semantic_slice[1] > num_one_step_obs
            ):
                raise ValueError(
                    "Semantic observation slices must lie inside one observation frame"
                )
        self.register_buffer(
            "estimation_target_indices",
            torch.tensor(estimation_target_indices, dtype=torch.long),
        )

        self.encoder = _mlp(
            temporal_steps * num_one_step_obs,
            enc_hidden_dims,
            self.num_est_prob + latent_dim,
            activation,
        )
        target_input_dim = (
            num_one_step_privileged_obs
            if self.is_privileged_obs
            else num_one_step_obs
        )
        self.target = _mlp(
            target_input_dim,
            tar_hidden_dims,
            latent_dim,
            activation,
        )
        self.mode_gate = _mlp(num_one_step_obs, [gate_hidden_dim], num_modes, activation)
        self.mode_prototypes = nn.Parameter(torch.randn(num_modes, latent_dim))
        self.proto = nn.Embedding(num_prototypes, latent_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _flatten_history(self, obs_history: torch.Tensor) -> torch.Tensor:
        if obs_history.dim() == 3:
            return obs_history.reshape(obs_history.shape[0], -1)
        if obs_history.dim() != 2:
            raise ValueError("obs_history must have shape [batch, history * observation]")
        return obs_history

    def _source(
        self, obs_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history = self._flatten_history(obs_history)
        expected_dim = self.temporal_steps * self.num_one_step_obs
        if history.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected history width {expected_dim}, got {history.shape[-1]}"
            )
        parts = self.encoder(history.detach())
        estimate = parts[..., : self.num_est_prob]
        source_latent = parts[..., self.num_est_prob :]

        current_obs = history[..., : self.num_one_step_obs]
        mode_logits = self.mode_gate(current_obs)
        mode_probs = F.softmax(mode_logits, dim=-1)
        normalized_modes = F.normalize(self.mode_prototypes, dim=-1, p=2.0)
        modal_scale = torch.matmul(mode_probs, normalized_modes)
        latent = F.normalize(source_latent * modal_scale, dim=-1, p=2.0, eps=1e-8)
        return estimate, latent, mode_probs, mode_logits

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        estimate, latent, _, _ = self._source(obs_history)
        return estimate.detach(), latent.detach()

    def encode(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        estimate, latent, mode_probs, _ = self._source(obs_history)
        return estimate, latent, mode_probs

    @torch.no_grad()
    def get_latent(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        estimate, latent, _, _ = self._source(obs_history)
        return estimate, latent

    @torch.no_grad()
    def mode_probabilities(self, obs_history: torch.Tensor) -> torch.Tensor:
        _, _, probabilities, _ = self._source(obs_history)
        return probabilities

    @staticmethod
    def _soft_activity(activity: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return torch.clamp((activity - low) / (high - low), min=0.0, max=1.0)

    def _semantic_mode_targets(
        self, obs_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create causal wheel/leg/hybrid soft labels for estimator training.

        LZHMine histories are newest-first.  The gate still consumes only the
        current frame; the previous frame is used only by this training-time
        label builder to distinguish a fixed leg pose from active leg motion.
        """
        history = self._flatten_history(obs_history).detach()
        current_obs = history[..., : self.num_one_step_obs]
        if self.temporal_steps > 1:
            previous_obs = history[
                ..., self.num_one_step_obs : 2 * self.num_one_step_obs
            ]
        else:
            previous_obs = current_obs

        command = current_obs[
            :, self.mode_command_slice[0] : self.mode_command_slice[1]
        ]
        current_dof_vel = current_obs[
            :, self.mode_dof_vel_slice[0] : self.mode_dof_vel_slice[1]
        ]
        current_action = current_obs[
            :, self.mode_action_slice[0] : self.mode_action_slice[1]
        ]
        previous_action = previous_obs[
            :, self.mode_action_slice[0] : self.mode_action_slice[1]
        ]

        # Commanded wheel activity avoids classifying passive wheel rotation as
        # wheel propulsion.
        wheel_activity = current_action[:, self.mode_wheel_dof_indices].abs().mean(
            dim=-1
        )
        leg_action_delta = (
            current_action[:, self.mode_leg_dof_indices]
            - previous_action[:, self.mode_leg_dof_indices]
        ).abs().mean(dim=-1)
        leg_velocity = current_dof_vel[:, self.mode_leg_dof_indices].abs().mean(
            dim=-1
        )
        leg_activity = (
            self.leg_action_delta_weight * leg_action_delta
            + self.leg_velocity_weight * leg_velocity
        )

        wheel_active = self._soft_activity(
            wheel_activity, self.wheel_activity_low, self.wheel_activity_high
        )
        leg_active = self._soft_activity(
            leg_activity, self.leg_activity_low, self.leg_activity_high
        )
        target_mass = torch.stack(
            (
                wheel_active * (1.0 - leg_active),
                (1.0 - wheel_active) * leg_active,
                wheel_active * leg_active,
            ),
            dim=-1,
        )
        moving_confidence = target_mass.sum(dim=-1)
        targets = target_mass / moving_confidence.unsqueeze(-1).clamp_min(1e-6)

        command_activity = torch.norm(command, dim=-1)
        static_mask = (
            (command_activity < self.static_command_threshold)
            & (wheel_activity < self.wheel_activity_low)
            & (leg_activity < self.leg_activity_low)
        )
        valid_mask = (~static_mask) & (moving_confidence > 1e-6)
        confidence = moving_confidence * valid_mask.float()
        return targets, confidence, valid_mask, wheel_activity, leg_activity

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        learning_rate: float = None,
    ) -> Tuple[float, ...]:
        if learning_rate is not None:
            self.learning_rate = learning_rate
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

        privileged = next_critic_obs[..., : self.num_one_step_privileged_obs].detach()
        target_values = privileged.index_select(-1, self.estimation_target_indices)
        predicted_values, source_latent, mode_probs, mode_logits = self._source(
            obs_history
        )
        target_input = (
            privileged
            if self.is_privileged_obs
            else privileged[..., : self.num_one_step_obs]
        )
        target_latent = F.normalize(
            self.target(target_input), dim=-1, p=2.0, eps=1e-8
        )

        with torch.no_grad():
            self.proto.weight.copy_(F.normalize(self.proto.weight, dim=-1, p=2.0, eps=1e-8))

        source_scores = source_latent @ self.proto.weight.T
        target_scores = target_latent @ self.proto.weight.T
        with torch.no_grad():
            source_assignments = self._sinkhorn(source_scores)
            target_assignments = self._sinkhorn(target_scores)

        source_log_probs = F.log_softmax(source_scores / self.temperature, dim=-1)
        target_log_probs = F.log_softmax(target_scores / self.temperature, dim=-1)
        swap_loss = -0.5 * (
            source_assignments * target_log_probs
            + target_assignments * source_log_probs
        ).mean()
        estimation_loss = F.mse_loss(predicted_values, target_values)

        normalized_modes = F.normalize(self.mode_prototypes, dim=-1, p=2.0, eps=1e-8)
        similarity = source_latent.detach() @ normalized_modes.T
        mode_loss = F.cross_entropy(similarity, torch.argmax(mode_probs, dim=-1))

        semantic_loss = source_latent.new_zeros(())
        semantic_valid_ratio = source_latent.new_zeros(())
        semantic_mode_ratios = source_latent.new_zeros(3)
        mean_wheel_activity = source_latent.new_zeros(())
        mean_leg_activity = source_latent.new_zeros(())
        if self.mode_semantic_enabled and self.mode_semantic_loss_coef > 0.0:
            (
                semantic_targets,
                semantic_confidence,
                valid_mask,
                wheel_activity,
                leg_activity,
            ) = self._semantic_mode_targets(obs_history)
            per_sample_semantic_loss = -(
                semantic_targets * F.log_softmax(mode_logits, dim=-1)
            ).sum(dim=-1)
            confidence_sum = semantic_confidence.sum()
            if confidence_sum.item() > 0.0:
                semantic_loss = (
                    per_sample_semantic_loss * semantic_confidence
                ).sum() / confidence_sum
                semantic_mode_ratios = (
                    semantic_targets * semantic_confidence.unsqueeze(-1)
                ).sum(dim=0) / confidence_sum
            semantic_valid_ratio = valid_mask.float().mean()
            mean_wheel_activity = wheel_activity.mean()
            mean_leg_activity = leg_activity.mean()

        total_loss = (
            estimation_loss
            + swap_loss
            + self.mode_loss_coef * mode_loss
            + self.mode_semantic_loss_coef * semantic_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite MINE estimator loss")
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_mode_probs = mode_probs.detach().mean(dim=0)
        return (
            estimation_loss.item(),
            swap_loss.item(),
            mode_loss.item(),
            semantic_loss.item(),
            semantic_valid_ratio.item(),
            semantic_mode_ratios[self.WHEEL_MODE].item(),
            semantic_mode_ratios[self.LEG_MODE].item(),
            semantic_mode_ratios[self.HYBRID_MODE].item(),
            mean_wheel_activity.item(),
            mean_leg_activity.item(),
            mean_mode_probs[self.WHEEL_MODE].item(),
            mean_mode_probs[self.LEG_MODE].item(),
            mean_mode_probs[self.HYBRID_MODE].item(),
        )

    @torch.no_grad()
    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores - scores.max(dim=1, keepdim=True).values
        assignments = torch.exp(scores / self.sinkhorn_epsilon).T
        assignments /= assignments.sum().clamp_min(1e-12)
        num_prototypes, batch_size = assignments.shape
        for _ in range(self.sinkhorn_iterations):
            assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
            assignments /= float(num_prototypes)
            assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
            assignments /= float(batch_size)
        return (assignments * float(batch_size)).T
