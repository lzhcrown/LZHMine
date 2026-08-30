"""Gated modal dual-encoder used by the Nezha MINE policy.

The source encoder consumes proprioceptive history available on the robot.  The
target encoder is training-only and consumes privileged simulator state.  A
soft modality gate blends learnable modal prototypes before the source latent
is passed to the policy.
"""

from typing import List, Tuple

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
    training, play, export, MuJoCo and deployment.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_one_step_privileged_obs: int,
        estimation_target_indices: List[int],
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
        gate_balance_coef: float = 0.05,
        gate_entropy_coef: float = 0.01,
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
        self.num_est_prob = len(estimation_target_indices)
        self.num_latent = latent_dim
        self.num_modes = num_modes
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.mode_loss_coef = mode_loss_coef
        self.gate_balance_coef = gate_balance_coef
        self.gate_entropy_coef = gate_entropy_coef
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
        self.target = _mlp(
            num_one_step_privileged_obs,
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

    def _source(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        mode_probs = F.softmax(self.mode_gate(current_obs), dim=-1)
        normalized_modes = F.normalize(self.mode_prototypes, dim=-1, p=2.0)
        modal_scale = torch.matmul(mode_probs, normalized_modes)
        latent = F.normalize(source_latent * modal_scale, dim=-1, p=2.0, eps=1e-8)
        return estimate, latent, mode_probs

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        estimate, latent, _ = self._source(obs_history)
        return estimate.detach(), latent.detach()

    def encode(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._source(obs_history)

    @torch.no_grad()
    def get_latent(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        estimate, latent, _ = self._source(obs_history)
        return estimate, latent

    @torch.no_grad()
    def mode_probabilities(self, obs_history: torch.Tensor) -> torch.Tensor:
        _, _, probabilities = self._source(obs_history)
        return probabilities

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        learning_rate: float = None,
    ) -> Tuple[float, float, float, float]:
        if learning_rate is not None:
            self.learning_rate = learning_rate
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

        privileged = next_critic_obs[..., : self.num_one_step_privileged_obs].detach()
        target_values = privileged.index_select(-1, self.estimation_target_indices)
        predicted_values, source_latent, mode_probs = self._source(obs_history)
        target_latent = F.normalize(self.target(privileged), dim=-1, p=2.0, eps=1e-8)

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
            (source_assignments * target_log_probs).sum(dim=-1).mean()
            + (target_assignments * source_log_probs).sum(dim=-1).mean()
        )
        estimation_loss = F.mse_loss(predicted_values, target_values)

        normalized_modes = F.normalize(self.mode_prototypes, dim=-1, p=2.0, eps=1e-8)
        modal_target = F.normalize(mode_probs @ normalized_modes, dim=-1, p=2.0, eps=1e-8)
        mode_loss = (1.0 - F.cosine_similarity(source_latent, modal_target, dim=-1)).mean()

        mean_mode_probs = mode_probs.mean(dim=0)
        balance_loss = torch.sum(
            mean_mode_probs * torch.log(mean_mode_probs * float(self.num_modes) + 1e-8)
        )
        entropy_loss = -torch.sum(mode_probs * torch.log(mode_probs + 1e-8), dim=-1).mean()
        total_loss = (
            estimation_loss
            + swap_loss
            + self.mode_loss_coef * mode_loss
            + self.gate_balance_coef * balance_loss
            + self.gate_entropy_coef * entropy_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite MINE estimator loss")
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            estimation_loss.item(),
            swap_loss.item(),
            mode_loss.item(),
            entropy_loss.item(),
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
