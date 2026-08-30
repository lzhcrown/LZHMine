"""Actor-critic policy backed by the gated modal MINE estimator."""

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .actor_critic import get_activation
from .mine_estimator import MINEEstimator


def _policy_mlp(input_dim: int, hidden_dims: List[int], output_dim: int, activation: str) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), get_activation(activation)))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class MINEActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_one_step_privileged_obs: int,
        estimation_target_indices: List[int],
        num_actions: int,
        actor_hidden_dims: List[int] = [512, 256, 128],
        critic_hidden_dims: List[int] = [512, 256, 128],
        enc_hidden_dims: List[int] = [256, 128, 64],
        tar_hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 16,
        num_modes: int = 3,
        gate_hidden_dim: int = 64,
        num_prototypes: int = 32,
        temperature: float = 3.0,
        estimator_learning_rate: float = 1e-3,
        mode_loss_coef: float = 0.5,
        gate_balance_coef: float = 0.05,
        gate_entropy_coef: float = 0.01,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if num_actor_obs % num_one_step_obs:
            raise ValueError("Actor observation width must be a whole number of frames")

        self.history_size = num_actor_obs // num_one_step_obs
        self.num_actor_obs = num_actor_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_actions = num_actions
        self.num_est_prob = len(estimation_target_indices)
        self.latent_dim = latent_dim

        self.estimator = MINEEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=num_one_step_obs,
            num_one_step_privileged_obs=num_one_step_privileged_obs,
            estimation_target_indices=estimation_target_indices,
            enc_hidden_dims=enc_hidden_dims,
            tar_hidden_dims=tar_hidden_dims,
            latent_dim=latent_dim,
            num_modes=num_modes,
            gate_hidden_dim=gate_hidden_dim,
            num_prototypes=num_prototypes,
            temperature=temperature,
            learning_rate=estimator_learning_rate,
            mode_loss_coef=mode_loss_coef,
            gate_balance_coef=gate_balance_coef,
            gate_entropy_coef=gate_entropy_coef,
            activation=activation,
        )
        actor_input_dim = num_one_step_obs + self.num_est_prob + latent_dim
        self.actor = _policy_mlp(actor_input_dim, actor_hidden_dims, num_actions, activation)
        self.critic = _policy_mlp(num_critic_obs, critic_hidden_dims, 1, activation)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def policy_parameters(self):
        """Parameters optimized by PPO (the estimator has its own optimizer)."""
        return list(self.actor.parameters()) + list(self.critic.parameters()) + [self.std]

    def reset(self, dones=None):
        del dones

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_input(self, obs_history: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            estimate, latent = self.estimator(obs_history)
        current_obs = obs_history[..., : self.num_one_step_obs]
        return torch.cat((current_obs, estimate, latent), dim=-1)

    def update_distribution(self, obs_history: torch.Tensor):
        mean = self.actor(self._actor_input(obs_history))
        safe_std = self.std.clamp_min(1e-4)
        self.distribution = Normal(mean, mean * 0.0 + safe_std)

    def act(self, obs_history=None, **kwargs):
        del kwargs
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None, **kwargs):
        del observations, kwargs
        return self.actor(self._actor_input(obs_history))

    def evaluate(self, critic_observations, **kwargs):
        del kwargs
        return self.critic(critic_observations)


class MINEPolicyExporter(nn.Module):
    """Self-contained online policy used by play, sim-to-sim and deployment."""

    def __init__(self, actor_critic: MINEActorCritic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()
        estimator = actor_critic.estimator
        self.encoder = copy.deepcopy(estimator.encoder).cpu()
        self.mode_gate = copy.deepcopy(estimator.mode_gate).cpu()
        self.num_one_step_obs = estimator.num_one_step_obs
        self.num_est_prob = estimator.num_est_prob
        self.register_buffer(
            "mode_prototypes",
            estimator.mode_prototypes.detach().clone().cpu(),
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.encoder(obs_history)
        estimate = parts[..., : self.num_est_prob]
        latent = parts[..., self.num_est_prob :]
        current_obs = obs_history[..., : self.num_one_step_obs]
        mode_probs = F.softmax(self.mode_gate(current_obs), dim=-1)
        normalized_modes = F.normalize(self.mode_prototypes, dim=-1, p=2.0)
        modal_scale = torch.matmul(mode_probs, normalized_modes)
        latent = F.normalize(latent * modal_scale, dim=-1, p=2.0)
        return self.actor(torch.cat((current_obs, estimate, latent), dim=-1))

    @torch.jit.export
    def get_mode_probabilities(self, obs_history: torch.Tensor) -> torch.Tensor:
        current_obs = obs_history[..., : self.num_one_step_obs]
        return F.softmax(self.mode_gate(current_obs), dim=-1)

    def export(self, path: str):
        self.eval()
        scripted = torch.jit.script(self)
        scripted.save(path)
