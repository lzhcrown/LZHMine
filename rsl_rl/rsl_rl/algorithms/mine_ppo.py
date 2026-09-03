"""PPO integration for the gated modal dual encoder."""

import torch
import torch.nn as nn

from rsl_rl.modules.mine_actor_critic import MINEActorCritic
from rsl_rl.storage.mine_rollout_storage import MINERolloutStorage


class MINEPPO:
    actor_critic: MINEActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        **kwargs,
    ):
        del kwargs
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.actor_critic = actor_critic.to(device)
        self.storage = None
        # Match wheel_gym_CQ's checkpoint and optimizer parameter group.  The
        # estimator also retains its dedicated optimizer; policy inference is
        # detached, so PPO normally contributes no estimator gradients.
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate
        )
        self.transition = MINERolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = MINERolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        totals = {
            "value": 0.0,
            "surrogate": 0.0,
            "estimation": 0.0,
            "swap": 0.0,
            "mode": 0.0,
            "semantic_mode": 0.0,
            "semantic_valid_ratio": 0.0,
            "wheel_target_ratio": 0.0,
            "leg_target_ratio": 0.0,
            "hybrid_target_ratio": 0.0,
            "wheel_activity": 0.0,
            "leg_activity": 0.0,
            "wheel_gate_probability": 0.0,
            "leg_gate_probability": 0.0,
            "hybrid_gate_probability": 0.0,
        }
        updates = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for batch in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                next_critic_obs_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = batch

            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            (
                estimation_loss,
                swap_loss,
                mode_loss,
                semantic_loss,
                semantic_valid_ratio,
                wheel_target_ratio,
                leg_target_ratio,
                hybrid_target_ratio,
                wheel_activity,
                leg_activity,
                wheel_gate_probability,
                leg_gate_probability,
                hybrid_gate_probability,
            ) = self.actor_critic.estimator.update(
                obs_batch,
                next_critic_obs_batch,
                learning_rate=self.learning_rate,
            )

            ratio = torch.exp(
                actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1)
            )
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            policy_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            self.optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["estimation"] += estimation_loss
            totals["swap"] += swap_loss
            totals["mode"] += mode_loss
            totals["semantic_mode"] += semantic_loss
            totals["semantic_valid_ratio"] += semantic_valid_ratio
            totals["wheel_target_ratio"] += wheel_target_ratio
            totals["leg_target_ratio"] += leg_target_ratio
            totals["hybrid_target_ratio"] += hybrid_target_ratio
            totals["wheel_activity"] += wheel_activity
            totals["leg_activity"] += leg_activity
            totals["wheel_gate_probability"] += wheel_gate_probability
            totals["leg_gate_probability"] += leg_gate_probability
            totals["hybrid_gate_probability"] += hybrid_gate_probability
            updates += 1

        self.storage.clear()
        if updates == 0:
            raise RuntimeError("MINEPPO received no mini-batches")
        return {name: value / updates for name, value in totals.items()}
