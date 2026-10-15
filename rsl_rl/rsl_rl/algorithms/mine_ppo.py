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
        self.last_minibatch_diagnostics = []
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
        # Copy now: storage only receives the transition after env.step(), and
        # an environment may modify the buffers it returned (e.g. clearing the
        # histories of reset envs). The stored inputs must be the ones the
        # action was computed from, or PPO sees a spurious KL at every reset.
        self.transition.observations = obs.clone()
        self.transition.critic_observations = critic_obs.clone()
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

    @staticmethod
    def _kl_per_sample(mu, sigma, old_mu, old_sigma):
        # Preserve the scheduler's historical formula and numerical offset.
        return torch.sum(
            torch.log(sigma / old_sigma + 1e-5)
            + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square())
            - 0.5,
            dim=-1,
        )

    @staticmethod
    def _grouped_kl(kl, dones):
        reset = dones.flatten().bool()
        metrics = {"kl": kl.mean().item()}
        for name, mask in (("reset", reset), ("non_reset", ~reset)):
            count = int(mask.sum().item())
            metrics[name + "_count"] = count
            # Absent groups are omitted, never reported as a misleading zero.
            if count:
                metrics["kl_" + name] = kl[mask].mean().item()
        return metrics

    @torch.no_grad()
    def rollout_diagnostics(self, prefix, include_latent=False, chunk_size=4096):
        """Evaluate the complete rollout without sampling actions/changing RNG.

        Chunking bounds GPU memory. Only the small KL/latent tensors are kept
        for exact whole-rollout reductions, including the latent median.
        """
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("Diagnostics require a complete rollout")
        obs = self.storage.observations.flatten(0, 1)
        critic = self.storage.privileged_observations
        critic = obs if critic is None else critic.flatten(0, 1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)
        old_values = self.storage.values.flatten(0, 1)
        kls, latents = [], []
        value_error = obs.new_zeros(())
        for start in range(0, obs.shape[0], chunk_size):
            stop = start + chunk_size
            mean = self.actor_critic.act_inference(obs[start:stop])
            kls.append(self._kl_per_sample(
                mean, self.actor_critic.std, old_mu[start:stop], old_sigma[start:stop]
            ))
            if include_latent:
                values = self.actor_critic.evaluate(critic[start:stop])
                value_error = torch.maximum(
                    value_error, (values - old_values[start:stop]).abs().max()
                )
                _, latent = self.actor_critic.estimator.get_latent(obs[start:stop])
                latents.append(latent)
        kl = torch.cat(kls)
        metrics = {
            prefix + "_" + name: value
            for name, value in self._grouped_kl(kl, self.storage.dones).items()
        }
        metrics[prefix + "_kl_sample_max"] = kl.max().item()
        if include_latent:
            metrics[prefix + "_value_max_abs_error"] = value_error.item()
            metrics.update(self.actor_critic.estimator.latent_statistics(torch.cat(latents)))
        return metrics

    def update(self):
        # This audit happens before *any* estimator/PPO update, resets included.
        rollout_metrics = self.rollout_diagnostics("pre_update", include_latent=True)
        self.last_minibatch_diagnostics = []
        totals = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
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
            "kl_mean": 0.0,
            "kl_estimator_shift": 0.0,
        }
        # kl_first is measured before any parameter changes, so it must be
        # ~0 for a consistent rollout; kl_estimator_shift is the KL that one
        # estimator step alone induces in the actor's action mean.
        kl_first = None
        kl_max = 0.0
        estimator_diagnostics = {}
        updates = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs, include_dones=True
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
                dones_batch,
            ) = batch

            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            with torch.inference_mode():
                kl_samples = self._kl_per_sample(
                    mu_batch, sigma_batch, old_mu_batch, old_sigma_batch
                )
                batch_metrics = self._grouped_kl(kl_samples, dones_batch)
                kl = batch_metrics["kl"]
            if kl_first is None:
                kl_first = kl
            kl_max = max(kl_max, kl)
            if self.desired_kl is not None and self.schedule == "adaptive":
                if kl > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

            batch_metrics.update(
                update=updates + 1,
                epoch=updates // self.num_mini_batches + 1,
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            )
            batch_metrics["lr_at_floor"] = int(
                batch_metrics["learning_rate"] <= 1e-5 * (1.0 + 1e-6)
            )

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
            with torch.inference_mode():
                # The actor is unchanged here, so this difference comes only
                # from the estimator step just taken.
                shifted_mean = self.actor_critic.act_inference(obs_batch)
                kl_estimator_shift = torch.sum(
                    torch.square(shifted_mean - mu_batch)
                    / (2.0 * torch.square(sigma_batch)),
                    dim=-1,
                ).mean().item()
            batch_metrics["kl_estimator_shift"] = kl_estimator_shift
            self.last_minibatch_diagnostics.append(batch_metrics)
            for name, value in self.actor_critic.estimator.last_diagnostics.items():
                estimator_diagnostics[name] = (
                    estimator_diagnostics.get(name, 0.0) + value
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
            totals["entropy"] += entropy_batch.mean().item()
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
            totals["kl_mean"] += kl
            totals["kl_estimator_shift"] += kl_estimator_shift
            updates += 1

        if updates == 0:
            raise RuntimeError("MINEPPO received no mini-batches")
        rollout_metrics.update(self.rollout_diagnostics("post_update"))
        self.storage.clear()
        totals.update(estimator_diagnostics)
        means = {name: value / updates for name, value in totals.items()}
        means.update(kl_first=kl_first, kl_max=kl_max)
        for name in ("reset", "non_reset"):
            count = sum(row[name + "_count"] for row in self.last_minibatch_diagnostics)
            means["update_" + name + "_count"] = count
            if count:
                means["kl_" + name] = sum(
                    row.get("kl_" + name, 0.0) * row[name + "_count"]
                    for row in self.last_minibatch_diagnostics
                ) / count
        rates = [row["learning_rate"] for row in self.last_minibatch_diagnostics]
        means.update(
            learning_rate_min=min(rates),
            learning_rate_max=max(rates),
            learning_rate_mean=sum(rates) / updates,
            learning_rate_floor_fraction=sum(
                row["lr_at_floor"] for row in self.last_minibatch_diagnostics
            ) / updates,
        )
        # Latent health is measured on the whole rollout before learning, not
        # by averaging mini-batch medians or per-dimension standard deviations.
        means.update(rollout_metrics)
        return means
