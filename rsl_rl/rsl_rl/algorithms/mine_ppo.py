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
        estimator_update_schedule="interleaved",
        **kwargs,
    ):
        del kwargs
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        if estimator_update_schedule not in ("interleaved", "after_ppo"):
            raise ValueError("estimator_update_schedule must be 'interleaved' or 'after_ppo'")
        self.estimator_update_schedule = estimator_update_schedule
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
    def rollout_diagnostics(self, prefix, include_latent=False, chunk_size=4096,
                            capture_means=False, estimator_reference_means=None):
        """Evaluate the complete rollout without sampling actions/changing RNG.

        Chunking bounds GPU memory. Only the small KL/latent tensors are kept
        for exact whole-rollout reductions, including the latent median.
        B can capture means after PPO and compare them after the estimator
        block. That comparison holds the actor and sigma fixed, and has no
        numerical KL offset. It is NOT post_update KL minus post_ppo KL.
        """
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("Diagnostics require a complete rollout")
        obs = self.storage.observations.flatten(0, 1)
        critic = self.storage.privileged_observations
        critic = obs if critic is None else critic.flatten(0, 1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)
        old_values = self.storage.values.flatten(0, 1)
        kls, latents, means, estimator_kls = [], [], [], []
        value_error = obs.new_zeros(())
        for start in range(0, obs.shape[0], chunk_size):
            stop = start + chunk_size
            mean = self.actor_critic.act_inference(obs[start:stop])
            if capture_means:
                means.append(mean)
            if estimator_reference_means is not None:
                estimator_kls.append(torch.sum(
                    (mean - estimator_reference_means[start:stop]).square()
                    / (2.0 * self.actor_critic.std.square()), dim=-1,
                ))
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
        if estimator_kls:
            estimator_kl = torch.cat(estimator_kls)
            metrics.update({
                "estimator_phase_" + name: value
                for name, value in self._grouped_kl(estimator_kl, self.storage.dones).items()
            })
            metrics["estimator_phase_kl_sample_max"] = estimator_kl.max().item()
        if include_latent:
            metrics[prefix + "_value_max_abs_error"] = value_error.item()
            metrics.update(self.actor_critic.estimator.latent_statistics(torch.cat(latents)))
        return (metrics, torch.cat(means)) if capture_means else metrics

    def _update_estimator(self, obs, next_critic_obs, learning_rate,
                          reference_mean=None, reference_sigma=None):
        # In B the actor has already completed all PPO steps. Re-evaluate the
        # current policy before EACH estimator step, never reuse old PPO means.
        if reference_mean is None:
            with torch.inference_mode():
                reference_mean = self.actor_critic.act_inference(obs)
                reference_sigma = self.actor_critic.std.detach()
        losses = self.actor_critic.estimator.update(
            obs, next_critic_obs, learning_rate=learning_rate,
        )
        names = (
            "estimation", "swap", "mode", "semantic_mode", "semantic_valid_ratio",
            "wheel_target_ratio", "leg_target_ratio", "hybrid_target_ratio",
            "wheel_activity", "leg_activity", "wheel_gate_probability",
            "leg_gate_probability", "hybrid_gate_probability",
        )
        metrics = dict(zip(names, losses))
        with torch.inference_mode():
            shifted_mean = self.actor_critic.act_inference(obs)
            metrics["kl_estimator_shift"] = torch.sum(
                torch.square(shifted_mean - reference_mean)
                / (2.0 * torch.square(reference_sigma)), dim=-1,
            ).mean().item()
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
        deferred = self.estimator_update_schedule == "after_ppo"
        # Store only index views and scalar rates, not multi-GB batch copies.
        estimator_batches = []

        def record_estimator(metrics, batch_metrics, rate):
            batch_metrics["kl_estimator_shift"] = metrics["kl_estimator_shift"]
            batch_metrics["estimator_learning_rate"] = rate
            for name, value in metrics.items():
                totals[name] += value
            for name, value in self.actor_critic.estimator.last_diagnostics.items():
                estimator_diagnostics[name] = estimator_diagnostics.get(name, 0.0) + value

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs, include_dones=True,
            include_indices=deferred,
        )

        for batch in generator:
            if deferred:
                batch, indices = batch[:-1], batch[-1]
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

            if deferred:
                estimator_batches.append((indices, self.learning_rate, batch_metrics))
            else:
                estimator_metrics = self._update_estimator(
                    obs_batch, next_critic_obs_batch, self.learning_rate,
                    reference_mean=mu_batch, reference_sigma=sigma_batch,
                )
                record_estimator(estimator_metrics, batch_metrics, self.learning_rate)
            self.last_minibatch_diagnostics.append(batch_metrics)

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
            totals["kl_mean"] += kl
            updates += 1

        if updates == 0:
            raise RuntimeError("MINEPPO received no mini-batches")
        estimator_reference = None
        if deferred:
            post_ppo, estimator_reference = self.rollout_diagnostics(
                "post_ppo", capture_means=True,
            )
            rollout_metrics.update(post_ppo)
            for indices, rate, batch_metrics in estimator_batches:
                estimator_obs, estimator_next_critic = self.storage.estimator_batch(indices)
                metrics = self._update_estimator(estimator_obs, estimator_next_critic, rate)
                record_estimator(metrics, batch_metrics, rate)
            # The main optimizer also owns these parameters; leave no stale
            # gradients from the deferred block for the next rollout/update.
            self.actor_critic.estimator.optimizer.zero_grad(set_to_none=True)
        rollout_metrics.update(self.rollout_diagnostics(
            "post_update", estimator_reference_means=estimator_reference,
        ))
        self.storage.clear()
        totals.update(estimator_diagnostics)
        means = {name: value / updates for name, value in totals.items()}
        means.update(kl_first=kl_first, kl_max=kl_max)
        means.update(update_ppo_count=updates, update_estimator_count=updates)
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
