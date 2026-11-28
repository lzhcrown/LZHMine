# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""WF concurrent teacher/student PPO and same-state latent distillation."""

import math

import torch
from torch import nn, optim

from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS


class MoECTS:
    def __init__(
        self, actor_critic, num_envs, num_learning_epochs=5, num_mini_batches=4,
        clip_param=0.2, gamma=0.99, lam=0.95, betas=(0.9, 0.999),
        weight_decay=0.0, value_loss_coef=1.0, entropy_coef=0.01,
        load_balance_coef=0.01, learning_rate=0.001,
        student_encoder_learning_rate=0.001, max_grad_norm=1.0,
        use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01,
        teacher_env_ratio=0.75, normalize_advantage_per_mini_batch=False,
        device="cpu", temporal_loss_coef=0.0,
    ):
        if num_envs < 2 or not 0 < teacher_env_ratio < 1:
            raise ValueError("CTS requires at least two environments and a teacher ratio in (0, 1)")
        if num_learning_epochs < 1 or num_mini_batches < 1:
            raise ValueError("Learning epochs and mini-batches must be positive")
        if actor_critic.is_recurrent:
            raise ValueError("CTS requires a feed-forward policy")
        if not math.isfinite(temporal_loss_coef) or temporal_loss_coef < 0:
            raise ValueError("temporal_loss_coef must be finite and nonnegative")
        self.actor_critic = actor_critic.to(device)
        self.policy = self.actor_critic
        self.device = device
        self.num_envs = num_envs
        self.teacher_num_envs = min(max(int(num_envs * teacher_env_ratio), 1), num_envs - 1)
        self.student_num_envs = num_envs - self.teacher_num_envs
        self.student_env_idxs = torch.arange(self.student_num_envs, device=device) * num_envs // self.student_num_envs
        teacher_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        teacher_mask[self.student_env_idxs] = False
        self.teacher_env_idxs = teacher_mask.nonzero(as_tuple=False).flatten()
        self.env_order = torch.cat((self.teacher_env_idxs, self.student_env_idxs))

        self.ppo_parameters = ([p for p in actor_critic.teacher_encoder.parameters() if p.requires_grad]
                               + list(actor_critic.actor.parameters())
                               + list(actor_critic.critic.parameters()) + [actor_critic.std])
        self.optimizer = optim.Adam(self.ppo_parameters, lr=learning_rate,
                                    betas=betas, weight_decay=weight_decay)
        self.student_parameters = [p for p in actor_critic.student_moe_encoder.parameters() if p.requires_grad]
        self.optimizer_stu_enc = optim.Adam(self.student_parameters,
                                            lr=student_encoder_learning_rate,
                                            betas=betas, weight_decay=weight_decay)
        self.storage = None
        self.transition = RolloutStorageCTS.Transition()
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.load_balance_coef = load_balance_coef
        self.temporal_loss_coef = float(temporal_loss_coef)
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        if num_envs != self.num_envs:
            raise ValueError("CTS environment count cannot change after grouping")
        if min(self.teacher_num_envs, self.student_num_envs) * num_transitions_per_env < self.num_mini_batches:
            raise ValueError("Every CTS mini-batch requires both teacher and student samples; reduce num_mini_batches")
        self.storage = RolloutStorageCTS(
            num_envs, self.teacher_num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, action_shape, self.device,
            collect_temporal_pairs=self.temporal_loss_coef > 0,
            store_platform_mask=self.actor_critic.critic_task_count == 2,
        )

    def train_mode(self):
        self.actor_critic.train()

    def test_mode(self):
        self.actor_critic.eval()

    @torch.no_grad()
    def act(self, obs, critic_obs, platform_mask=None):
        if self.transition.actions is not None:
            raise RuntimeError("Process the preceding environment step before requesting another action")
        results = []
        for indices, is_teacher in ((self.teacher_env_idxs, True), (self.student_env_idxs, False)):
            history, privileged = obs[indices], critic_obs[indices]
            actions = self.actor_critic.act(history, privileged, is_teacher=is_teacher)
            labels = platform_mask[indices] if platform_mask is not None else None
            results.append((
                actions, self.actor_critic.evaluate(history, privileged, is_teacher, labels),
                self.actor_critic.get_actions_log_prob(actions),
                self.actor_critic.action_mean, self.actor_critic.action_std,
            ))
        names = ("actions", "values", "actions_log_prob", "action_mean", "action_sigma")
        for name, parts in zip(names, zip(*results)):
            setattr(self.transition, name, torch.cat(parts, dim=0))
        # Indexing makes snapshots before an environment can mutate its buffers.
        self.transition.observations = obs[self.env_order]
        self.transition.critic_observations = critic_obs[self.env_order]
        if platform_mask is not None:
            self.transition.platform_mask = platform_mask[self.env_order]
        actions = torch.empty_like(self.transition.actions)
        actions[self.env_order] = self.transition.actions
        return actions

    def process_env_step(self, rewards, dones, infos):
        if self.storage is None or self.transition.actions is None:
            raise RuntimeError("Initialize storage and call act before process_env_step")
        self.transition.rewards = rewards.reshape(-1)[self.env_order].clone()
        self.transition.dones = dones.reshape(-1)[self.env_order]
        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device).reshape(-1)[self.env_order]
            # Keep WF/RSL-RL's timeout convention: bootstrap with stored V(s_t).
            # This intentionally does not evaluate the automatically reset
            # observation, which belongs to a different episode.
            self.transition.rewards += self.gamma * self.transition.values.squeeze(-1) * time_outs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    @torch.no_grad()
    def compute_returns(self, obs, critic_obs, platform_mask=None):
        last_values = torch.cat([
            self.actor_critic.evaluate(obs[indices], critic_obs[indices], is_teacher,
                                       platform_mask[indices] if platform_mask is not None else None)
            for indices, is_teacher in ((self.teacher_env_idxs, True), (self.student_env_idxs, False))
        ], dim=0)
        self.storage.compute_returns(last_values, self.gamma, self.lam,
                                     not self.normalize_advantage_per_mini_batch)

    def _batch_results(self, batch):
        split = batch["teacher_count"]
        results = []
        for selection, is_teacher in ((slice(None, split), True), (slice(split, None), False)):
            obs = batch["observations"][selection]
            privileged = batch["critic_observations"][selection]
            labels = batch["platform_mask"][selection] if "platform_mask" in batch else None
            self.actor_critic.act(obs, privileged, is_teacher=is_teacher)
            results.append((
                self.actor_critic.get_actions_log_prob(batch["actions"][selection]),
                self.actor_critic.evaluate(obs, privileged, is_teacher, labels),
                self.actor_critic.action_mean, self.actor_critic.action_std,
                self.actor_critic.entropy,
                self.actor_critic.actor_gate_weights,
                self.actor_critic.critic_gate_weights,
                self.actor_critic.action_residual,
            ))
        return [torch.cat(parts, dim=0) for parts in zip(*results)]

    def _adapt_learning_rate(self, means, stds, old_means, old_stds):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.no_grad():
            kl = (torch.log(stds / old_stds + 1.0e-5)
                  + (old_stds.square() + (old_means - means).square()) / (2 * stds.square())
                  - 0.5).sum(dim=-1).mean().item()
        if kl > self.desired_kl * 2:
            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
        elif 0 < kl < self.desired_kl / 2:
            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _temporal_loss(self, batch, current_means):
        """Current-policy final-mean smoothness, averaged over valid pairs/DOFs.

        Current means already belong to the PPO computation graph. Recompute
        predecessor means with the *same* current parameters and CTS branch;
        old stored means or sampled previous actions are not this objective.
        """
        valid = batch["temporal_mask"]
        valid_count = valid.sum()
        squared_error = current_means.sum() * 0.0
        split = batch["teacher_count"]
        for selection, is_teacher in ((slice(None, split), True), (slice(split, None), False)):
            selected_valid = valid[selection]
            if not bool(selected_valid.any()):
                continue
            previous_means, _, _, _ = self.actor_critic.action_mean_with_diagnostics(
                batch["previous_observations"][selection][selected_valid],
                batch["previous_critic_observations"][selection][selected_valid],
                is_teacher=is_teacher,
            )
            difference = current_means[selection][selected_valid] - previous_means
            squared_error = squared_error + difference.square().sum()
        denominator = current_means.shape[-1] * valid_count.clamp_min(1)
        return squared_error / denominator, valid_count

    def update(self):
        totals = {key: 0.0 for key in (
            "value", "surrogate", "teacher_surrogate", "student_surrogate",
            "entropy", "mean_latent_loss",
        )}
        student_experts = self.actor_critic.expert_num
        if student_experts:
            totals.update(mean_load_balance_loss=0.0, gate_entropy=0.0)
        temporal_error_sum = 0.0
        temporal_pair_count = 0
        if self.temporal_loss_coef > 0:
            totals.update(temporal=0.0, temporal_valid_pairs=0.0)
        ppo_updates = student_updates = 0
        task_value_errors = None
        if self.actor_critic.critic_type == "task_mlp":
            task_value_errors = torch.zeros(2, device=self.device)
            task_value_counts = torch.zeros(2, device=self.device, dtype=torch.long)
        residual_squared_sum = residual_saturated = residual_elements = 0
        residual_abs_max = 0.0
        routing_stats = {
            name: {
                "weights": torch.zeros(2, count, device=self.device),
                "entropy": torch.zeros(2, device=self.device),
                "samples": [0, 0],
            }
            for name, count in (("actor", self.actor_critic.actor_expert_num),
                                ("critic", self.actor_critic.critic_expert_num))
            if count
        }
        self.optimizer_stu_enc.zero_grad(set_to_none=True)
        # Finish all PPO updates before deriving targets from the new teacher.
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            log_probs, values, means, stds, entropy, actor_weights, critic_weights, residual = self._batch_results(batch)
            if self.actor_critic.residual_enabled:
                correction = residual.detach()
                residual_squared_sum += correction.square().sum().item()
                residual_elements += correction.numel()
                residual_abs_max = max(residual_abs_max, correction.abs().max().item())
                scales = self.actor_critic.actor.residual_action_scales
                residual_saturated += (correction.abs() >= .95 * scales).sum().item()
            self._adapt_learning_rate(means, stds, batch["mu"], batch["sigma"])
            advantages = batch["advantages"].squeeze(-1)
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
            ratio = torch.exp(log_probs - batch["actions_log_prob"].squeeze(-1))
            losses = torch.maximum(-advantages * ratio,
                                   -advantages * ratio.clamp(1 - self.clip_param, 1 + self.clip_param))
            split = batch["teacher_count"]
            teacher_surrogate = losses[:split].mean()
            student_surrogate = losses[split:].mean()
            surrogate = teacher_surrogate + student_surrogate
            value_errors = (values - batch["returns"]).square()
            if self.use_clipped_value_loss:
                clipped_values = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
                value_errors = torch.maximum(value_errors, (clipped_values - batch["returns"]).square())
            value_loss = value_errors.mean()
            if task_value_errors is not None:
                platform = batch["platform_mask"]
                task_masks = torch.stack((~platform, platform))
                task_value_errors += (task_masks * value_errors.detach().flatten()).sum(dim=1)
                task_value_counts += task_masks.sum(dim=1)
            entropy_mean = entropy.mean()
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean
            if self.temporal_loss_coef > 0:
                temporal_loss, valid_count = self._temporal_loss(batch, means)
                loss = loss + self.temporal_loss_coef * temporal_loss
                count = int(valid_count.item())
                temporal_error_sum += temporal_loss.detach().item() * count
                temporal_pair_count += count
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            for name, routing_weights in (("actor", actor_weights), ("critic", critic_weights)):
                if name not in routing_stats:
                    continue
                # Diagnostics use the routing already computed for PPO. Keep
                # the teacher and student distributions visible separately.
                stats = routing_stats[name]
                weights = routing_weights.detach()
                for group, selection in enumerate((slice(None, split), slice(split, None))):
                    group_weights = weights[selection]
                    stats["weights"][group] += group_weights.sum(dim=0)
                    stats["entropy"][group] -= (
                        group_weights * group_weights.clamp_min(1.0e-8).log()
                    ).sum()
                    stats["samples"][group] += group_weights.shape[0]
            for key, value in (("value", value_loss), ("surrogate", surrogate),
                               ("teacher_surrogate", teacher_surrogate), ("student_surrogate", student_surrogate),
                               ("entropy", entropy_mean)):
                totals[key] += value.item()
            ppo_updates += 1

        self.optimizer.zero_grad(set_to_none=True)
        expert_usage = torch.zeros(student_experts, device=self.device)
        student_samples = 0
        for observations, privileged in self.storage.student_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs):
            student_latent, weights = self.actor_critic.student_moe_encoder(
                observations, self.actor_critic.student_height_scan(privileged),
            )
            with torch.no_grad():
                teacher_latent = self.actor_critic.teacher_encoder(privileged)
            latent_loss = (student_latent - teacher_latent).square().mean()
            student_loss = latent_loss
            if student_experts:
                load_balance = (weights.mean(dim=0) - 1.0 / student_experts).square().mean()
                student_loss = student_loss + self.load_balance_coef * load_balance
            self.optimizer_stu_enc.zero_grad(set_to_none=True)
            student_loss.backward()
            nn.utils.clip_grad_norm_(self.student_parameters,
                                     self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer_stu_enc.step()
            totals["mean_latent_loss"] += latent_loss.item()
            if student_experts:
                detached_weights = weights.detach()
                totals["mean_load_balance_loss"] += load_balance.item()
                totals["gate_entropy"] += -(detached_weights * detached_weights.clamp_min(1.0e-8).log()).sum().item()
                expert_usage += detached_weights.sum(dim=0)
            student_samples += observations.shape[0]
            student_updates += 1

        for key in ("value", "surrogate", "teacher_surrogate", "student_surrogate", "entropy"):
            totals[key] /= ppo_updates
        if task_value_errors is not None:
            for name, error, count in zip(("value_ordinary", "value_high_platform"),
                                          task_value_errors.tolist(), task_value_counts.tolist()):
                if count:
                    totals[name] = error / count
        totals["mean_latent_loss"] /= student_updates
        if self.actor_critic.residual_enabled:
            totals["residual_rms"] = math.sqrt(residual_squared_sum / residual_elements)
            totals["residual_abs_max"] = residual_abs_max
            totals["residual_saturation_fraction"] = residual_saturated / residual_elements
        if self.temporal_loss_coef > 0:
            totals["temporal"] = temporal_error_sum / max(temporal_pair_count, 1)
            # Every valid rollout pair appears once in each PPO epoch.
            totals["temporal_valid_pairs"] = temporal_pair_count / self.num_learning_epochs
        if student_experts:
            totals["mean_load_balance_loss"] /= student_updates
            totals["gate_entropy"] /= student_samples
            for index, usage in enumerate((expert_usage / student_samples).tolist()):
                totals["expert_%d_usage" % index] = usage
        for network, stats in routing_stats.items():
            sample_count = sum(stats["samples"])
            totals["%s_gate_entropy" % network] = stats["entropy"].sum().item() / sample_count
            for index, usage in enumerate((stats["weights"].sum(dim=0) / sample_count).tolist()):
                totals["%s_expert_%d_usage" % (network, index)] = usage
            for group, name in enumerate(("teacher", "student")):
                totals["%s_%s_gate_entropy" % (network, name)] = stats["entropy"][group].item() / stats["samples"][group]
                for index, usage in enumerate((stats["weights"][group] / stats["samples"][group]).tolist()):
                    totals["%s_%s_expert_%d_usage" % (network, name, index)] = usage
        self.storage.clear()
        return totals
