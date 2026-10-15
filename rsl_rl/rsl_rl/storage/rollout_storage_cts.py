# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Feed-forward CTS storage with teacher rows followed by student rows.

Mini-batches include both groups, keep every sample including remainder rows,
and gather only the current batch rather than duplicating the full rollout.
"""

import torch


class RolloutStorageCTS:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, teacher_num_envs, num_transitions_per_env,
                 obs_shape, critic_obs_shape, actions_shape, device="cpu"):
        if not 0 < teacher_num_envs < num_envs:
            raise ValueError("CTS storage requires both teacher and student environments")
        if num_transitions_per_env < 1:
            raise ValueError("Rollout length must be positive")
        self.num_envs = num_envs
        self.teacher_num_envs = teacher_num_envs
        self.student_num_envs = num_envs - teacher_num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        shape = (num_transitions_per_env, num_envs)
        self.observations = torch.zeros(*shape, *obs_shape, device=device)
        self.privileged_observations = torch.zeros(*shape, *critic_obs_shape, device=device)
        self.actions = torch.zeros(*shape, *actions_shape, device=device)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.dones = torch.zeros(*shape, 1, device=device, dtype=torch.bool)
        self.values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.actions_log_prob = torch.zeros_like(self.rewards)
        self.step = 0

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("CTS rollout buffer is full")
        pairs = (
            (self.observations, transition.observations),
            (self.privileged_observations, transition.critic_observations),
            (self.actions, transition.actions), (self.values, transition.values),
            (self.mu, transition.action_mean), (self.sigma, transition.action_sigma),
            (self.rewards, transition.rewards.reshape(-1, 1)),
            (self.dones, transition.dones.reshape(-1, 1)),
            (self.actions_log_prob, transition.actions_log_prob.reshape(-1, 1)),
        )
        for target, source in pairs:
            target[self.step].copy_(source.detach())
        self.step += 1

    add_transition = add_transitions

    def clear(self):
        self.step = 0

    def _require_full(self):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("A complete CTS rollout is required before learning")

    def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
        self._require_full()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            not_terminal = (~self.dones[step]).float()
            delta = self.rewards[step] + gamma * next_values * not_terminal - self.values[step]
            advantage = delta + gamma * lam * not_terminal * advantage
            self.returns[step].copy_(advantage + self.values[step])
        self.advantages.copy_(self.returns - self.values)
        if normalize_advantage:
            self.advantages.sub_(self.advantages.mean()).div_(self.advantages.std() + 1.0e-8)

    def _permuted_slices(self, sample_count, num_mini_batches):
        if not 1 <= num_mini_batches <= sample_count:
            raise ValueError("Every CTS mini-batch must include samples from both groups")
        indices = torch.randperm(sample_count, device=self.device)
        # Integer boundaries cover the entire permutation, even when the
        # number of samples is not divisible by the number of mini-batches.
        return [indices[i * sample_count // num_mini_batches:
                        (i + 1) * sample_count // num_mini_batches]
                for i in range(num_mini_batches)]

    def _gather(self, tensor, indices):
        # Indices are logically env-major; storage remains time-major.
        env_ids = indices // self.num_transitions_per_env
        time_ids = indices % self.num_transitions_per_env
        return tensor[time_ids, env_ids]

    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        self._require_full()
        teacher_count = self.teacher_num_envs * self.num_transitions_per_env
        student_count = self.student_num_envs * self.num_transitions_per_env
        fields = {
            "observations": self.observations,
            "critic_observations": self.privileged_observations,
            "actions": self.actions, "values": self.values,
            "returns": self.returns, "advantages": self.advantages,
            "actions_log_prob": self.actions_log_prob,
            "mu": self.mu, "sigma": self.sigma,
        }
        for _ in range(num_epochs):
            teachers = self._permuted_slices(teacher_count, num_mini_batches)
            students = self._permuted_slices(student_count, num_mini_batches)
            for teacher_ids, student_ids in zip(teachers, students):
                ids = torch.cat((teacher_ids, teacher_count + student_ids))
                batch = {name: self._gather(data, ids) for name, data in fields.items()}
                batch["teacher_count"] = teacher_ids.numel()
                yield batch

    def student_mini_batch_generator(self, num_mini_batches, num_epochs=1):
        self._require_full()
        teacher_count = self.teacher_num_envs * self.num_transitions_per_env
        student_count = self.student_num_envs * self.num_transitions_per_env
        for _ in range(num_epochs):
            for indices in self._permuted_slices(student_count, num_mini_batches):
                indices = teacher_count + indices
                yield (self._gather(self.observations, indices),
                       self._gather(self.privileged_observations, indices))
