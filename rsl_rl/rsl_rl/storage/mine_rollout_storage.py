"""Rollout storage matching wheel_gym_CQ's MINE experiment."""

import torch

from .him_rollout_storage import HIMRolloutStorage


class MINERolloutStorage(HIMRolloutStorage):
    """MINE storage without LZHMine's later reward/advantage clamps."""

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, include_dones=False,
                             include_indices=False):
        """Keep reset labels aligned with the same shuffled transition indices.

        A reset sample means done[t] is true: step(action[t]) reset this env.
        The default ten-field interface remains compatible with other callers.
        Preserve the original shuffle/update order (one permutation per rollout).
        Optional indices let a deferred estimator replay exactly these samples
        without another shuffle or retaining copies of all privileged batches.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError("More mini-batches than rollout samples")
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
        critic = self.privileged_observations
        next_critic = self.next_privileged_observations
        fields = tuple(value.flatten(0, 1) for value in (
            self.observations,
            critic if critic is not None else self.observations,
            self.actions,
            next_critic if next_critic is not None else self.observations,
            self.values, self.advantages, self.returns, self.actions_log_prob,
            self.mu, self.sigma,
        ))
        dones = self.dones.flatten(0, 1)
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                batch_ids = indices[index * mini_batch_size:(index + 1) * mini_batch_size]
                batch = tuple(value[batch_ids] for value in fields)
                if include_dones:
                    batch += (dones[batch_ids],)
                if include_indices:
                    batch += (batch_ids,)
                yield batch

    def estimator_batch(self, indices):
        """Read a previously selected batch while the rollout is still live."""
        next_critic = self.next_privileged_observations
        if next_critic is None:
            next_critic = self.observations
        return (self.observations.flatten(0, 1)[indices],
                next_critic.flatten(0, 1)[indices])

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = (
                delta + next_is_not_terminal * gamma * lam * advantage
            )
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (
            self.advantages - self.advantages.mean()
        ) / (self.advantages.std() + 1e-8)
