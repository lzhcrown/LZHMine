"""Rollout storage matching wheel_gym_CQ's MINE experiment."""

from .him_rollout_storage import HIMRolloutStorage


class MINERolloutStorage(HIMRolloutStorage):
    """MINE storage without LZHMine's later reward/advantage clamps."""

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
