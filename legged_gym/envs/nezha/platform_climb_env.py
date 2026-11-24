"""G13 uses the assigned platform face and wheel support to score climbing."""

import math

import torch

from legged_gym.envs.nezha.nezha_moe_cts_env import NezhaMoECTSEnv
from legged_gym.terrains.platform_climb import platform_climb_state, route_coordinates


class NezhaPlatformClimbEnv(NezhaMoECTSEnv):
    def _init_buffers(self):
        super()._init_buffers()
        # Plane playback has no assigned ledge. The shared ordinary policy
        # and reward still work with the same observations on that surface.
        self.climb_enabled = (getattr(self.cfg.rewards, "platform_climb", False)
                              and getattr(self, "platform_tiles", None) is not None)
        if not self.climb_enabled:
            return
        terrain = self.cfg.terrain
        settings = self.cfg.rewards.platform_climb_settings
        self.settle_steps = max(1, math.ceil(settings["settle_time_s"] / self.dt))
        self.upper_support_reward = settings["upper_support_reward"]
        self.completion_reward = settings["completion_reward"]
        specs = terrain.wf_sub_terrains
        self.block_half_width = specs.get("platform_block", {}).get("platform_width", 0.) / 2
        self.pit_half_width = specs.get("platform_pit", {}).get("platform_width", 0.) / 2
        self.climb_center = self.root_states.new_zeros((self.num_envs, 2))
        self.climb_axis = torch.zeros_like(self.climb_center)
        self.cardinal_axes = self.climb_axis.new_tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        self.climb_is_block = torch.zeros_like(self.high_platform_mask)
        self.climb_half_width = torch.zeros_like(self.rew_buf)
        self.best_forward = torch.zeros_like(self.rew_buf)
        self.best_support = torch.zeros_like(self.rew_buf)
        self.forward_tracking = torch.zeros_like(self.rew_buf)
        self.new_support = torch.zeros_like(self.rew_buf)
        self.settled_steps = torch.zeros_like(self.episode_length_buf)
        self.climb_success = torch.zeros_like(self.high_platform_mask)
        self.climb_bypass = torch.zeros_like(self.high_platform_mask)
        self.climb_finished = torch.zeros_like(self.high_platform_mask)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        if self.climb_enabled:
            for name in ("platform_upper_support", "platform_completion"):
                self.episode_sums[name] = torch.zeros_like(self.rew_buf)

    def _sample_spawn_yaws(self, env_ids):
        yaws = super()._sample_spawn_yaws(env_ids)
        if not self.climb_enabled:
            return yaws
        # A pit can be climbed on any face. Assign the face once at reset
        # and initially point toward it, so the forward command is meaningful
        # to a student that sees only proprioception.
        cardinal = (yaws[:, 0] / (math.pi / 2)).round().long()
        self.climb_axis[env_ids] = self.cardinal_axes[cardinal.remainder(4)]
        return torch.where(self.high_platform_mask[env_ids, None],
                           cardinal[:, None] * (math.pi / 2), yaws)

    def check_termination(self):
        super().check_termination()
        if not self.climb_enabled:
            return
        wheels = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wheel_body_indices, :2]
        support_z = self.contact_forces[:, self.wheel_body_indices, 2]
        state = platform_climb_state(
            self.root_states[:, :2], wheels, support_z, self.projected_gravity[:, 2],
            self.climb_center, self.climb_axis, self.climb_is_block, self.climb_half_width,
            tile_half_length=self.cfg.terrain.terrain_length / 2,
        )
        failed = self.reset_buf.bool() & ~self.time_out_buf
        valid = self.high_platform_mask & state.in_lane & ~failed
        # Pay tracking only for new ground along the assigned approach.
        # Turning in front of the wall or retreating and returning cannot
        # collect the same distance a second time.
        speed = self.commands[:, 0]
        delta = (state.forward - self.best_forward).clamp_min(0)
        progress = (delta / (speed * self.dt).clamp_min(1e-6)).clamp_max(1)
        target_xy = speed[:, None] * self.climb_axis
        velocity_error = (self.root_states[:, 7:9] - target_xy).square().sum(dim=1)
        tracking = torch.exp(-velocity_error / self.cfg.rewards.tracking_sigma)
        self.forward_tracking.copy_(torch.where(valid & (speed > 0), tracking * progress, 0.0))
        self.best_forward.add_(torch.where(valid, delta, 0.0))
        self.new_support.copy_(torch.where(valid, (state.wheel_fraction - self.best_support).clamp_min(0), 0.0))
        self.best_support.add_(self.new_support)
        self.settled_steps.copy_(torch.where(valid & state.settled, self.settled_steps + 1, 0))
        self.climb_success.copy_(self.settled_steps >= self.settle_steps)
        self.climb_bypass.copy_(self.high_platform_mask & state.bypass & ~failed)
        task_terminal = self.climb_success | self.climb_bypass
        self.time_out_buf &= ~task_terminal
        self.reset_buf |= task_terminal
        self.climb_finished.copy_(self.high_platform_mask & self.reset_buf.bool())

    def _reward_tracking_lin_vel(self):
        ordinary = super()._reward_tracking_lin_vel()
        if self.climb_enabled:
            return torch.where(self.high_platform_mask, self.forward_tracking, ordinary)
        return ordinary

    def _reward_termination(self):
        termination = super()._reward_termination()
        return termination * ~self.climb_success if self.climb_enabled else termination

    def compute_reward(self):
        super().compute_reward()
        if self.climb_enabled:
            # These discrete events are in return units, after the continuous
            # Dreamwaq reward. A wheel-load handoff or a contact bounce cannot
            # pay repeatedly, and successful episodes end on this same step.
            support = self.new_support * self.upper_support_reward
            completion = self.climb_success * self.completion_reward
            self.rew_buf += support + completion
            self.episode_sums["platform_upper_support"] += support
            self.episode_sums["platform_completion"] += completion

    def _terrain_move_masks(self, env_ids):
        up, down = super()._terrain_move_masks(env_ids)
        if not self.climb_enabled:
            return up, down
        platform = self.high_platform_mask[env_ids]
        success, finished = self.climb_success[env_ids], self.climb_finished[env_ids]
        return torch.where(platform, success, up), torch.where(platform, finished & ~success, down)

    def reset_idx(self, env_ids):
        if not self.climb_enabled:
            return super().reset_idx(env_ids)
        self.extras.pop("platform_outcomes", None)
        if not len(env_ids):
            return
        # Snapshot completed episodes before terrain reassignment or clearing.
        ids = env_ids[self.climb_finished[env_ids]]
        if len(ids):
            self.extras["platform_outcomes"] = {
                "env_ids": ids, "is_block": self.climb_is_block[ids],
                "success": self.climb_success[ids], "bypass": self.climb_bypass[ids],
                "peak_wheel_fraction": self.best_support[ids],
            }
        super().reset_idx(env_ids)
        row, col = self.terrain_levels[env_ids], self.terrain_types[env_ids]
        terrain = self.cfg.terrain
        self.climb_center[env_ids, 0] = (row + .5) * terrain.terrain_length
        self.climb_center[env_ids, 1] = (col + .5) * terrain.terrain_width
        if self.block_tiles is not None:
            self.climb_is_block[env_ids] = self.block_tiles[row, col]
        self.climb_half_width[env_ids] = torch.where(self.climb_is_block[env_ids],
                                                      self.block_half_width, self.pit_half_width)
        self.best_forward[env_ids], _ = route_coordinates(
            self.root_states[env_ids, :2], self.climb_center[env_ids], self.climb_axis[env_ids],
        )
        for buffer in (self.best_support, self.new_support, self.forward_tracking, self.settled_steps,
                       self.climb_success, self.climb_bypass, self.climb_finished):
            buffer[env_ids] = 0
