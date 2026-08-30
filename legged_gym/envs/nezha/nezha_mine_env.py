"""Nezha wheel-legged environment for gated-modal training."""

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import get_scale_shift


class NezhaMINEEnv(LeggedRobot):
    """Nezha-specific observation and hybrid joint/wheel control interface.

    Both actor and critic histories are newest-first.  The four estimator
    targets (body-frame linear velocity and base height) occupy indices 65:69
    in every privileged frame.
    """

    LEG_JOINT_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

    def _get_noise_scale_vec(self, cfg):
        noise = torch.zeros(cfg.env.num_one_step_observations, device=self.device)
        scales = cfg.noise.noise_scales
        level = cfg.noise.noise_level
        noise[0:3] = scales.ang_vel * level * self.obs_scales.ang_vel
        noise[3:6] = scales.gravity * level
        noise[9:21] = scales.dof_pos * level * self.obs_scales.dof_pos
        noise[21:37] = scales.dof_vel * level * self.obs_scales.dof_vel
        noise[37:49] = scales.dof_pos * level * self.obs_scales.dof_pos
        self.add_noise = cfg.noise.add_noise
        return noise

    def _build_one_step_observations(self, add_noise):
        leg_ids = self.LEG_JOINT_IDS
        joint_error = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        actor_clean = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                joint_error[:, leg_ids],
                self.dof_vel * self.obs_scales.dof_vel,
                self.dof_pos[:, leg_ids] * self.obs_scales.dof_pos,
                self.actions,
            ),
            dim=-1,
        )
        actor_obs = actor_clean
        if add_noise and self.add_noise:
            actor_obs = actor_clean + (2.0 * torch.rand_like(actor_clean) - 1.0) * self.noise_scale_vec

        if self.cfg.terrain.measure_heights:
            terrain_heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1.0,
                1.0,
            ) * self.obs_scales.height_measurements
        else:
            terrain_heights = torch.zeros(
                self.num_envs,
                self.cfg.env.num_height_observations,
                device=self.device,
            )
        base_height = self._get_base_heights().unsqueeze(1)
        force_scale, force_shift = get_scale_shift(
            self.cfg.normalization.contact_force_range
        )
        foot_forces = (
            self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
            - force_shift
        ) * force_scale
        privileged = torch.cat(
            (
                actor_clean,
                self.base_lin_vel * self.obs_scales.lin_vel,
                base_height * self.obs_scales.height_measurements,
                terrain_heights,
                foot_forces,
                self.payload,
                self.com_displacement,
            ),
            dim=-1,
        )
        if actor_obs.shape[-1] != self.num_one_step_obs:
            raise RuntimeError(
                f"Nezha actor observation mismatch: {actor_obs.shape[-1]} != {self.num_one_step_obs}"
            )
        if privileged.shape[-1] != self.num_one_step_privileged_obs:
            raise RuntimeError(
                "Nezha privileged observation mismatch: "
                f"{privileged.shape[-1]} != {self.num_one_step_privileged_obs}"
            )
        return actor_obs, privileged

    def compute_observations(self):
        actor_obs, privileged_obs = self._build_one_step_observations(add_noise=True)
        self.obs_buf = torch.cat(
            (actor_obs, self.obs_buf[:, : -self.num_one_step_obs]), dim=-1
        )
        self.privileged_obs_buf = torch.cat(
            (
                privileged_obs,
                self.privileged_obs_buf[:, : -self.num_one_step_privileged_obs],
            ),
            dim=-1,
        )

    def compute_termination_observations(self, env_ids):
        if env_ids.numel() == 0:
            return self.privileged_obs_buf.new_zeros((0, self.num_privileged_obs))
        _, privileged_obs = self._build_one_step_observations(add_noise=False)
        termination_history = torch.cat(
            (
                privileged_obs,
                self.privileged_obs_buf[:, : -self.num_one_step_privileged_obs],
            ),
            dim=-1,
        )
        return termination_history[env_ids]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) and hasattr(self, "obs_buf"):
            self.obs_buf[env_ids] = 0.0
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            low, high = self.cfg.domain_rand.initial_joint_pos_range
            scale = low + (high - low) * torch.rand(
                len(env_ids), self.num_dof, device=self.device
            )
            self.dof_pos[env_ids] = self.default_dof_pos * scale
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        from isaacgym import gymtorch

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reward_dof_vel(self):
        leg_velocity = self.dof_vel.clone()
        leg_velocity[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(leg_velocity), dim=1)
