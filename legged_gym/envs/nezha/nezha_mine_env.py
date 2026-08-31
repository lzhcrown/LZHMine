"""Nezha wheel-legged environment for gated-modal training."""

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import get_scale_shift


class NezhaMINEEnv(LeggedRobot):
    """Nezha-specific observation and hybrid joint/wheel control interface.

    Both actor and critic histories are newest-first.  The four estimator
    targets (body-frame linear velocity and base height) occupy indices 371:375
    in every privileged frame.
    """

    LEG_JOINT_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

    def _process_rigid_body_props(self, props, env_id):
        """Keep the randomized base mass/COM for the full privileged frame."""
        props = super()._process_rigid_body_props(props, env_id)
        if not hasattr(self, "mine_base_mass"):
            self.mine_base_mass = torch.zeros(
                self.num_envs, 1, dtype=torch.float32, device=self.device
            )
            self.mine_base_com = torch.zeros(
                self.num_envs, 3, dtype=torch.float32, device=self.device
            )
        self.mine_base_mass[env_id, 0] = props[0].mass
        self.mine_base_com[env_id] = torch.tensor(
            [props[0].com.x, props[0].com.y, props[0].com.z],
            dtype=torch.float32,
            device=self.device,
        )
        return props

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
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        base_height = self._get_base_heights().unsqueeze(1)
        force_scale, force_shift = get_scale_shift(
            self.cfg.normalization.contact_force_range
        )
        all_contact_forces = (
            self.contact_forces.reshape(self.num_envs, -1) - force_shift
        ) * force_scale
        if all_contact_forces.shape[-1] != 51:
            raise RuntimeError(
                "The complete Nezha privileged frame requires 17 rigid bodies "
                f"(51 contact-force values), got {all_contact_forces.shape[-1]}"
            )
        actual_p_gains = self.Kp_factors * self.p_gains.unsqueeze(0)
        actual_d_gains = self.Kd_factors * self.d_gains.unsqueeze(0)
        privileged = torch.cat(
            (
                actor_clean,
                dof_acc * self.obs_scales.dof_acc,
                self.torques * self.obs_scales.torques,
                self.mine_base_mass - self.mine_base_mass.mean(),
                self.mine_base_com,
                actual_p_gains / 20.0,
                actual_d_gains / 0.2,
                all_contact_forces,
                terrain_heights,
                self.base_lin_vel * self.obs_scales.lin_vel,
                base_height * self.obs_scales.height_measurements,
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

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        out_of_limits[:, self.wheel_indices] = 0.0
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5.0 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_feet_contact_uniform(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_mean = torch.mean(contact.float(), dim=1, keepdim=True)
        return torch.sum(torch.abs(contact.float() - contact_mean), dim=1)

    def _reward_smoothness(self):
        second_difference = self.actions - 2.0 * self.last_actions + self.last_last_actions
        second_difference[:, self.wheel_indices] = 0.0
        second_difference *= self.cfg.control.action_scale
        valid = (self.last_actions != 0.0) & (self.last_last_actions != 0.0)
        return torch.sum(torch.square(second_difference) * valid.float(), dim=1)

    def _reward_joint_power(self):
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_wheel_contact_vel_y(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        penalty = torch.square(self.feet_vel[:, :, 1]) * contact.float()
        return torch.mean(penalty, dim=1)
