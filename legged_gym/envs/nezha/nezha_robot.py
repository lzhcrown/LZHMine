"""Nezha robot dynamics, randomization and rewards shared by the CTS task."""

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    quat_apply,
    quat_from_angle_axis,
    quat_rotate,
    quat_rotate_inverse,
    torch_rand_float,
)

from legged_gym.envs.base.legged_robot import LeggedRobot


class NezhaRobot(LeggedRobot):
    """Nezha mixed leg-position and wheel-velocity control."""

    LEG_JOINT_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        cfg.domain_rand.rand_interval = np.ceil(
            cfg.domain_rand.rand_interval_s / self.dt
        )

    def _init_buffers(self):
        super()._init_buffers()
        self.fail_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.envs_steps_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.joint_pos_target = torch.zeros_like(self.dof_pos)
        self.last_joint_pos_target = torch.zeros_like(self.dof_pos)
        self.last_last_joint_pos_target = torch.zeros_like(self.dof_pos)
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.base_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.rigid_body_external_forces = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
        )
        self.rigid_body_external_torques = torch.zeros_like(
            self.rigid_body_external_forces
        )
        wheel_names = list(self.cfg.asset.feet_names)
        indices = [self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], name
        ) for name in wheel_names]
        if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("Nezha wheel contact bodies must be distinct valid rigid bodies")
        self.wheel_body_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.num_wheels = len(wheel_names)

    def check_termination(self):
        fail = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        # Preserve the snapshot's double increment for contact failures.
        self.fail_buf += fail
        fail |= self.projected_gravity[:, 2] > -0.5
        self.fail_buf *= fail
        self.fail_buf += fail
        self.time_out_buf.copy_(self.episode_length_buf > self.max_episode_length)
        self.reset_buf.copy_((
            self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt
        ) | self.time_out_buf)

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        clip = self.cfg.rewards.clip_single_reward * self.dt
        for name, reward_function in zip(
            self.reward_names, self.reward_functions
        ):
            reward = reward_function() * self.reward_scales[name]
            reward = torch.clip(reward, -clip, clip)
            self.rew_buf += reward
            self.episode_sums[name] += reward
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)
        if "termination" in self.reward_scales:
            reward = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += reward
            self.episode_sums["termination"] += reward

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._resample_commands(env_ids)
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)

        self._randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_last_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_joint_pos_target[env_ids] = 0.0
        self.last_last_joint_pos_target[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids])
                / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges[
                "lin_vel_x"
            ][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.obs_buf[env_ids] = 0.0
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        if self.cfg.init_state.rand_init_dof:
            self.dof_pos[env_ids] += torch_rand_float(
                -self.cfg.init_state.rand_init_dof_range,
                self.cfg.init_state.rand_init_dof_range,
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.custom_origins:
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )
        initial_yaws = torch_rand_float(
            -self.cfg.terrain.yaw_init_range,
            self.cfg.terrain.yaw_init_range,
            (len(env_ids), 1),
            device=self.device,
        )
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.root_states[env_ids, 3:7] = quat_from_angle_axis(
            initial_yaws, yaw_axis
        )[:, 0, :]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2],
            dim=1,
        )
        move_up = distance > self.terrain.env_length / 2
        move_down = (
            self.episode_sums["tracking_lin_vel"][env_ids]
            / self.max_episode_length_s
            < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.5
        ) & ~move_up
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(
                self.terrain_levels[env_ids], self.max_terrain_level
            ),
            torch.clip(self.terrain_levels[env_ids], min=0),
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def update_command_curriculum(self, env_ids):
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.2,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.2,
                0.0,
                self.cfg.commands.max_curriculum,
            )
        if (
            torch.mean(self.episode_sums["tracking_ang_vel"][env_ids])
            / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_ang_vel"]
        ):
            self.command_ranges["ang_vel_yaw"][0] = np.clip(
                self.command_ranges["ang_vel_yaw"][0] - 0.1,
                -self.cfg.commands.max_curriculum_yaw,
                0.0,
            )
            self.command_ranges["ang_vel_yaw"][1] = np.clip(
                self.command_ranges["ang_vel_yaw"][1] + 0.1,
                0.0,
                self.cfg.commands.max_curriculum_yaw,
            )

    def _post_physics_step_callback(self):
        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.dt)
            == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * (self.commands[:, 3] - heading), -1.0, 1.0
            )
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots:
            self._push_robots()
        randomize_ids = (
            self.episode_length_buf % int(self.cfg.domain_rand.rand_interval)
            == 0
        ).nonzero(as_tuple=False).flatten()
        self._randomize_dof_props(randomize_ids)
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )

    def _push_robots(self):
        env_ids = (
            self.envs_steps_buf
            % int(self.cfg.domain_rand.push_interval_s / self.cfg.sim.dt)
            == 0
        ).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        max_push_force = (
            self.robot_base_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0.0
        random_forces = torch_rand_float(
            -max_push_force,
            max_push_force,
            (self.num_envs, 3),
            device=self.device,
        )
        self.rigid_body_external_forces[env_ids, 0, 0:3] = quat_rotate(
            self.base_quat[env_ids], random_forces[env_ids]
        )
        self.rigid_body_external_forces[env_ids, 0, 2] *= 0.5
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device
            )
            low, high = self.cfg.domain_rand.joint_friction_range
            self.joint_friction_coeffs = torch_rand_float(
                low, high, (self.num_envs, 1), device=self.device
            )
            for index in range(len(props)):
                self.dof_pos_limits[index, 0] = props["lower"][index].item()
                self.dof_pos_limits[index, 1] = props["upper"][index].item()
                self.dof_vel_limits[index] = props["velocity"][index].item()
                self.torque_limits[index] = props["effort"][index].item()
                midpoint = torch.mean(self.dof_pos_limits[index])
                width = (
                    self.dof_pos_limits[index, 1]
                    - self.dof_pos_limits[index, 0]
                )
                self.dof_pos_limits[index, 0] = midpoint - (
                    0.5 * width * self.cfg.rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[index, 1] = midpoint + (
                    0.5 * width * self.cfg.rewards.soft_dof_pos_limit
                )
        for index in range(len(props)):
            props["friction"][index] = (
                self.cfg.domain_rand.default_joint_friction[index]
                * self.joint_friction_coeffs[env_id, 0].item()
            )
            props["damping"][index] = self.cfg.domain_rand.default_joint_damping[
                index
            ]
            props["armature"][index] = self.cfg.domain_rand.default_joint_armature[
                index
            ]
        return props

    def _randomize_dof_props(self, env_ids):
        if len(env_ids) == 0 or not self.cfg.domain_rand.randomize_joint_friction:
            return
        low, high = self.cfg.domain_rand.joint_friction_range
        self.joint_friction_coeffs[env_ids] = torch_rand_float(
            low, high, (len(env_ids), 1), device=self.device
        )

    def _refresh_actor_dof_props(self, env_ids):
        if not self.cfg.domain_rand.randomize_joint_friction:
            return
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            props = self.gym.get_actor_dof_properties(
                self.envs[env_id], self.actor_handles[env_id]
            )
            for index in range(self.num_dof):
                props["friction"][index] = (
                    self.cfg.domain_rand.default_joint_friction[index]
                    * self.joint_friction_coeffs[env_id, 0].item()
                )
            self.gym.set_actor_dof_properties(
                self.envs[env_id], self.actor_handles[env_id], props
            )

    def _process_rigid_body_props(self, props, env_id):
        # Add displacement to the URDF COM, as in wheel_gym_CQ.
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass += self.payload[env_id, 0].item()
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(
                props[0].com.x + self.com_displacement[env_id, 0].item(),
                props[0].com.y + self.com_displacement[env_id, 1].item(),
                props[0].com.z + self.com_displacement[env_id, 2].item(),
            )
        if not hasattr(self, "robot_base_mass"):
            self.robot_base_mass = torch.zeros(
                self.num_envs, 1, dtype=torch.float32, device=self.device
            )
            self.robot_base_com = torch.zeros(
                self.num_envs, 3, dtype=torch.float32, device=self.device
            )
        self.robot_base_mass[env_id, 0] = props[0].mass
        self.robot_base_com[env_id] = torch.tensor(
            [props[0].com.x, props[0].com.y, props[0].com.z],
            dtype=torch.float32,
            device=self.device,
        )
        return props

    def _compute_torques(self, actions):
        dof_error = self.default_dof_pos - self.dof_pos
        dof_error[:, self.wheel_indices] = 0.0
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.wheel_indices] = 0.0
        self.joint_pos_target.copy_(self.default_dof_pos + actions_scaled)
        velocity_reference = torch.zeros_like(actions_scaled)
        wheel_velocity = actions * self.cfg.control.vel_scale
        velocity_reference[:, self.wheel_indices] = wheel_velocity[
            :, self.wheel_indices
        ]
        if self.cfg.control.control_type == "P":
            torques = self.p_gains * (actions_scaled + dof_error) + self.d_gains * (
                velocity_reference - self.dof_vel
            )
        elif self.cfg.control.control_type == "V":
            torques = self.p_gains * (
                actions_scaled - self.dof_vel
            ) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif self.cfg.control.control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(
                f"Unknown controller type: {self.cfg.control.control_type}"
            )

        torque_scale = torch_rand_float(
            0.8,
            1.1,
            (self.num_envs, self.num_dof),
            device=self.device,
        )
        torques *= torque_scale
        if self.cfg.domain_rand.randomize_motor_strength:
            torques *= self.motor_strength_factors
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_dof_vel(self):
        # Penalize leg rates without corrupting measured wheel velocities.
        return torch.sum(torch.square(self.dof_vel[:, self.LEG_JOINT_IDS]), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(
            self.dof_pos - self.dof_pos_limits[:, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self.dof_pos - self.dof_pos_limits[:, 1]
        ).clip(min=0.0)
        out_of_limits[:, self.wheel_indices] = 0.0
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5.0
            * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filtered = torch.logical_or(contact, self.last_contacts)
        self.last_contacts.copy_(contact)
        first_contact = (self.feet_air_time > 0.0) * contact_filtered
        self.feet_air_time += self.dt
        reward = torch.sum(
            (self.feet_air_time - 0.6) * first_contact, dim=1
        )
        reward *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filtered
        return reward

    def _reward_feet_contact_uniform(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_mean = torch.mean(contact.float(), dim=1)
        return torch.sum(
            torch.abs(contact.float() - contact_mean.unsqueeze(1)), dim=1
        )

    def _reward_smoothness(self):
        difference = torch.square(
            self.joint_pos_target
            - 2.0 * self.last_joint_pos_target
            + self.last_last_joint_pos_target
        )
        difference *= self.last_actions != 0.0
        difference *= self.last_last_actions != 0.0
        return torch.sum(difference, dim=1)

    def _reward_joint_power(self):
        return torch.sum(
            torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1
        )

    def _reward_wheel_contact_vel_y(self):
        if self.num_wheels == 0:
            return torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        contact = self.contact_forces[:, self.wheel_body_indices, 2] > 1.0
        rigid_body_states = self.rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )
        wheel_velocity_y = rigid_body_states[
            :, self.wheel_body_indices, 8
        ]
        penalty = torch.square(wheel_velocity_y) * contact.float()
        return torch.mean(penalty, dim=1)
