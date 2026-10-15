"""Isaac Gym Nezha task using the WF concurrent teacher-student interface."""

from copy import deepcopy

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.nezha.nezha_robot import NezhaRobot
from rsl_rl.utils.cts_observations import (
    CTSObservationHistory,
    build_privileged_observations,
    build_proprioception,
    contact_history_features,
    default_observation_config,
    observation_dimensions,
    validate_observation_config,
)


class NezhaMoECTSEnv(NezhaRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        provided = getattr(cfg.env, "observation_config", None)
        self.observation_config = deepcopy(
            default_observation_config(history_length=cfg.env.frame_stack)
            if provided is None else provided
        )
        validate_observation_config(self.observation_config)
        current, history, privileged = observation_dimensions(self.observation_config)
        cfg.env.num_one_step_observations = current
        cfg.env.num_observations = history
        cfg.env.num_one_step_privileged_obs = privileged
        cfg.env.num_privileged_obs = privileged
        cfg.env.frame_stack = self.observation_config["history_length"]
        cfg.env.critic_frame_stack = 1
        cfg.env.history_order = self.observation_config["history_order"]
        cfg.env.history_layout = self.observation_config["history_layout"]
        cfg.env.observation_config = deepcopy(self.observation_config)
        if not cfg.terrain.measure_heights:
            raise ValueError("The WF teacher observation requires terrain height measurements")
        if self.observation_config.get("contact_history_length") != 3:
            raise ValueError("The WF contact observation requires three physics-step samples")
        if self.observation_config.get("height_grid_order") != "xy":
            raise ValueError("The WF height scan requires xy grid ordering")
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        if sorted(self.wheel_indices.tolist()) != sorted(self.observation_config["wheel_indices"]):
            raise ValueError("Observation wheel indices do not match Nezha's DOF mapping")
        self.observation_history = CTSObservationHistory(
            self.num_envs, self.observation_config, device=self.device
        )
        self.physics_dof_vel = self.dof_vel.clone()
        self.wheel_contact_history = torch.zeros(
            self.num_envs, 3, self.num_wheels, 3, device=self.device
        )
        self.current_obs = torch.zeros(self.num_envs, self.num_one_step_obs, device=self.device)

    def _get_noise_scale_vec(self, cfg):
        # The base allocator calls this; actual noise is applied per raw term
        # by the shared observation builder, before clipping and scaling.
        self.add_noise = cfg.noise.add_noise
        return torch.zeros(cfg.env.num_one_step_observations, device=self.device)

    def _init_height_points(self):
        # Isaac Lab GridPatternCfg's xy order varies x fastest. Gym's default
        # meshgrid(x,y) varies y fastest, so transpose its conceptual layout.
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        self.num_height_points = x.numel() * y.numel()
        if self.num_height_points != self.cfg.env.num_height_observations:
            raise ValueError("Terrain grid size differs from num_height_observations")
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device)
        points[:, :, 0] = x.repeat(y.numel())
        points[:, :, 1] = y.repeat_interleave(x.numel())
        return points

    def step(self, actions):
        self.envs_steps_buf += self.cfg.control.decimation
        clip = self.cfg.normalization.clip_actions
        # Keep persistent buffers allocated outside inference_mode. Replacing
        # them during rollout would make a later external reset/load fail.
        self.actions.copy_(actions.to(self.device).clamp(-clip, clip))
        self.delayed_actions = self.actions.unsqueeze(1).repeat(1, self.cfg.control.decimation, 1)
        if self.cfg.domain_rand.delay:
            delays = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
            for substep in range(self.cfg.control.decimation):
                self.delayed_actions[:, substep] = self.last_actions + (
                    self.actions - self.last_actions
                ) * (substep >= delays)
        self.render()
        for substep in range(self.cfg.control.decimation):
            self.torques.copy_(self._compute_torques(self.delayed_actions[:, substep]).view(self.torques.shape))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.dof_acc[:] = (self.dof_vel - self.physics_dof_vel) / self.sim_params.dt
            self.physics_dof_vel.copy_(self.dof_vel)
            self.wheel_contact_history[:, 1:] = self.wheel_contact_history[:, :-1].clone()
            self.wheel_contact_history[:, 0] = self.contact_forces[:, self.wheel_body_indices]
        termination_ids, termination_privileged = self.post_physics_step()
        return (self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf,
                self.extras, termination_ids, termination_privileged)

    def _refresh_base_kinematics(self):
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._refresh_base_kinematics()
        bodies = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        self.feet_pos = bodies[:, self.feet_indices, 0:3]
        self.feet_vel = bodies[:, self.feet_indices, 7:10]
        self._post_physics_step_callback()
        # Push randomization changes root velocity inside the callback.
        self._refresh_base_kinematics()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations()
        self.disturbance.zero_()
        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(self.actions)
        self.last_last_joint_pos_target.copy_(self.last_joint_pos_target)
        self.last_joint_pos_target.copy_(self.joint_pos_target)
        self.last_dof_vel.copy_(self.dof_vel)
        self.last_root_vel.copy_(self.root_states[:, 7:13])
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        return env_ids, termination_privileged

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.observation_history.reset(env_ids)
        self.physics_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.dof_acc[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.wheel_contact_history[env_ids] = 0.0
        self.contact_forces[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.joint_pos_target[env_ids] = 0.0
        # Indexed setters already update root/dof tensors. Recompute derived
        # state from those reset values, without re-reading stale Gym tensors.
        self._refresh_base_kinematics()
        self.measured_heights = self._get_heights()
        self.base_height = (self.root_states[:, 2:3] - self.measured_heights).mean(dim=-1)

    def _raw_observations(self):
        return {
            "base_lin_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "projected_gravity": self.projected_gravity,
            "velocity_commands": self.commands[:, :3],
            "joint_pos": self.dof_pos - self.default_dof_pos,
            "joint_vel": self.dof_vel,
            "actions": self.actions,
            "joint_acc": self.dof_acc,
            "joint_torque": self.torques,
            "contact_force": contact_history_features(self.wheel_contact_history),
            "height_scan": self.root_states[:, 2:3] - self.observation_config["height_offset"] - self.measured_heights,
        }

    def compute_observations(self):
        raw = self._raw_observations()
        self.current_obs.copy_(build_proprioception(
            raw, self.observation_config, add_noise=self.add_noise,
            noise_level=self.cfg.noise.noise_level,
        ))
        self.obs_buf.copy_(self.observation_history.update(self.current_obs))
        self.privileged_obs_buf.copy_(build_privileged_observations(raw, self.observation_config))

    def get_current_obs(self):
        return self.current_obs

    def compute_termination_observations(self, env_ids):
        if env_ids.numel() == 0:
            return self.privileged_obs_buf.new_zeros((0, self.num_privileged_obs))
        return build_privileged_observations(self._raw_observations(), self.observation_config)[env_ids]
