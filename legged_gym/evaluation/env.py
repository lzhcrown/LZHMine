# Adapted from wheel_gym_CQ/wheel_legged_gym/evaluation/env.py
"""Independent, fixed-terrain evaluation with measurements before automatic reset."""
import math
from types import SimpleNamespace

import numpy as np
import torch

from legged_gym.envs.nezha.nezha_moe_cts_env import NezhaMoECTSEnv
from .metrics import VectorEpisodeMetrics
from .timing import attempt_step_limit


class EvaluationEnv(NezhaMoECTSEnv):
    """Run independent traversals for each simulation slot up to its optional quota.

    Construction does not start a measured episode. Call ``reset()`` once, then
    step until ``evaluation_complete``. Each slot keeps its terrain type and its
    own spawn RNG, including when other robots finish earlier. The inherited
    actor factory uses collision group ``env_id``, so robots cannot collide.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 evaluation_settings=None, episode_sink=None, trajectory_sink=None, physics_sink=None):
        self.evaluation = (SimpleNamespace(**evaluation_settings)
                           if isinstance(evaluation_settings, dict)
                           else evaluation_settings or cfg.evaluation)
        self.metrics = None
        self.completed_episodes = []
        self.episode_sink = episode_sink
        self._evaluation_started = False
        quota = getattr(self.evaluation, "episodes_per_env", 1)
        self._episode_quota = math.inf if quota is None else int(quota)
        if self._episode_quota < 1:
            raise ValueError("episodes_per_env must be positive")
        self._seed = int(getattr(self.evaluation, "seed", 0))
        self._spawn_rngs = [np.random.default_rng(self._seed + i)
                            for i in range(cfg.env.num_envs)]
        self._validate_evaluation_cfg(cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Discard the constructor's unmeasured initialization draws.
        self._spawn_rngs = [np.random.default_rng(self._seed + i)
                            for i in range(self.num_envs)]
        self.episode_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Raw-data attempt IDs advance even when an explicit reset interrupts a
        # trial; episode_counts alone tracks the completed-trial quota.
        self.attempt_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._episode_status = [None] * self.num_envs
        self._episode_reason = [""] * self.num_envs
        self.measurement_joint_names = list(self.gym.get_actor_dof_names(self.envs[0], self.actor_handles[0]))
        if (self.measurement_joint_names != list(self.dof_names) or
                len(self.measurement_joint_names) != self.num_dof or
                self.num_actions != self.num_dof or self.torques.shape[1] != self.num_dof):
            raise ValueError("Recorded joint columns must match the actor DOF and applied torque ordering")
        self.metrics = VectorEpisodeMetrics(
            self.num_envs, self.device,
            trajectory_stride=int(getattr(self.evaluation, "trajectory_stride", 1)),
            trajectory_sink=trajectory_sink, physics_sink=physics_sink, num_joints=self.num_dof,
        )

    @staticmethod
    def _validate_evaluation_cfg(cfg):
        if cfg.terrain.mesh_type not in ("trimesh", "heightfield"):
            raise ValueError("Evaluation terrain requires trimesh or heightfield")
        if cfg.terrain.curriculum or cfg.commands.curriculum or cfg.commands.heading_command:
            raise ValueError("Evaluation requires fixed terrain and body-frame commands")
        if cfg.noise.add_noise or cfg.init_state.rand_init_dof:
            raise ValueError("Evaluation requires observation noise and initial DOF randomization off")
        if getattr(cfg.domain_rand, "randomize_torque_scale", True):
            raise ValueError("Evaluation must explicitly disable randomize_torque_scale")

    @property
    def evaluation_complete(self):
        return bool(self._episode_quota != math.inf and self._evaluation_started and
                    torch.all(self.episode_counts >= self._episode_quota).item())

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.max_episode_length = attempt_step_limit(self.max_episode_length_s, self.dt)

    def _get_env_origins(self):
        self.custom_origins = True
        self.terrain_types = torch.arange(self.num_envs, device=self.device) % len(self.terrain.type_names)
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_terrain_level = 1
        self.terrain_origins = torch.as_tensor(self.terrain.env_origins,
                                              dtype=torch.float, device=self.device)
        self.env_origins = self.terrain_origins[0, self.terrain_types].clone()
        self.traversal_bounds = torch.as_tensor(self.terrain.tile_bounds,
                                               dtype=torch.float, device=self.device)[self.terrain_types]
        bounds = self.terrain.tile_bounds
        self.terrain_x_min, self.terrain_x_max = float(bounds[:, 0].min()), float(bounds[:, 1].max())
        self.terrain_y_min, self.terrain_y_max = float(bounds[:, 2].min()), float(bounds[:, 3].max())
        self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_directions = torch.zeros(self.num_envs, 2, device=self.device)
        self.spawn_positions = torch.zeros(self.num_envs, 3, device=self.device)

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if not hasattr(self, "total_mass_kg"):
            self.total_mass_kg = torch.zeros(self.num_envs, device=self.device)
        # COT uses the whole articulated robot, not the base-body mass feature.
        self.total_mass_kg[env_id] = sum(float(body.mass) for body in props)
        return props

    def _resample_commands(self, env_ids):
        self.commands[env_ids] = 0.
        self.commands[env_ids, 0] = float(getattr(self.evaluation, "command_vx", 1.0))

    def _post_physics_step_callback(self):
        # No command resampling, teleportation, curriculum, pushes, or randomizers.
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height = (self.root_states[:, 2:3] - self.measured_heights).mean(dim=1)

    def step(self, actions):
        if not self._evaluation_started:
            raise RuntimeError("Call reset() before evaluation step()")
        self._step_commands = self.commands[:, :3].clone()
        if self._episode_quota != math.inf:
            actions = actions.clone()
            actions[self.episode_counts >= self._episode_quota] = 0.
        return super().step(actions)

    def _before_physics_substep(self):
        if self.metrics is not None and self._evaluation_started and not getattr(self.evaluation, 'no_record', False):
            # Read actual simulator velocity; applied torque includes clipping.
            self._omega_before = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1].clone()

    def _after_physics_substep(self):
        if self.metrics is not None and self._evaluation_started and not getattr(self.evaluation, 'no_record', False):
            omega_after = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
            self.metrics.physics_step(self.torques, self._omega_before, omega_after,
                                      self.sim_params.dt)

    def _sync_reset_state(self, env_ids):
        self._refresh_base_kinematics()
        self.dof_acc[env_ids] = 0.
        self.torques[env_ids] = 0.
        self.measured_heights = self._get_heights()
        self.base_height = (self.root_states[:, 2:3] - self.measured_heights).mean(dim=1)

    def _raw_episode_metadata(self, env_id):
        gravity = self.sim_params.gravity
        vector = [float(gravity.x), float(gravity.y), float(gravity.z)]
        return {
            "dof_names": list(self.measurement_joint_names),
            "mass_kg": float(self.total_mass_kg[env_id]),
            "gravity_m_s2": math.sqrt(sum(value * value for value in vector)),
            "gravity_vector_m_s2": vector,
            "goal_direction_xy": self.goal_directions[env_id].detach().cpu().tolist(),
        }

    def _raw_state_metadata(self, env_id, prefix):
        def values(tensor):
            return [value if math.isfinite(value) else None for value in tensor.detach().cpu().tolist()]
        return {
            prefix + "_base_position": values(self.root_states[env_id, :3]),
            prefix + "_velocity_world": values(self.root_states[env_id, 7:10]),
            prefix + "_velocity_body": values(self.base_lin_vel[env_id]),
            prefix + "_angular_velocity_body": values(self.base_ang_vel[env_id]),
            prefix + "_angular_velocity_world": values(self.root_states[env_id, 10:13]),
            prefix + "_orientation_xyzw": values(self.root_states[env_id, 3:7]),
        }

    def reset(self):
        if self._evaluation_started:
            raise RuntimeError("Evaluation already started; automatic resets manage episode quotas")
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        self._evaluation_started = True
        self._start(env_ids)
        self.compute_observations()
        clip = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clamp(self.obs_buf, -clip, clip)
        self.privileged_obs_buf = torch.clamp(self.privileged_obs_buf, -clip, clip)
        return self.obs_buf, self.privileged_obs_buf

    def compute_reward(self):
        self.rew_buf.zero_()
