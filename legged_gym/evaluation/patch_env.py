# Adapted from wheel_gym_CQ/wheel_legged_gym/evaluation/patch_env.py
"""Measure a terrain attempt from wheel entry to exit or settled obstacle landing."""
import math
from contextlib import contextmanager

import numpy as np
from isaacgym import gymtorch
import torch

from legged_gym.envs.nezha.nezha_moe_cts_env import NezhaMoECTSEnv
from .crossing import WheelWindowCrossing
from .env import EvaluationEnv
from .geometry import wheel_geometry_from_cfg
from .patch_terrain import PatchEvaluationTerrain
from .traversal import ObstacleTraversal, wheel_centers_and_extents


def _wheel_bounds_xy(center, extent):
    return torch.stack((center[..., 0] - extent[..., 0], center[..., 0] + extent[..., 0],
                        center[..., 1] - extent[..., 1], center[..., 1] + extent[..., 1]), dim=-1)


def project_wheel_cylinders(rigid_states, collision_offsets, collision_axes, radii, widths):
    """Exact XY support bounds of oriented collision cylinders in FL/FR/RL/RR order.

    Rigid states contain link poses, not wheel DOF indices or robot-root poses.
    This projection intentionally permits an airborne wheel to cross the patch.
    """
    return _wheel_bounds_xy(*wheel_centers_and_extents(rigid_states, collision_offsets, collision_axes, radii, widths))


class PatchEvaluationEnv(EvaluationEnv):
    """Approach is an attempt; only the wheel-crossing window has raw samples.

    A control step which first detects entry is excluded in full. Recording
    starts at its endpoint and remains active through any gap between axles.
    Exit/failure is recorded before reset. The boundary resolution is one policy
    step; the implementation does not claim substep-interpolated crossing times.
    """

    joint_angle_trace = None

    @contextmanager
    def record_joint_angles(self, path, policy):
        from legged_gym.scripts._joint_angles import open_joint_angles

        terrains = [self.terrain.type_names[index] for index in self.terrain_types.tolist()]
        with open_joint_angles(path, self, self.evaluation.task, terrains,
                               student_experts=policy.expert_num, actor_experts=policy.actor_expert_num) as trace:
            self.joint_angle_trace = trace
            try:
                yield trace
            finally:
                self.joint_angle_trace = None

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 evaluation_settings=None, episode_sink=None, trajectory_sink=None, physics_sink=None):
        settings = evaluation_settings or cfg.evaluation
        geometry = settings.get("wheel_geometry") if isinstance(settings, dict) else getattr(settings, "wheel_geometry", None)
        self.wheel_geometry = geometry or wheel_geometry_from_cfg(cfg)
        for name in ("left_wheel_y", "right_wheel_y", "front_wheel_x", "rear_wheel_x",
                     "wheel_radius", "wheel_x_min", "wheel_x_max", "wheel_y_min", "wheel_y_max"):
            setattr(cfg.terrain, name, self.wheel_geometry[name])
        self.crossing = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless,
                         evaluation_settings, episode_sink, trajectory_sink, physics_sink)
        indices = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                   for name in self.wheel_geometry["wheel_names"]]
        if any(index < 0 for index in indices) or len(set(indices)) != 4:
            raise ValueError("Patch evaluation requires four distinct wheel rigid-body handles")
        self.measurement_wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        for name in ("collision_offsets", "collision_axes", "radii", "widths",
                     "center_offsets", "default_half_extents"):
            setattr(self, "_wheel_" + name, torch.tensor(self.wheel_geometry[name],
                                                        dtype=torch.float, device=self.device))
        self.crossing = WheelWindowCrossing(self.num_envs, self.device, exit_rule="opposite_or_sides")
        self._init_obstacle_traversal()
        self.attempt_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._attempt_metadata = [{} for _ in range(self.num_envs)]

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self.terrain = PatchEvaluationTerrain(self.cfg.terrain, self.num_envs)
        if self.cfg.terrain.mesh_type == "trimesh":
            self._create_trimesh()
        else:
            self._create_heightfield()
        self._create_envs()

    def _get_env_origins(self):
        super()._get_env_origins()
        self.patch_bounds = torch.as_tensor(self.terrain.patch_bounds, dtype=torch.float,
                                           device=self.device)[self.terrain_types]
        self.selected_sides = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        sign = 1. if float(getattr(self.evaluation, "command_vx", 1.)) > 0 else -1.
        self.travel_signs = torch.full((self.num_envs,), sign, device=self.device)

    def _reset_root_states(self, env_ids):
        for env_id in env_ids.tolist():
            replica = env_id // len(self.terrain.type_names)
            index = int(self.episode_counts[env_id]) if hasattr(self, "episode_counts") else 0
            option = getattr(self.evaluation, "wheel_side", "all")
            if option not in ("all", "left", "right", "both"):
                raise ValueError("wheel_side must be all, left, right, or both")
            side = ("left" if (replica + index) % 2 == 0 else "right") if option == "both" else option
            spawn_side = "center" if side == "all" else side
            sign = int(self.travel_signs[env_id])
            sample = self.terrain.sample_spawns(
                np.asarray([int(self.terrain_types[env_id])]), self._spawn_rngs[env_id],
                direction=sign, side=spawn_side,
                xy_jitter=float(getattr(self.evaluation, "spawn_xy_jitter", 0.)))
            ground = torch.as_tensor(sample["ground_positions"][0], device=self.device,
                                     dtype=self.root_states.dtype)
            self.root_states[env_id] = self.base_init_state
            self.root_states[env_id, :3] = ground
            self.root_states[env_id, 2] += self.base_init_state[2]
            # Reverse motion changes the commanded velocity, never the body yaw.
            self.root_states[env_id, 3:7] = self.root_states.new_tensor([0., 0., 0., 1.])
            self.root_states[env_id, 7:13] = 0.
            self.spawn_positions[env_id] = self.root_states[env_id, :3]
            self.goal_positions[env_id] = torch.as_tensor(sample["goal_positions"][0], device=self.device)
            self.goal_directions[env_id] = self.root_states.new_tensor([sign, 0.])
            self.traversal_bounds[env_id] = torch.as_tensor(sample["bounds"][0], device=self.device)
            self.patch_bounds[env_id] = torch.as_tensor(sample["patch_bounds"][0], device=self.device)
            self.selected_sides[env_id] = {"left": 0, "right": 1, "all": 2}[side]
        indices = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(indices), len(indices))
        self._resample_commands(env_ids)

    def _initial_wheel_bounds(self, env_ids):
        # Indexed root/DOF reset has not refreshed rigid_state yet. FK of the exact
        # reset pose supplies the initial edge positions without a hidden sim step.
        centers = self.root_states[env_ids, None, :3] + self._wheel_center_offsets[None]
        extent = self._wheel_default_half_extents[None]
        return torch.stack((centers[..., 0] - extent[..., 0], centers[..., 0] + extent[..., 0],
                            centers[..., 1] - extent[..., 1], centers[..., 1] + extent[..., 1]), dim=-1)

    def _init_obstacle_traversal(self):
        names = self.terrain.type_names
        required = [name.startswith(("stairs_", "platform_")) for name in names]
        self.obstacle_traversal = None
        if not any(required):
            return
        self.obstacle_traversal = ObstacleTraversal(self.num_envs, self.dt, self.device)
        self.terrain.metadata["obstacle_landing"] = {
            "settle_time_s": ObstacleTraversal.SETTLE_TIME_S,
            "required_control_steps": self.obstacle_traversal.required_steps,
            "wheel_bottom_height_tolerance_m": ObstacleTraversal.HEIGHT_TOLERANCE_M,
            "minimum_supported_wheels": 3, "upward_force_threshold_N": ObstacleTraversal.SUPPORT_FORCE_N,
            "maximum_gravity_z": ObstacleTraversal.MAX_GRAVITY_Z,
            "maximum_abs_world_vz_m_s": ObstacleTraversal.MAX_VERTICAL_SPEED_M_S,
            "maximum_roll_pitch_rate_norm_rad_s": ObstacleTraversal.MAX_TILT_RATE_RAD_S,
            "exit_edge": "For stable_landing, xmin/xmax identifies the far face of the obstacle. exit_time_s records the settled landing time.",
        }
        self.requires_landing = torch.tensor(required, device=self.device)[self.terrain_types]
        top_half, outer_half, heights = [], [], []
        for name, tile in zip(names, self.terrain.metadata["tiles"]):
            if name.startswith("stairs_"):
                top_half.append(tile["central_platform_size_m"] / 2)
                outer_half.append(tile["patch_length_m"] / 2 - tile["tread_width_m"])
                heights.append(tile["central_platform_height_m"])
            else:
                half = tile.get("platform_size_xy_m", [0.])[0] / 2
                top_half.append(half)
                outer_half.append(half)
                heights.append(tile.get("platform_height_m", 0.))
        self.middle_half_width = self.patch_bounds.new_tensor(top_half)[self.terrain_types, None]
        self.obstacle_half_width = self.patch_bounds.new_tensor(outer_half)[self.terrain_types, None]
        self.middle_height = self.patch_bounds.new_tensor(heights)[self.terrain_types, None]

    def _landed_on_far_side(self, bounds, bottom_z, support_z):
        x = self.patch_bounds[:, :2].mean(dim=1, keepdim=True)
        y = self.patch_bounds[:, 2:].mean(dim=1, keepdim=True)
        x0, x1, y0, y1 = bounds.unbind(dim=-1)
        middle = self.middle_half_width
        outer = self.obstacle_half_width
        tolerance = self.obstacle_traversal.HEIGHT_TOLERANCE_M
        on_middle = ((x0 >= x - middle) & (x1 <= x + middle) &
                     (y0 >= y - middle) & (y1 <= y + middle) &
                     ((bottom_z - self.middle_height).abs() <= tolerance))
        in_lane = ((y0 >= y - outer) & (y1 <= y + outer)).all(dim=1)
        far_x = torch.where(self.travel_signs[:, None] > 0,
                            (x0 >= x + outer) & (x1 <= self.traversal_bounds[:, 1:2]),
                            (x1 <= x - outer) & (x0 >= self.traversal_bounds[:, 0:1]))
        # The external pads and the approach/exit beyond the outermost riser
        # are at z=0 for both signed stairs and every fixed square platform.
        on_far_ground = far_x & (bottom_z.abs() <= tolerance)
        return self.obstacle_traversal.update(on_middle, on_far_ground, in_lane, support_z,
                                              self.projected_gravity[:, 2], self.root_states[:, 9],
                                              self.base_ang_vel[:, :2])

    def _current_wheel_geometry(self):
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.measurement_wheel_indices]
        centers, extents = wheel_centers_and_extents(states, self._wheel_collision_offsets,
                                                     self._wheel_collision_axes, self._wheel_radii, self._wheel_widths)
        bounds = _wheel_bounds_xy(centers, extents)
        forces = self.contact_forces[:, self.measurement_wheel_indices]
        contact = torch.linalg.vector_norm(forces, dim=-1)
        contact = contact > float(getattr(self.evaluation, "contact_force_threshold", 1.))
        # Diagnostic ground contact while the wheel's XY projection overlaps the
        # patch. Net force does not locate the contact point: it could be on the
        # adjacent platform. This flag is never a requirement for successful exit.
        p = self.patch_bounds[:, None]
        overlap = ((bounds[..., 1] >= p[..., 0]) & (bounds[..., 0] <= p[..., 1]) &
                   (bounds[..., 3] >= p[..., 2]) & (bounds[..., 2] <= p[..., 3]))
        landing = None
        if self.obstacle_traversal is not None:
            landed = self._landed_on_far_side(bounds, centers[..., 2] - extents[..., 2], forces[..., 2])
            landing = self.requires_landing, landed
        return bounds, contact & overlap, landing

    def _start(self, env_ids):
        """Start approaches without recording measurement-window samples."""
        if not len(env_ids):
            return
        self.crossing.reset(env_ids, window_bounds=self.patch_bounds[env_ids],
                            travel_direction=self.travel_signs[env_ids],
                            selected_side=self.selected_sides[env_ids],
                            wheel_bounds=self._initial_wheel_bounds(env_ids))
        if self.obstacle_traversal is not None:
            self.obstacle_traversal.reset(env_ids)
        self.attempt_indices[env_ids] += 1
        self.attempt_active[env_ids] = True
        for env_id in env_ids.tolist():
            self._episode_status[env_id], self._episode_reason[env_id] = None, ""
            self._attempt_metadata[env_id] = {
                "env_id": env_id, "episode_index": int(self.attempt_indices[env_id]),
                "terrain": self.terrain.type_names[int(self.terrain_types[env_id])],
                "terrain_id": int(self.terrain_types[env_id]), "layout": "patch",
                "task": getattr(self.evaluation, "task", ""),
                "checkpoint": str(getattr(self.evaluation, "checkpoint", "")),
                "eval_seed": self._seed, "spawn_seed": self._seed + env_id,
                "spawn_position": self.spawn_positions[env_id].tolist(),
                "goal_position": self.goal_positions[env_id].tolist(),
                "patch_bounds": self.patch_bounds[env_id].tolist(),
                "corridor_bounds": self.traversal_bounds[env_id].tolist(),
                "command_vx": float(self.commands[env_id, 0]),
                "travel_direction": int(self.travel_signs[env_id]),
                "selected_side": ("left", "right", "all")[int(self.selected_sides[env_id])],
                "expected_leading_axle": "front" if self.travel_signs[env_id] > 0 else "rear",
                "leading_axle": None,
                "window_started": False, "entry_time_s": None, "exit_time_s": None,
                "measurement_boundary_resolution_s": self.dt,
                "max_tilt_deg": float(getattr(self.evaluation, "max_tilt_deg", 60.)),
                "contact_force_threshold": float(getattr(self.evaluation, "contact_force_threshold", 1.)),
                **self._raw_episode_metadata(env_id),
                **self._raw_state_metadata(env_id, "attempt_start"),
            }
        if self.joint_angle_trace is not None:
            self.joint_angle_trace.start(env_ids)

    def _update_window_metadata(self, wheel_bounds):
        ids = self.attempt_active.nonzero(as_tuple=False).flatten()
        bounds_cpu = wheel_bounds[ids].detach().cpu().tolist()
        for env_id, current_bounds in zip(ids.tolist(), bounds_cpu):
            started = bool(self.crossing.window_started[env_id])
            item = self._attempt_metadata[env_id]
            item.update(window_started=started,
                        entry_time_s=float(self.crossing.entry_time_s[env_id]) if started else None,
                        trial_elapsed_s=float(self.episode_length_buf[env_id]) * self.dt,
                        wheel_bounds_xy=current_bounds,
                        wheels_entered_mask=self.crossing.wheels_entered_mask[env_id].tolist(),
                        wheel_contact_mask=self.crossing.wheel_contact_mask[env_id].tolist())
            if started:
                item["leading_axle"] = "front" if self.crossing.leading_axle[env_id] == 0 else "rear"
            exit_time = float(self.crossing.exit_time_s[env_id])
            item["exit_time_s"] = exit_time if math.isfinite(exit_time) and exit_time >= 0 else None
            item["exit_edge"] = (None, "xmin", "xmax", "ymin", "ymax")[int(self.crossing.exit_edge_code[env_id])]
            if bool(self.metrics.active[env_id]):
                self.metrics.metadata[env_id].update(item)

    def _begin_measurements(self, env_ids):
        if not len(env_ids):
            return
        positions = self.root_states[env_ids, :3].detach().cpu().tolist()
        velocities = self.root_states[env_ids, 7:10].detach().cpu().tolist()
        for env_id, position, velocity in zip(env_ids.tolist(), positions, velocities):
            item = self._attempt_metadata[env_id]
            item.update(entry_base_position=position, entry_velocity_world=velocity,
                        entry_wheel_bounds_xy=item["wheel_bounds_xy"])
            item.update(self._raw_state_metadata(env_id, "start"))
            item.update(self._raw_state_metadata(env_id, "entry"))
        gravity = self.sim_params.gravity
        self.metrics.start(
            env_ids, position_world=self.root_states[env_ids, :3],
            velocity_world=self.root_states[env_ids, 7:10], goal_direction_xy=self.goal_directions[env_ids],
            mass_kg=self.total_mass_kg[env_ids],
            gravity_m_s2=math.sqrt(gravity.x ** 2 + gravity.y ** 2 + gravity.z ** 2),
            metadata=[self._attempt_metadata[env_id] for env_id in env_ids.tolist()])

    def check_termination(self):
        if self.crossing is None or not self._evaluation_started:
            return
        bounds, contact, landing = self._current_wheel_geometry()
        invalid = ~torch.isfinite(self.root_states).all(dim=1) | ~torch.isfinite(bounds).all(dim=2).all(dim=1)
        if bool((invalid & self.attempt_active).any()):
            raise RuntimeError("Nonfinite evaluation state/geometry; stop collection instead of counting a fall")
        event = self.crossing.update(bounds, self.episode_length_buf.to(torch.float64) * self.dt, contact,
                                     landing=landing)
        if bool((event["failed"] & self.attempt_active).any()):
            raise RuntimeError("Invalid evaluation geometry/clock; stop collection instead of counting a fall")
        self._update_window_metadata(bounds)
        # Only already-active windows record this control interval's raw state.
        self.metrics.control_step(self.root_states[:, :3], self.root_states[:, 7:10],
                                  self.base_lin_vel, self.base_ang_vel, self._step_commands, self.dt,
                                  orientation_world=self.root_states[:, 3:7],
                                  angular_velocity_world=self.root_states[:, 10:13])
        entering = event["entered"] & self.attempt_active
        self._begin_measurements(entering.nonzero(as_tuple=False).flatten())
        tilted = self.projected_gravity[:, 2] > -math.cos(math.radians(
            float(getattr(self.evaluation, "max_tilt_deg", 60.))))
        # Protocol 7 defines a fall by base tilt only. Contacts, retreat and
        # corridor departure do not terminate the attempt; it may recover.
        failure = tilted
        success = event["exited"] & ~failure
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~failure & ~success
        self.reset_buf = (failure | success | timeout) & self.attempt_active
        self.time_out_buf = timeout & self.attempt_active
        terminal_ids = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        for env_id in terminal_ids:
            if failure[env_id]:
                self._episode_status[env_id] = "failure"
                self._episode_reason[env_id] = "fall"
            elif success[env_id]:
                self._episode_status[env_id] = "success"
                self._episode_reason[env_id] = ("stable_landing" if landing is not None and self.requires_landing[env_id]
                                                else "all_wheels_cleared")
            else:
                self._episode_status[env_id], self._episode_reason[env_id] = "timeout", "time_limit"
        if self.joint_angle_trace is not None:
            active_ids = self.attempt_active.nonzero(as_tuple=False).flatten()
            self.joint_angle_trace.sample(active_ids, {i: self._episode_reason[i] for i in terminal_ids})

    def _finish_attempt(self, env_ids, status, reason):
        rows = []
        positions = self.root_states[env_ids, :3].detach().cpu().tolist()
        velocities = self.root_states[env_ids, 7:10].detach().cpu().tolist()
        for env_id, position, velocity in zip(env_ids.tolist(), positions, velocities):
            ids = env_ids.new_tensor([env_id])
            item = self._attempt_metadata[env_id]
            if self.obstacle_traversal is not None and self.requires_landing[env_id]:
                item["reached_middle"] = bool(self.obstacle_traversal.reached_middle[env_id])
                item["stable_landing_steps"] = int(self.obstacle_traversal.landing_steps[env_id])
            trial_elapsed = float(self.episode_length_buf[env_id]) * self.dt
            item.update(trial_elapsed_s=trial_elapsed, terminal_base_position=position,
                        terminal_velocity_world=velocity,
                        terminal_wheel_bounds_xy=item.get("wheel_bounds_xy"),
                        window_end_time_s=trial_elapsed if item["window_started"] else None,
                        window_zero_duration=bool(item["window_started"] and self.metrics.control_steps[env_id] == 0))
            item.update(self._raw_state_metadata(env_id, "end"))
            item.update(self._raw_state_metadata(env_id, "terminal"))
            if bool(self.metrics.active[env_id]):
                self.metrics.metadata[env_id].update(self._attempt_metadata[env_id])
                rows.extend(self.metrics.finish(ids, status, reason=reason))
            else:
                item.update(window_started=False, entry_time_s=None, exit_time_s=None,
                            window_end_time_s=None, window_zero_duration=False)
                rows.extend(self.metrics.finish_unstarted(
                    ids, status, reason=reason, metadata=[self._attempt_metadata[env_id]]))
            self.attempt_active[env_id] = False
        self.completed_episodes.extend(rows)
        if rows and self.episode_sink is not None:
            self.episode_sink(rows)
        if status != "interrupted":
            self.episode_counts[env_ids] += 1

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        recording = self.crossing is not None and self.metrics is not None and self._evaluation_started
        if recording:
            for env_id in env_ids.tolist():
                if bool(self.attempt_active[env_id]):
                    self._finish_attempt(env_ids.new_tensor([env_id]),
                                         self._episode_status[env_id] or "interrupted",
                                         self._episode_reason[env_id] or "explicit_reset")
        NezhaMoECTSEnv.reset_idx(self, env_ids)
        self._sync_reset_state(env_ids)
        if recording:
            self._start(env_ids[self.episode_counts[env_ids] < self._episode_quota])

    def interrupt(self, reason="interrupted"):
        if self.crossing is not None:
            ids = self.attempt_active.nonzero(as_tuple=False).flatten()
            if len(ids):
                self._finish_attempt(ids, "interrupted", reason)
        return self.completed_episodes
