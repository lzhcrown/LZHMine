"""Episode boundaries for Isaac Gym student playback on the checkpoint's WF map."""

import numpy as np
from isaacgym import gymtorch
import torch

from legged_gym.envs.nezha.nezha_moe_cts_env import NezhaMoECTSEnv
from legged_gym.evaluation.geometry import wheel_geometry_from_cfg
from legged_gym.evaluation.timing import MAX_ATTEMPT_SECONDS, attempt_step_limit
from legged_gym.evaluation.traversal import ObstacleTraversal, wheel_centers_and_extents
from legged_gym.scripts._platform_play import wf_landing_distance, wf_obstacle_bounds


class StudentPlayEnv(NezhaMoECTSEnv):
    """Keep playing through the top/valley until a supported landing on the far side."""

    def __init__(self, cfg, **kwargs):
        self.traversal = None
        self.joint_angles = None
        vx, vy = cfg.commands.ranges.lin_vel_x[0], cfg.commands.ranges.lin_vel_y[0]
        self.travel_axis = int(abs(vy) > abs(vx))
        self.travel_sign = -1 if (vx, vy)[self.travel_axis] < 0 else 1
        super().__init__(cfg=cfg, **kwargs)
        self.play_step_limit = attempt_step_limit(MAX_ATTEMPT_SECONDS, self.dt)
        self.play_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if hasattr(getattr(self, "terrain", None), "tile_types"):
            self._prepare_play_courses()

    def terrain_names(self):
        tiles = getattr(getattr(self, "terrain", None), "tile_types", None)
        if tiles is not None:
            return tiles[self.terrain_levels.tolist(), self.terrain_types.tolist()].tolist()
        return [self.cfg.terrain.mesh_type] * self.num_envs

    def reset(self):
        # Start the control clock with the first policy action, including on
        # older maps whose training reset takes an extra zero-action step.
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def _prepare_play_courses(self):
        cfg = self.cfg.terrain
        courses, landing_distances = {}, {}
        rows, cols = self.terrain_levels.tolist(), self.terrain_types.tolist()
        for tile in set(zip(rows, cols)):
            courses[tile] = wf_obstacle_bounds(self.terrain, cfg, *tile)
            if courses[tile] is not None:
                landing_distances[tile] = wf_landing_distance(
                    self.terrain, courses[tile][0], self.travel_axis, self.travel_sign)
        if not any(value is not None for value in courses.values()):
            return
        geometry = wheel_geometry_from_cfg(self.cfg)
        self.play_wheel_ids = [self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], name) for name in geometry["wheel_names"]]
        wheel_envelope = ((geometry["wheel_x_min"], geometry["wheel_x_max"]),
                          (geometry["wheel_y_min"], geometry["wheel_y_max"]))[self.travel_axis]
        self.landing_length = self.root_states.new_tensor(
            [landing_distances.get(tile, 0.) for tile in zip(rows, cols)])[:, None]
        outer = np.zeros((self.num_envs, 4))
        middle = np.zeros_like(outer)
        heights = np.zeros(self.num_envs)
        spawns = np.zeros((self.num_envs, 2))
        for i, tile in enumerate(zip(rows, cols)):
            if courses[tile] is None:
                continue
            outer[i], middle[i], heights[i] = courses[tile]
            tile_lo = np.asarray(tile) * [cfg.terrain_length, cfg.terrain_width]
            tile_hi = tile_lo + [cfg.terrain_length, cfg.terrain_width]
            edges = outer[i].reshape(2, 2)
            axis = self.travel_axis
            spawns[i] = edges.mean(axis=1)
            if self.travel_sign > 0:
                spawns[i, axis] = min((tile_lo[axis] + edges[axis, 0]) / 2,
                                      edges[axis, 0] - wheel_envelope[1] - .05)
            else:
                spawns[i, axis] = max((tile_hi[axis] + edges[axis, 1]) / 2,
                                      edges[axis, 1] - wheel_envelope[0] + .05)
        self.course_outer = self.root_states.new_tensor(outer)
        self.course_middle = self.root_states.new_tensor(middle)
        self.course_height = self.root_states.new_tensor(heights)
        self.course_mask = self.course_height != 0
        self.course_spawn = self.root_states.new_tensor(spawns)
        self.wheel_geometry = tuple(self.root_states.new_tensor(geometry[name])
                                    for name in ("collision_offsets", "collision_axes", "radii", "widths"))
        self.traversal = ObstacleTraversal(self.num_envs, self.dt, self.device)

    def _reset_root_states(self, env_ids):
        if self.traversal is None:
            return super()._reset_root_states(env_ids)
        routed = self.course_mask[env_ids]
        other = env_ids[~routed]
        if len(other):
            super()._reset_root_states(other)
        ids = env_ids[routed]
        if len(ids):
            # WF training starts pits and stairs at their center. Playback starts
            # on the approach so ascent/descent are both part of every attempt.
            self.root_states[ids] = self.base_init_state
            self.root_states[ids, :2] = self.course_spawn[ids]
            self.root_states[ids, 3:7] = self.root_states.new_tensor([0., 0., 0., 1.])
            self.root_states[ids, 7:13] = 0.
            ids_int32 = ids.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(ids_int32), len(ids))
        self.traversal.reset(env_ids)

    def _landed(self):
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.play_wheel_ids]
        centers, extents = wheel_centers_and_extents(states, *self.wheel_geometry)
        xy = centers[:, :, :2]
        lower, upper = xy - extents[:, :, :2], xy + extents[:, :, :2]
        bottom = centers[:, :, 2] - extents[:, :, 2]
        tol = self.traversal.HEIGHT_TOLERANCE_M
        middle = self.course_middle.reshape(-1, 1, 2, 2)
        on_middle = ((lower >= middle[..., 0]) & (upper <= middle[..., 1])).all(dim=2)
        on_middle &= (bottom - self.course_height[:, None]).abs() <= tol
        lane = self.course_outer.reshape(-1, 1, 2, 2)
        axis, side = self.travel_axis, 1 - self.travel_axis
        within = (lower[:, :, side] >= lane[:, :, side, 0]) & (upper[:, :, side] <= lane[:, :, side, 1])
        edge = lane[:, :, axis, int(self.travel_sign > 0)]
        past = self.travel_sign * (xy[:, :, axis] - edge)
        on_far_ground = (past >= extents[:, :, axis]) & (past + extents[:, :, axis] <= self.landing_length)
        indices = ((xy + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).round().long()
        x = indices[..., 0].clamp(0, self.terrain.tot_rows - 1)
        y = indices[..., 1].clamp(0, self.terrain.tot_cols - 1)
        on_far_ground &= (self.height_samples[x, y] == 0) & (bottom.abs() <= tol)
        return self.traversal.update(
            on_middle, on_far_ground, within.all(dim=1) & self.course_mask,
            self.contact_forces[:, self.play_wheel_ids, 2], self.projected_gravity[:, 2],
            self.root_states[:, 9], self.base_ang_vel[:, :2])

    def check_termination(self):
        super().check_termination()
        failed = self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt
        if self.traversal is not None:
            self.play_success.copy_(self._landed() & ~failed)
        self.time_out_buf.copy_((self.episode_length_buf >= self.play_step_limit) & ~failed & ~self.play_success)
        self.reset_buf.copy_(failed | self.play_success | self.time_out_buf)
        self.terminal_reasons = {}
        for env_id in self.reset_buf.nonzero(as_tuple=False).flatten().tolist():
            if self.play_success[env_id]:
                reason = "stable_landing"
            elif self.time_out_buf[env_id]:
                reason = "time_limit"
            else:
                reason = "fall"
            self.terminal_reasons[env_id] = reason
        if self.joint_angles is not None:
            # The indexed reset later in this same control step overwrites dof_pos.
            self.joint_angles.sample(terminal_reasons=self.terminal_reasons)
