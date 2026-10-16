# SPDX-License-Identifier: Apache-2.0
"""Isaac Gym adapter for legbot_lab WF-CTS-MOE (daff9fb) curricula.

Adapted from tasks/go2/mdp/commands.py by wty-yy and curriculums.py.
Modified for LZHMine's tensor buffers, resets and reward-scale interface.
See licenses/dependencies/legbot_lab-Apache-2.0.txt.

The command sampler and terrain promotion rules follow ``Go2RLGymCommand``
and ``terrain_levels_vel_gym``. In particular, commanded XY velocities are
summed as vectors at resampling, not integrated as absolute travelled distance.
Nezha's optional reward-driven range progression keeps these WF sampling rules.
"""

from copy import deepcopy
from itertools import product
import math

import torch


COMMAND_AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")


def is_wf_profile(cfg):
    return (getattr(cfg.commands, "curriculum_profile", None) == "legbot_wf"
            and getattr(cfg.terrain, "generator_profile", None) == "legbot_wf")


def configure_wf_timing(env):
    """Use the configured timebase, as Isaac Lab does, for WF step counts.

    Gym exposes SimParams.dt as a C float: 0.005 becomes 0.004999999888.
    Applying ceil to 25 / (4 * that value) incorrectly adds an episode step.
    Keep old profiles' timing intact for checkpoint compatibility.
    """
    if not is_wf_profile(env.cfg):
        return
    env.dt = env.cfg.control.decimation * env.cfg.sim.dt
    env.max_episode_length = math.ceil(env.max_episode_length_s / env.dt)
    env.cfg.domain_rand.push_interval = math.ceil(env.cfg.domain_rand.push_interval_s / env.dt)
    env.cfg.domain_rand.rand_interval = math.ceil(env.cfg.domain_rand.rand_interval_s / env.dt)


def default_wf_command_config():
    """Return independent, JSON-serializable Legbot WF command settings."""
    terrain_limits = {}
    for kind in (
        "wave", "slope_up", "slope_down", "rough_slope", "random_rough",
        "hf_pyramid_slope", "hf_pyramid_slope_inv",
    ):
        terrain_limits[kind] = {
            "lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-1.5, 1.5],
        }
    for kind in (
        "stairs_up", "stairs_down", "obstacles", "stepping_stones", "gap",
        "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    ):
        terrain_limits[kind] = {
            "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-1.5, 1.5],
        }
    terrain_limits["flat"] = {
        "lin_vel_x": [-3.0, 3.0], "lin_vel_y": [-1.0, 1.0],
        "ang_vel_yaw": [-3.14, 3.14],
    }
    return {
        "num_steps_per_iter": 24,
        "range_curriculum": "iteration",
        "resampling_time": 5.0,
        "dynamic_resample_commands": True,
        "limit_vel_invert_when_continuous": True,
        "limit_vel_prob": 0.2,
        "limit_vel": {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]},
        "limit_ang_vel_at_zero_command_prob": 0.2,
        "zero_command_curriculum": {
            "start_iter": 0, "end_iter": 1500, "start_value": 0.0, "end_value": 0.15,
        },
        "command_range_curriculum": [
            {"iter": 20000, "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
            {"iter": 50000, "lin_vel_x": [-2.0, 2.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-2.0, 2.0]},
            {"iter": 75000, "lin_vel_x": [-3.0, 3.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-3.14, 3.14]},
        ],
        "terrain_max_command_ranges": terrain_limits,
        "reward_curriculum": {
            "lin_vel_z": {"initial": -2.0, "final": -1.0, "start_iter": 0, "end_iter": 1500},
            "base_height": {"initial": -1.0, "final": -10.0, "start_iter": 0, "end_iter": 5000},
        },
    }


def linear_schedule(iteration, start_iter, end_iter, initial, final):
    fraction = min(max((iteration - start_iter) / (end_iter - start_iter), 0.0), 1.0)
    return initial + fraction * (final - initial)


def sample_disjoint_intervals(lower_bound, minimum, maximum):
    """Reference sampler, including its positive endpoint fallback at saturation.

    WF adds 1e-6 to the total interval width. Consequently a saturated interval
    returns its upper endpoint plus at most 1e-6; retain that numerical behavior.
    """
    width_neg = torch.relu(-lower_bound - minimum)
    width_pos = torch.relu(maximum - lower_bound)
    u = torch.rand_like(lower_bound) * (width_neg + width_pos + 1e-6)
    return torch.where(u < width_neg, minimum + u, maximum - width_pos + (u - width_neg))


class WFCurriculum:
    """State owned by an environment, with buffers safe across inference mode."""

    def __init__(self, env):
        self.env = env
        self.cfg = deepcopy(env.cfg.commands.wf_settings)
        # Older WF checkpoints omit this key and retain the iteration schedule.
        self.range_curriculum = self.cfg.get("range_curriculum", "iteration")
        if self.range_curriculum not in ("iteration", "nezha_reward"):
            raise ValueError("Unknown command range curriculum: " + self.range_curriculum)
        self.initial_ranges = deepcopy(env.command_ranges)
        self.steps_per_iteration = self.cfg["num_steps_per_iter"]
        self.resampling_time = self.cfg["resampling_time"]
        self.terrain_length = max(
            0.0, env.cfg.terrain.terrain_length
            - 2.0 * getattr(env.cfg.terrain, "sub_terrain_border_width", 0.0),
        )
        # These tensors must never be replaced during an inference-mode rollout.
        with torch.inference_mode(False):
            self.time_left = torch.zeros(env.num_envs, device=env.device)
            self.commands_xy_accumulation = torch.zeros(env.num_envs, 2, device=env.device)
            self.max_move_distance = torch.zeros(env.num_envs, device=env.device)
            self.last_is_limit_vel = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            self.env_command_ranges = {
                key: torch.tensor(env.command_ranges[key], device=env.device, dtype=torch.float).repeat(env.num_envs, 1)
                for key in COMMAND_AXES
            }
            self.limit_vel_comb = torch.tensor(list(product(
                *(self.cfg["limit_vel"][key] for key in COMMAND_AXES)
            )), device=env.device)
        self.zero_command_prob = 0.0
        self._ranges_signature = None
        self.update_command_ranges()

    @property
    def iteration(self):
        return self.env.common_step_counter // self.steps_per_iteration

    @property
    def uses_reward_command_curriculum(self):
        return self.range_curriculum == "nezha_reward"

    def update_command_ranges(self):
        env = self.env
        if env.cfg.commands.curriculum and not self.uses_reward_command_curriculum:
            ranges = self.initial_ranges
            for stage in sorted(self.cfg["command_range_curriculum"], key=lambda item: item["iter"]):
                if self.iteration >= stage["iter"]:
                    ranges = stage
            for key in COMMAND_AXES:
                env.command_ranges[key] = list(ranges[key])
        signature = (env.cfg.commands.curriculum,) + tuple(
            tuple(env.command_ranges[key]) for key in COMMAND_AXES
        )
        if signature == self._ranges_signature:
            return
        for key in COMMAND_AXES:
            self.env_command_ranges[key][:] = torch.tensor(
                env.command_ranges[key], device=env.device,
            )
        if env.cfg.commands.curriculum:
            if env.cfg.terrain.mesh_type in ("trimesh", "heightfield"):
                column_types = env.terrain.column_types
                for column, kind in enumerate(column_types):
                    if kind not in self.cfg["terrain_max_command_ranges"]:
                        raise ValueError(f"Missing WF command limits for terrain {kind!r}")
                    ids = (env.terrain_types == column).nonzero(as_tuple=False).flatten()
                    self._set_terrain_ranges(ids, kind)
            else:
                ids = torch.arange(env.num_envs, device=env.device)
                self._set_terrain_ranges(ids, "flat")
        self.max_lin_vel = max(abs(value) for key in COMMAND_AXES[:2] for value in env.command_ranges[key])
        self._ranges_signature = signature

    def update_reward_command_ranges(self, env_ids):
        """Nezha's independent tracking-reward gates, before episode sums clear.

        Divide by the maximum episode length, as in the original Nezha task;
        short failed episodes must not qualify on a few good tracking steps.
        The environment retains the original once-per-episode-clock cadence.
        """
        env = self.env
        if not self.uses_reward_command_curriculum or not env.cfg.commands.curriculum or len(env_ids) == 0:
            return
        for reward, axis, increment, limit in (
            ("tracking_lin_vel", "lin_vel_x", 0.2, env.cfg.commands.max_curriculum),
            ("tracking_ang_vel", "ang_vel_yaw", 0.1, env.cfg.commands.max_curriculum_yaw),
        ):
            if (torch.mean(env.episode_sums[reward][env_ids]) / env.max_episode_length
                    > 0.8 * env.reward_scales[reward]):
                bounds = env.command_ranges[axis]
                bounds[0] = max(-limit, min(bounds[0] - increment, 0.0))
                bounds[1] = min(limit, max(bounds[1] + increment, 0.0))
        # Refresh per-terrain intersections without changing the other WF
        # sampling, standing, terrain-level or reward-curriculum rules.
        self.update_command_ranges()

    def _set_terrain_ranges(self, ids, kind):
        limits = self.cfg["terrain_max_command_ranges"][kind]
        for key in COMMAND_AXES:
            self.env_command_ranges[key][ids, 0] = max(limits[key][0], self.env.command_ranges[key][0])
            self.env_command_ranges[key][ids, 1] = min(limits[key][1], self.env.command_ranges[key][1])

    def update_reward_curriculum(self):
        if not self.env.cfg.commands.curriculum:
            return
        for name, schedule in self.cfg["reward_curriculum"].items():
            self.env.reward_scales[name] = linear_schedule(
                self.iteration, schedule["start_iter"], schedule["end_iter"],
                schedule["initial"], schedule["final"],
            ) * self.env.dt

    def reset(self, env_ids):
        """Call after terrain/root reset, but before clearing the episode clock.

        Isaac Lab's manager reset order exposes the completed episode length to
        the first new sample. At timeout this saturates dynamic XY sampling.
        """
        self.time_left[env_ids] = self.resampling_time
        self.commands_xy_accumulation[env_ids] = 0.0
        self.max_move_distance[env_ids] = 0.0
        self.last_is_limit_vel[env_ids] = False
        self.resample(env_ids)

    def step(self):
        """Isaac Lab command.compute order: clock, resample, displacement metric."""
        self.time_left.sub_(self.env.dt)
        # Float subtraction can leave ~1e-6 at the nominal 250-step boundary.
        # WF uses time_left <= 0; use that same rule rather than a step modulus.
        ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        self.resample(ids)
        distance = torch.norm(self.env.root_states[:, :2] - self.env.env_origins[:, :2], dim=1)
        torch.maximum(self.max_move_distance, distance, out=self.max_move_distance)

    def terrain_move_masks(self, env_ids):
        distance = self.max_move_distance[env_ids]
        move_up = distance > self.terrain_length / 2.0
        target = torch.norm(self.commands_xy_accumulation[env_ids], dim=1)
        target *= self.resampling_time * (1.0 - self.zero_command_prob)
        move_down = (distance < target * 0.5) & ~move_up
        return move_up, move_down

    def resample(self, env_ids):
        if len(env_ids) == 0:
            return
        env = self.env
        self.update_command_ranges()
        previous = env.commands[env_ids, :3].clone()
        remaining_dist = torch.clamp(
            0.625 * self.terrain_length
            - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.resampling_time,
            min=0.0,
        )
        self.time_left[env_ids] = self.resampling_time
        dynamic = env.cfg.commands.curriculum and self.cfg["dynamic_resample_commands"]
        if dynamic:
            remaining_steps = env.max_episode_length - env.episode_length_buf[env_ids] + 1
            if (remaining_steps == 0).any():
                raise ValueError("Zero remaining episode length during WF command resampling")
            low_bound = torch.clamp(remaining_dist / ((remaining_steps + 1e-9) * env.dt), min=0.0)
        for axis, key in enumerate(COMMAND_AXES):
            limits = self.env_command_ranges[key][env_ids]
            if dynamic and axis < 2:
                values = sample_disjoint_intervals(low_bound, limits[:, 0], limits[:, 1])
            else:
                values = limits[:, 0] + torch.rand(len(env_ids), device=env.device) * (limits[:, 1] - limits[:, 0])
            env.commands[env_ids, axis] = values
        if not env.cfg.commands.curriculum:
            # Evaluation writes fixed ranges; preserve them, including small
            # commands, with no standing or limiting-command substitutions.
            self.zero_command_prob = 0.0
            self.commands_xy_accumulation[env_ids] += env.commands[env_ids, :2]
            return
        if not dynamic:
            env.commands[env_ids, :2] *= (torch.norm(env.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=env.device)
        limit_prob = self.cfg["limit_vel_prob"]
        limited = rand_prob < limit_prob
        limited_ids = env_ids[limited]
        if limit_prob > 0.0:
            changed_ids = limited_ids
            if self.cfg["limit_vel_invert_when_continuous"]:
                was_limited = self.last_is_limit_vel[limited_ids]
                env.commands[limited_ids[was_limited], :3] = -previous[limited][was_limited]
                changed_ids = limited_ids[~was_limited]
            choice = torch.randint(0, len(self.limit_vel_comb), (len(changed_ids),), device=env.device)
            for axis, key in enumerate(COMMAND_AXES):
                signs = self.limit_vel_comb[choice, axis]
                values = torch.where(signs == -1,
                                     self.env_command_ranges[key][changed_ids, 0],
                                     self.env_command_ranges[key][changed_ids, 1])
                values[signs == 0] = 0.0
                env.commands[changed_ids, axis] = values
            self.last_is_limit_vel[env_ids] = limited

        schedule = self.cfg["zero_command_curriculum"]
        self.zero_command_prob = 0.0 if schedule is None else linear_schedule(
            self.iteration, schedule["start_iter"], schedule["end_iter"],
            schedule["start_value"], schedule["end_value"],
        )
        if self.zero_command_prob > 0.0:
            next_time = torch.clamp(
                env.max_episode_length_s - env.episode_length_buf[env_ids] * env.dt
                - remaining_dist / (0.8 * self.max_lin_vel + 1e-9),
                min=0.0, max=self.resampling_time,
            )
            zero_mask = (rand_prob >= limit_prob) & (rand_prob < limit_prob + self.zero_command_prob) & (next_time > 0.0)
            zero_ids = env_ids[zero_mask]
            env.commands[zero_ids, :3] = 0.0
            self.time_left[zero_ids] = next_time[zero_mask]
            yaw_prob = self.cfg["limit_ang_vel_at_zero_command_prob"]
            if yaw_prob > 0.0:
                yaw_ids = zero_ids[torch.rand(len(zero_ids), device=env.device) < yaw_prob]
                positive = torch.rand(len(yaw_ids), device=env.device) >= 0.5
                env.commands[yaw_ids, 2] = torch.where(
                    positive, self.env_command_ranges["ang_vel_yaw"][yaw_ids, 1],
                    self.env_command_ranges["ang_vel_yaw"][yaw_ids, 0],
                )
        self.commands_xy_accumulation[env_ids] += env.commands[env_ids, :2]
