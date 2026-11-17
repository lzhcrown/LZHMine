# Adapted from wheel_gym_CQ/wheel_legged_gym/evaluation/metrics.py
"""Raw evaluation records, independent of the simulator.

Physics rows retain applied torques and both angular-velocity endpoints. Control
rows retain states, commands and unfiltered finite-difference acceleration.
Energy, power, CoT, tracking scores, instability thresholds and success rates
are deliberately left to offline analysis. No warmup samples are discarded.
"""

import csv
import json
import math
from pathlib import Path

import torch
from uuid import uuid4
from .timing import TIMING_FIELDS, traversal_record


STATUSES = ("success", "failure", "timeout", "interrupted")
COMPLETED_STATUSES = STATUSES[:3]
ID_FIELDS = ["run_id", "trial_id", "env_id", "episode_index", "terrain", "terrain_id", "eval_seed", "task", "checkpoint"]
META_FIELDS = ID_FIELDS + ["metadata_json"]
WINDOW_FIELDS = [
    "window_started", "entry_time_s", "exit_time_s", "trial_elapsed_s",
    "leading_axle", "travel_direction", "selected_side", "wheels_entered_mask", "wheel_contact_mask",
]
EPISODE_STATE_FIELDS = [
    prefix + "_world_" + axis + "_m" for prefix in ("start", "end") for axis in ("x", "y", "z")
] + [
    prefix + "_world_v" + axis + "_mps" for prefix in ("start", "end") for axis in ("x", "y", "z")
]
EPISODE_VALUE_FIELDS = [
    "elapsed_s", "physics_elapsed_s", "control_sample_count", "physics_sample_count",
    "mass_kg", "gravity_m_s2", "goal_direction_x", "goal_direction_y",
] + EPISODE_STATE_FIELDS
EPISODE_FIELDS = META_FIELDS + WINDOW_FIELDS + ["status", "reason", "has_physics_samples"] + EPISODE_VALUE_FIELDS
TRIAL_INDEX_FIELDS = ID_FIELDS + [
    "status", "reason", "window_started", "has_physics_samples", "entry_time_s", "exit_time_s",
    "trial_elapsed_s", "elapsed_s", "control_sample_count", "physics_sample_count",
    "spawn_seed", "command_vx", "travel_direction", "exit_edge",
    "spawn_x_m", "spawn_y_m", "spawn_z_m", "goal_x_m", "goal_y_m", "goal_z_m",
]


def trial_id_for(run_id, row):
    """Stable across interleaved CSV streams; distinct across capture runs."""
    return "{}/{}/{}/env{:04d}/ep{:04d}".format(
        run_id, row.get("task", ""), row.get("terrain", ""), int(row["env_id"]), int(row["episode_index"]))


TRAJECTORY_VALUE_FIELDS = [
    "control_step", "time_start_s", "time_end_s", "elapsed_s", "control_dt_s",
    "trial_time_start_s", "trial_time_end_s",
] + [
    prefix + "world_" + axis + "_m" for prefix in ("", "previous_") for axis in ("x", "y", "z")
] + [
    prefix + "world_v" + axis + "_mps" for prefix in ("", "previous_") for axis in ("x", "y", "z")
] + [
    "body_v" + axis + "_mps" for axis in ("x", "y", "z")
] + [
    "body_w" + axis + "_radps" for axis in ("x", "y", "z")
] + [
    "command_vx_mps", "command_vy_mps", "command_wz_radps",
    "world_ax_mps2", "world_ay_mps2", "world_az_mps2",
    "acceleration_xy_mps2", "acceleration_xyz_mps2",
    "world_qx", "world_qy", "world_qz", "world_qw",
    "world_wx_radps", "world_wy_radps", "world_wz_radps",
]
TRAJECTORY_FIELDS = META_FIELDS + WINDOW_FIELDS + TRAJECTORY_VALUE_FIELDS
PHYSICS_BASE_FIELDS = ID_FIELDS + [
    "entry_time_s", "travel_direction", "selected_side", "physics_step",
    "control_step", "substep_index", "time_start_s", "time_end_s", "dt_s",
    "trial_time_start_s", "trial_time_end_s",
]
SUMMARY_FIELDS = [
    "scope", "terrain", "travel_direction", "episodes", "completed_episodes", "success", "failure",
    "timeout", "interrupted", "started_attempts", "started_tests",
    "started_success", "started_failure", "started_timeout", "started_interrupted",
    "not_started_attempts", "not_started_failure", "not_started_timeout", "not_started_interrupted",
]

# Only explicitly raw provenance/geometry may enter metadata_json. In particular,
# old metric keys or an incoming metadata_json string cannot bypass the schema.
RAW_METADATA_FIELDS = set(ID_FIELDS + WINDOW_FIELDS + [
    "layout", "seed", "spawn_seed", "spawn_position", "goal_position", "patch_bounds", "corridor_bounds",
    "command_vx", "expected_leading_axle", "measurement_boundary_resolution_s", "max_tilt_deg",
    "contact_force_threshold", "training_failure_persistence_s", "wheel_bounds_xy", "currentwheel_bounds_xy",
    "entry_wheel_bounds_xy", "terminal_wheel_bounds_xy", "window_end_time_s", "window_zero_duration",
    "dof_names", "mass_kg", "gravity_m_s2", "gravity_world_xyz", "gravity_vector_m_s2", "goal_direction_xy", "exit_edge",
] + [
    prefix + suffix for prefix in ("start", "end", "attempt_start", "entry", "terminal")
    for suffix in ("_base_position", "_velocity_world", "_velocity_body", "_angular_velocity_body",
                   "_angular_velocity_world", "_orientation_xyzw")
])


def _sanitize(value):
    """Make nested raw metadata JSON safe, representing missing/nonfinite as null."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean_metadata(item):
    return {key: _sanitize(value) for key, value in dict(item).items() if key in RAW_METADATA_FIELDS}


def physics_fields(num_joints):
    """Ordered raw columns; j00 maps to the first entry in metadata dof_names."""
    return PHYSICS_BASE_FIELDS + [
        pattern.format(j) for pattern in ("torque_j{:02d}_nm", "joint_vel_before_j{:02d}_radps", "joint_vel_after_j{:02d}_radps")
        for j in range(num_joints)
    ]


class VectorEpisodeMetrics:
    """Keep window state on device and stream only raw rows to CPU sinks.

    ``start`` takes selected-environment tensors; both step methods take all
    environment rows. The terminal control state must be recorded before
    ``finish`` and simulator reset. A sample mask gates records and clocks;
    previous state baselines still advance across excluded active intervals.
    """

    def __init__(self, num_envs, device, *, trajectory_stride=1,
                 trajectory_sink=None, physics_sink=None, num_joints=None):
        if int(num_envs) != num_envs or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if int(trajectory_stride) != trajectory_stride or trajectory_stride < 0:
            raise ValueError("trajectory_stride must be a nonnegative integer")
        if num_joints is not None and (int(num_joints) != num_joints or num_joints <= 0):
            raise ValueError("num_joints must be a positive integer")
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.num_joints = None if num_joints is None else int(num_joints)
        self.trajectory_stride, self.trajectory_sink = int(trajectory_stride), trajectory_sink
        self.physics_sink = physics_sink
        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.metadata = [{} for _ in range(self.num_envs)]
        for name in ("elapsed_s", "physics_elapsed_s", "mass_kg", "gravity_m_s2"):
            setattr(self, name, torch.zeros(self.num_envs, dtype=torch.float64, device=self.device))
        for name in ("control_steps", "physics_steps", "_substeps"):
            setattr(self, name, torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
        for name, width in (("start_position", 3), ("start_velocity", 3), ("end_position", 3),
                            ("end_velocity", 3), ("previous_position", 3), ("previous_velocity", 3),
                            ("goal_direction_xy", 2)):
            setattr(self, name, torch.zeros((self.num_envs, width), dtype=torch.float64, device=self.device))

    def _tensor(self, value, shape, name):
        result = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        if tuple(result.shape) != tuple(shape):
            raise ValueError("{} must have shape {}, got {}".format(name, shape, tuple(result.shape)))
        return result

    def _ids(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.numel() and (bool((ids < 0).any()) or bool((ids >= self.num_envs).any())):
            raise ValueError("environment ID out of range")
        if ids.unique().numel() != ids.numel():
            raise ValueError("environment IDs must be unique")
        return ids

    def _sample_mask(self, sample_mask):
        if sample_mask is None:
            return self.active
        mask = torch.as_tensor(sample_mask, dtype=torch.bool, device=self.device)
        if tuple(mask.shape) != (self.num_envs,):
            raise ValueError("sample_mask must have shape (num_envs,)")
        return self.active & mask

    @staticmethod
    def _dt(dt):
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        return dt

    @torch.no_grad()
    def start(self, env_ids, *, position_world, velocity_world, goal_direction_xy,
              mass_kg, gravity_m_s2=9.81, metadata=None):
        ids = self._ids(env_ids)
        n = ids.numel()
        if bool(self.active[ids].any()):
            raise ValueError("finish an active episode before starting another")
        position = self._tensor(position_world, (n, 3), "position_world")
        velocity = self._tensor(velocity_world, (n, 3), "velocity_world")
        direction = self._tensor(goal_direction_xy, (n, 2), "goal_direction_xy")
        mass = torch.as_tensor(mass_kg, dtype=torch.float64, device=self.device).reshape(-1)
        gravity = torch.as_tensor(gravity_m_s2, dtype=torch.float64, device=self.device)
        gravity = gravity.expand(n) if gravity.ndim == 0 else gravity.reshape(-1)
        if mass.numel() != n or gravity.numel() != n:
            raise ValueError("mass/gravity must contain one value per selected environment (gravity may be scalar)")
        if any(not bool(torch.isfinite(value).all()) for value in (position, velocity, direction, mass, gravity)):
            raise ValueError("episode initial states must be finite")
        if bool((torch.linalg.vector_norm(direction, dim=1) <= 1e-12).any()) or bool((mass <= 0).any()) or bool((gravity <= 0).any()):
            raise ValueError("goal directions must be nonzero and mass/gravity must be positive")
        metadata = [{} for _ in range(n)] if metadata is None else metadata
        if len(metadata) != n:
            raise ValueError("metadata must contain one dict per selected environment")
        items = [_clean_metadata(item) for item in metadata]
        for name in ("elapsed_s", "physics_elapsed_s", "control_steps", "physics_steps", "_substeps"):
            getattr(self, name)[ids] = 0
        for name in ("start_position", "end_position", "previous_position"):
            getattr(self, name)[ids] = position
        for name in ("start_velocity", "end_velocity", "previous_velocity"):
            getattr(self, name)[ids] = velocity
        self.goal_direction_xy[ids] = direction
        self.mass_kg[ids], self.gravity_m_s2[ids] = mass, gravity
        self.episode_index[ids] += 1
        for env_id, index, item in zip(ids.cpu().tolist(), self.episode_index[ids].cpu().tolist(), items):
            item["env_id"] = env_id
            item.setdefault("episode_index", index)
            item["window_started"] = True
            self.metadata[env_id] = item
        self.active[ids] = True

    def update_metadata(self, env_ids, metadata):
        ids = self._ids(env_ids)
        if len(metadata) != ids.numel():
            raise ValueError("metadata must contain one dict per selected environment")
        for env_id, item in zip(ids.cpu().tolist(), metadata):
            self.metadata[env_id].update(_clean_metadata(item))
            self.metadata[env_id]["env_id"] = env_id

    def _rows(self, ids, columns, *, physics=False):
        if not ids.numel():
            return []
        values = torch.stack(list(columns.values()), dim=1)[ids].cpu().tolist()
        rows = []
        for env_id, numbers in zip(ids.cpu().tolist(), values):
            source = self.metadata[env_id]
            row = {key: source[key] for key in PHYSICS_BASE_FIELDS if key in source} if physics else dict(source)
            row.update((key, _sanitize(value)) for key, value in zip(columns, numbers))
            for key in ("control_step", "physics_step", "substep_index", "control_sample_count", "physics_sample_count"):
                if key in row:
                    row[key] = int(row[key])
            # Physics hooks run before control-boundary metadata is refreshed.
            # Derive trial clocks from the actual window clock, not stale metadata.
            if "time_start_s" in row:
                entry = source.get("entry_time_s")
                entry = 0.0 if entry is None else float(entry)
                row["trial_time_start_s"] = _sanitize(entry + row["time_start_s"])
                row["trial_time_end_s"] = _sanitize(entry + row["time_end_s"])
            rows.append(row)
        return rows

    @torch.no_grad()
    def physics_step(self, torque, joint_vel_before, joint_vel_after, dt, sample_mask=None):
        """Record actual applied torque and raw angular velocities at both ends."""
        dt = self._dt(dt)
        torque = torch.as_tensor(torque, dtype=torch.float64, device=self.device)
        if torque.ndim != 2 or torque.shape[0] != self.num_envs or torque.shape[1] == 0:
            raise ValueError("torque must have shape (num_envs, num_joints)")
        if self.num_joints is None:
            self.num_joints = torque.shape[1]
        elif torque.shape[1] != self.num_joints:
            raise ValueError("torque joint count differs from num_joints")
        before = self._tensor(joint_vel_before, torque.shape, "joint_vel_before")
        after = self._tensor(joint_vel_after, torque.shape, "joint_vel_after")
        sampled = self._sample_mask(sample_mask)
        time_start = self.physics_elapsed_s.clone()
        # Multiplying a bool mask by a Python float would produce float32 and
        # round dt before adding it to this float64 clock.
        self.physics_elapsed_s[sampled] += dt
        self.physics_steps += sampled.long()
        self._substeps += sampled.long()
        if self.physics_sink is not None:
            columns = dict(physics_step=self.physics_steps, control_step=self.control_steps + 1,
                           substep_index=self._substeps, time_start_s=time_start,
                           time_end_s=self.physics_elapsed_s, dt_s=torch.full_like(time_start, dt))
            for pattern, value in (("torque_j{:02d}_nm", torque), ("joint_vel_before_j{:02d}_radps", before),
                                   ("joint_vel_after_j{:02d}_radps", after)):
                columns.update((pattern.format(j), value[:, j]) for j in range(self.num_joints))
            rows = self._rows(sampled.nonzero(as_tuple=False).flatten(), columns, physics=True)
            if rows:
                self.physics_sink(rows)

    @torch.no_grad()
    def control_step(self, position_world, velocity_world, velocity_body,
                     angular_velocity_body, command_body, dt, sample_mask=None,
                     orientation_world=None, angular_velocity_world=None):
        """Record a control boundary, including unfiltered world acceleration.

        Commands are body-frame [vx, vy, yaw_rate]; optional orientation is xyzw.
        No threshold, warmup or tracking reduction is applied to these records.
        """
        dt = self._dt(dt)
        shape = (self.num_envs, 3)
        position = self._tensor(position_world, shape, "position_world")
        velocity = self._tensor(velocity_world, shape, "velocity_world")
        body = self._tensor(velocity_body, shape, "velocity_body")
        angular = self._tensor(angular_velocity_body, shape, "angular_velocity_body")
        command = self._tensor(command_body, shape, "command_body")
        quaternion = torch.full((self.num_envs, 4), float("nan"), dtype=torch.float64, device=self.device) if orientation_world is None else self._tensor(orientation_world, (self.num_envs, 4), "orientation_world")
        world_angular = torch.full_like(angular, float("nan")) if angular_velocity_world is None else self._tensor(angular_velocity_world, shape, "angular_velocity_world")
        sampled = self._sample_mask(sample_mask)
        acceleration = (velocity - self.previous_velocity) / dt
        time_start = self.elapsed_s.clone()
        self.elapsed_s[sampled] += dt
        self.control_steps += sampled.long()
        self.end_position[sampled] = position[sampled]
        self.end_velocity[sampled] = velocity[sampled]
        if self.trajectory_sink is not None and self.trajectory_stride:
            columns = dict(control_step=self.control_steps, time_start_s=time_start, time_end_s=self.elapsed_s,
                           elapsed_s=self.elapsed_s, control_dt_s=torch.full_like(time_start, dt))
            for prefix, value in (("", position), ("previous_", self.previous_position)):
                columns.update((prefix + "world_" + axis + "_m", value[:, j]) for j, axis in enumerate("xyz"))
            for prefix, value in (("", velocity), ("previous_", self.previous_velocity)):
                columns.update((prefix + "world_v" + axis + "_mps", value[:, j]) for j, axis in enumerate("xyz"))
            columns.update(("body_v" + axis + "_mps", body[:, j]) for j, axis in enumerate("xyz"))
            columns.update(("body_w" + axis + "_radps", angular[:, j]) for j, axis in enumerate("xyz"))
            columns.update((key, command[:, j]) for j, key in enumerate(("command_vx_mps", "command_vy_mps", "command_wz_radps")))
            columns.update(("world_a" + axis + "_mps2", acceleration[:, j]) for j, axis in enumerate("xyz"))
            columns["acceleration_xy_mps2"] = torch.linalg.vector_norm(acceleration[:, :2], dim=1)
            columns["acceleration_xyz_mps2"] = torch.linalg.vector_norm(acceleration, dim=1)
            columns.update(("world_q" + axis, quaternion[:, j]) for j, axis in enumerate("xyzw"))
            columns.update(("world_w" + axis + "_radps", world_angular[:, j]) for j, axis in enumerate("xyz"))
            ids = (sampled & (self.control_steps % self.trajectory_stride == 0)).nonzero(as_tuple=False).flatten()
            rows = self._rows(ids, columns)
            if rows:
                self.trajectory_sink(rows)
        self.previous_position[self.active] = position[self.active]
        self.previous_velocity[self.active] = velocity[self.active]
        self._substeps[self.active] = 0

    def _statuses(self, n, status, reason, allowed=STATUSES):
        statuses = [status] * n if isinstance(status, str) else list(status)
        reasons = [reason] * n if isinstance(reason, str) else list(reason)
        if len(statuses) != n or len(reasons) != n or any(value not in allowed for value in statuses):
            raise ValueError("supply one supported status and reason per environment")
        return statuses, reasons

    @torch.no_grad()
    def finish(self, env_ids, status, reason=""):
        ids = self._ids(env_ids)
        if not bool(self.active[ids].all()):
            raise ValueError("cannot finish an inactive episode")
        statuses, reasons = self._statuses(ids.numel(), status, reason)
        columns = dict(elapsed_s=self.elapsed_s, physics_elapsed_s=self.physics_elapsed_s,
                       control_sample_count=self.control_steps, physics_sample_count=self.physics_steps,
                       mass_kg=self.mass_kg, gravity_m_s2=self.gravity_m_s2,
                       goal_direction_x=self.goal_direction_xy[:, 0], goal_direction_y=self.goal_direction_xy[:, 1])
        for prefix, value in (("start", self.start_position), ("end", self.end_position)):
            columns.update((prefix + "_world_" + axis + "_m", value[:, j]) for j, axis in enumerate("xyz"))
        for prefix, value in (("start", self.start_velocity), ("end", self.end_velocity)):
            columns.update((prefix + "_world_v" + axis + "_mps", value[:, j]) for j, axis in enumerate("xyz"))
        rows = self._rows(ids, columns)
        for row, result, why in zip(rows, statuses, reasons):
            row.update(status=result, reason=why, window_started=True, has_physics_samples=row["physics_sample_count"] > 0)
        self.active[ids] = False
        return rows

    @torch.no_grad()
    def finish_unstarted(self, env_ids, status, reason="", metadata=None):
        """Keep pre-entry outcomes without inventing a measured window."""
        ids = self._ids(env_ids)
        if bool(self.active[ids].any()):
            raise ValueError("use finish() for a started measurement")
        statuses, reasons = self._statuses(ids.numel(), status, reason, STATUSES[1:])
        metadata = [{} for _ in range(ids.numel())] if metadata is None else metadata
        if len(metadata) != ids.numel():
            raise ValueError("metadata must contain one dict per selected environment")
        items = [_clean_metadata(item) for item in metadata]
        self.episode_index[ids] += 1
        rows = []
        for env_id, index, item, result, why in zip(ids.cpu().tolist(), self.episode_index[ids].cpu().tolist(), items, statuses, reasons):
            row = dict(item)
            row.update((key, None) for key in EPISODE_VALUE_FIELDS)
            for key in ("mass_kg", "gravity_m_s2"):
                row[key] = item.get(key)
            direction = item.get("goal_direction_xy")
            if direction is not None:
                row["goal_direction_x"], row["goal_direction_y"] = direction
            row.update(env_id=env_id, window_started=False, has_physics_samples=False, entry_time_s=None,
                       exit_time_s=None, status=result, reason=why, control_sample_count=0, physics_sample_count=0)
            row.setdefault("episode_index", index)
            self.metadata[env_id] = _clean_metadata(row)
            rows.append(row)
        return rows


def summarize_episodes(episodes):
    """Count outcomes by terrain and direction, with no scores or averages.

    Interrupted attempts are not completed tests. Attempts that never entered
    remain visible in separate counts. Legacy region rows default to started.
    """
    groups, directions = {}, {}

    def started(row):
        return str(row.get("window_started", True)).lower() in ("true", "1", "1.0")

    def direction(row):
        value = row.get("travel_direction")
        if value in (1, "1", "1.0", "+1", "forward"):
            return "forward"
        if value in (-1, "-1", "-1.0", "backward", "reverse"):
            return "backward"
        return "" if value is None else str(value)

    for row in episodes:
        key = (str(row.get("terrain", "unknown")), direction(row))
        groups.setdefault(key, []).append(row)
        directions.setdefault(key[1], []).append(row)
    if not directions:
        directions[""] = []
    batches = [("terrain", terrain, travel, groups[(terrain, travel)]) for terrain, travel in sorted(groups)]
    batches += [("overall", "all", travel, directions[travel]) for travel in sorted(directions)]
    result = []
    for scope, terrain, travel, rows in batches:
        counts = {status: sum(row["status"] == status for row in rows) for status in STATUSES}
        completed = [row for row in rows if row["status"] in COMPLETED_STATUSES]
        started_rows = [row for row in rows if started(row)]
        summary = dict(scope=scope, terrain=terrain, travel_direction=travel, episodes=len(rows),
                       completed_episodes=len(completed), started_attempts=len(started_rows),
                       started_tests=sum(started(row) for row in completed),
                       not_started_attempts=len(rows) - len(started_rows), **counts)
        for status in STATUSES:
            summary["started_" + status] = sum(row["status"] == status for row in started_rows)
        for status in STATUSES[1:]:
            summary["not_started_" + status] = counts[status] - summary["started_" + status]
        result.append(summary)
    return result


class CsvEvaluationWriter:
    """Stream and flush raw CSV files; refuse to overwrite existing captures."""

    def __init__(self, output_dir, num_joints=16, save_trajectories=True, run_id=None):
        if int(num_joints) != num_joints or num_joints <= 0:
            raise ValueError("num_joints must be a positive integer")
        self.num_joints = int(num_joints)
        self.run_id = run_id or uuid4().hex
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles, self._writers, self._closed = {}, {}, False
        names = self.file_specs(save_trajectories)
        for name, _ in names:
            if (self.output_dir / (name + ".csv")).exists():
                raise FileExistsError("evaluation CSV already exists: " + str(self.output_dir / (name + ".csv")))
        try:
            for name, fields in names:
                path = self.output_dir / (name + ".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("x", newline="", encoding="utf-8")
                self._handles[name] = handle
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                handle.flush()
                self._writers[name] = writer
        except Exception:
            self.close()
            raise

    def file_specs(self, save_trajectories):
        names = [("episodes", EPISODE_FIELDS), ("physics", physics_fields(self.num_joints)),
                 ("trial_index", TRIAL_INDEX_FIELDS), ("traversal_time", TIMING_FIELDS)]
        if save_trajectories:
            names.append(("trajectory", TRAJECTORY_FIELDS))
        return names

    def _write(self, name, rows):
        if self._closed:
            raise ValueError("evaluation writer is closed")
        if name not in self._writers:
            return
        writer = self._writers[name]
        for original in rows:
            row = {key: _sanitize(original.get(key)) for key in writer.fieldnames if key != "metadata_json"}
            row.update(run_id=self.run_id, trial_id=trial_id_for(self.run_id, original))
            if "metadata_json" in writer.fieldnames:
                raw = _clean_metadata(original)
                extra = {key: value for key, value in raw.items() if key not in writer.fieldnames}
                row["metadata_json"] = json.dumps(extra, ensure_ascii=False, sort_keys=True, allow_nan=False) if extra else None
            row = {key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, tuple, dict)) else value
                   for key, value in row.items()}
            writer.writerow(row)
        self._handles[name].flush()

    def write_episodes(self, rows):
        rows = list(rows)
        self._write("episodes", rows)
        index_rows = []
        for original in rows:
            item = dict(original)
            for prefix in ("spawn", "goal"):
                for axis, value in zip("xyz", original.get(prefix + "_position") or [None] * 3):
                    item[prefix + "_" + axis + "_m"] = value
            index_rows.append(item)
        self._write("trial_index", index_rows)
        self._write("traversal_time", (traversal_record(row) for row in rows))

    def write_trajectories(self, rows):
        self._write("trajectory", rows)

    def write_physics(self, rows):
        rows = list(rows)
        expected = set(physics_fields(self.num_joints)) - set(PHYSICS_BASE_FIELDS)
        for row in rows:
            actual = {key for key in row if key.startswith(("torque_j", "joint_vel_before_j", "joint_vel_after_j"))}
            if actual != expected:
                raise ValueError("physics row joint columns do not match writer num_joints")
        self._write("physics", rows)

    def write_summary(self, episodes):
        if self._closed:
            raise ValueError("evaluation writer is closed")
        rows = summarize_episodes(episodes)
        with (self.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
