"""Stream per-control-step joint states, deployed actions, controller targets and telemetry."""

import csv
from contextlib import contextmanager
from pathlib import Path


LEGS = ("FL", "FR", "RL", "RR")
LEG_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
# The continuous foot DOFs are wheel axle rotations. Keep their raw radians.
WHEEL_JOINT_NAMES = tuple(f"{leg}_foot_joint" for leg in LEGS)
JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
VELOCITY_FIELDS = ("command_vx_body_m_s", "command_vy_body_m_s", "command_yaw_rate_rad_s",
                   "base_vx_body_m_s", "base_vy_body_m_s", "base_yaw_rate_body_rad_s")
FORCE_FIELDS = tuple(f"{leg}_wheel_force_z_world_N" for leg in LEGS) + ("wheel_force_z_sum_world_N",)
ACTION_FIELDS = tuple(f"{name}_actor_raw_action" for name in JOINT_NAMES)
TARGET_FIELDS = tuple(f"{name}_target_rad" for name in LEG_JOINT_NAMES)


class JointAngleTrace:
    def __init__(self, stream, env, group, terrain_names, joint_indices, wheel_indices,
                 student_experts, actor_experts):
        self.stream = stream
        self.writer = csv.writer(stream)
        self.env = env
        self.group = group
        self.terrains = terrain_names
        self.joint_indices = env.episode_length_buf.new_tensor(joint_indices)
        self.leg_indices = self.joint_indices[:len(LEG_JOINT_NAMES)]
        self.wheel_indices = wheel_indices
        self.student_experts = student_experts
        self.actor_experts = actor_experts
        self.action_step = None
        self.attempts = [0] * env.num_envs
        self.writer.writerow(["group", "terrain", "env_id", "attempt", "attempt_step",
                              "attempt_time_s", "terminal", "termination_reason",
                              *(name + "_rad" for name in JOINT_NAMES),
                              "sim_step", "sim_time_s", *VELOCITY_FIELDS, *FORCE_FIELDS,
                              *(f"student_expert_{i:02d}_weight" for i in range(student_experts)),
                              *(f"actor_expert_{i:02d}_weight" for i in range(actor_experts)),
                              *ACTION_FIELDS, *TARGET_FIELDS])

    def begin_step(self, student_weights, actor_weights, actions):
        """Snapshot the deployed Actor output before environment clipping or action delay."""
        if (student_weights.shape != (self.env.num_envs, self.student_experts)
                or actor_weights.shape != (self.env.num_envs, self.actor_experts)):
            raise ValueError("Student/Actor weight dimensions differ from the checkpoint or environment count")
        if actions.shape != self.env.dof_pos.shape:
            raise ValueError("Actor action dimensions differ from the simulator's environment and DOF counts")
        # Commands can be resampled inside env.step. The gates are tensors
        # returned by this policy forward and are never resampled for logging.
        self.action_commands = self.env.commands[:, :3].detach().clone()
        self.raw_actions = actions.detach().clone()
        self.student_weights = student_weights.detach()
        self.actor_weights = actor_weights.detach()
        self.action_step = self.env.common_step_counter + 1

    def start(self, env_ids=None):
        """Write the post-reset state before the first action and physics sample."""
        self._write(env_ids, {}, start=True)

    def sample(self, env_ids=None, terminal_reasons=None):
        """Write refreshed state at the control endpoint, before any terminal reset."""
        self._write(env_ids, terminal_reasons or {}, start=False)

    def _write(self, env_ids, reasons, start):
        # Import lazily so CLI setup cannot import torch ahead of Isaac Gym.
        import torch
        env = self.env
        sim_step = env.common_step_counter
        if env_ids is None:
            row_ids = range(env.num_envs)
            ids = slice(None)
        else:
            if not len(env_ids):
                return
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.joint_indices.device)
            row_ids = ids.tolist()
        if not start and self.action_step != sim_step:
            raise RuntimeError("Joint-angle trace requires this control step's commands and policy routing")
        commands = env.commands[:, :3] if start else self.action_commands
        values = [env.dof_pos[ids][:, self.joint_indices], commands[ids],
                  env.base_lin_vel[ids, :2], env.base_ang_vel[ids, 2:3]]
        if start:
            steps = [0] * len(row_ids)
            # No physics step or deployed action has run for this fresh pose.
            missing = [""] * (len(FORCE_FIELDS) + self.student_experts + self.actor_experts
                              + len(ACTION_FIELDS) + len(TARGET_FIELDS))
        else:
            steps = env.episode_length_buf[ids].tolist()
            fz = env.contact_forces[ids, :, 2][:, self.wheel_indices]
            values.extend((fz, fz.sum(dim=1, keepdim=True), self.student_weights[ids], self.actor_weights[ids],
                           self.raw_actions[ids][:, self.joint_indices]))
            if env.cfg.control.control_type == "P":
                # The last physics substep has already applied its delayed/scaled action.
                values.append(env.joint_pos_target[ids][:, self.leg_indices])
                missing = []
            else:
                missing = [""] * len(TARGET_FIELDS)
        # Pack measured values on device and copy to the CPU once per batch.
        samples = torch.cat(values, dim=1).detach().cpu().tolist()
        for env_id, step, sample in zip(row_ids, steps, samples):
            if start:
                self.attempts[env_id] += 1
            reason = reasons.get(env_id, "")
            self.writer.writerow([self.group, self.terrains[env_id], env_id, self.attempts[env_id],
                                  step, step * env.dt, int(bool(reason)), reason, *sample[:len(JOINT_NAMES)],
                                  sim_step, sim_step * env.dt, *sample[len(JOINT_NAMES):], *missing])
        if start or reasons:
            self.stream.flush()


@contextmanager
def open_joint_angles(path, env, group, terrain_names, *, student_experts=0, actor_experts=0):
    """Open an exclusively created telemetry CSV, or yield None for ordinary playback."""
    if path is None:
        yield None
        return
    terrains = tuple(terrain_names)
    if len(terrains) != env.num_envs:
        raise ValueError("Joint-angle trace requires a terrain name for each environment")
    dof_names = list(env.dof_names)
    invalid = [name for name in JOINT_NAMES if dof_names.count(name) != 1]
    if invalid:
        raise ValueError("Joint-angle trace requires exactly one of each named DOF: " + ", ".join(invalid))
    joint_indices = [dof_names.index(name) for name in JOINT_NAMES]
    feet = list(env.cfg.asset.feet_names)
    wheel_indices = env.wheel_body_indices[[feet.index(f"{leg}_foot") for leg in LEGS]]
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        print(f"Recording joint angles and telemetry to {path}", flush=True)
        yield JointAngleTrace(stream, env, group, terrains, joint_indices, wheel_indices,
                             student_experts, actor_experts)
