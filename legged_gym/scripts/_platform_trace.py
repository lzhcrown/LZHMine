"""Trace the deployed student's first environment during fixed-platform play."""

import csv
import math
from contextlib import contextmanager
from pathlib import Path

import torch


FRONT_LEGS = ("FL_thigh", "FL_calf", "FR_thigh", "FR_calf")
WHEELS = ("FL", "FR", "RL", "RR")


def _yaw_pitch(quat):
    x, y, z, w = quat  # Isaac Gym quaternion order
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1., min(1., 2 * (w * y - z * x))))
    return yaw, pitch


class PlatformPlayTrace:
    def __init__(self, stream, env, command, shape, height):
        self.env = env
        self.writer = csv.writer(stream)
        self.command = command
        self.shape = shape
        self.height = height
        self.width = env.cfg.terrain.wf_sub_terrains["platform_" + shape]["platform_width"]
        self.attempt = 1
        self.attempt_step = 0
        self.prev_yaw = _yaw_pitch(env.root_states[0, 3:7].tolist())[0]
        self.leg_ids = [env.dof_names.index(name + "_joint") for name in FRONT_LEGS]
        self.wheel_ids = [env.dof_names.index(name + "_foot_joint") for name in WHEELS]
        body_ids = env.wheel_body_indices.tolist()
        self.front_bodies = [body_ids[env.cfg.asset.feet_names.index(name + "_foot")] for name in WHEELS[:2]]
        fields = [
            "platform_shape", "platform_height_m", "platform_width_m", "step", "time_s",
            "attempt", "attempt_time_s", "terminal",
            "cmd_vx_body_m_s", "cmd_vy_body_m_s", "cmd_yaw_rate_rad_s",
            "base_x_world_m", "base_y_world_m", "base_z_world_m", "base_yaw_world_rad",
            "base_pitch_rad", "base_yaw_rate_world_rad_s", "base_omega_z_body_rad_s",
        ]
        for side in WHEELS[:2]:
            fields += [side + "_wheel_" + axis + "_world_m" for axis in "xyz"]
            fields += [side + "_wheel_force_" + axis + "_world_N" for axis in "xyz"]
        for joint in FRONT_LEGS:
            fields += [joint + "_target_rad", joint + "_actual_rad"]
        fields += [name + "_wheel_target_rad_s" for name in WHEELS]
        self.num_fields = len(fields)
        self.writer.writerow(fields)

    def record(self, step, done):
        env = self.env
        self.attempt_step += 1
        terminal = bool(done.reshape(-1)[0].item())
        row = [self.shape, self.height, self.width, step, step * env.dt, self.attempt, self.attempt_step * env.dt,
               int(terminal), *self.command]
        if terminal:
            # step() has already reset root/joint states and cleared contact data.
            self.writer.writerow(row + [""] * (self.num_fields - len(row)))
            self.prev_yaw = _yaw_pitch(env.root_states[0, 3:7].tolist())[0]
            self.attempt += 1
            self.attempt_step = 0
            return
        bodies = env.rigid_body_states.view(env.num_envs, env.num_bodies, 13)
        wheel_state = torch.cat((bodies[0, self.front_bodies, :3],
                                 env.contact_forces[0, self.front_bodies]), dim=1)
        front_legs = torch.stack((env.joint_pos_target[0, self.leg_ids],
                                  env.dof_pos[0, self.leg_ids]), dim=1)
        # Nezha's P controller uses the last clipped/delayed substep action
        # times vel_scale as its wheel velocity reference.
        wheel_targets = env.delayed_actions[0, -1, self.wheel_ids] * env.cfg.control.vel_scale
        values = torch.cat((env.root_states[0, :7], env.base_ang_vel[0, 2:3], wheel_state.flatten(),
                            front_legs.flatten(), wheel_targets)).detach().cpu().tolist()
        yaw, pitch = _yaw_pitch(values[3:7])
        delta = yaw - self.prev_yaw
        yaw_rate = math.atan2(math.sin(delta), math.cos(delta)) / env.dt
        self.prev_yaw = yaw
        self.writer.writerow(row + values[:3] + [yaw, pitch, yaw_rate] + values[7:])


@contextmanager
def open_platform_trace(path, env, command, shape, height):
    if path is None:
        yield None
        return
    if shape is None or height is None:
        raise ValueError("--platform_trace requires a fixed --platform_shape and --platform_height")
    if env.cfg.control.control_type != "P":
        raise ValueError("Platform trace actuator targets require Nezha's P controller")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        print("Recording environment 0 to {}. Terminal rows have blank state fields.".format(path))
        yield PlatformPlayTrace(stream, env, command, shape, height)
