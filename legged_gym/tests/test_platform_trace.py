"""Validate the student replay CSV without importing Isaac Gym."""

import csv
import math
from types import SimpleNamespace

import pytest
import torch

from legged_gym.scripts._platform_trace import open_platform_trace


def set_orientation(env, yaw, pitch=0.):
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    env.root_states[0, 3:7] = torch.tensor([-sy * sp, cy * sp, sy * cp, cy * cp])


def trace_env():
    # Shuffle both DOF and contact body order, as restored assets name them independently.
    names = ["RR_foot", "FR_calf", "RL_foot", "FL_thigh", "FR_foot", "FL_foot", "FL_calf", "FR_thigh"]
    return SimpleNamespace(
        num_envs=2, num_bodies=5, dt=.02,
        cfg=SimpleNamespace(control=SimpleNamespace(control_type="P", vel_scale=10.),
                            asset=SimpleNamespace(feet_names=["RR_foot", "FR_foot", "RL_foot", "FL_foot"]),
                            terrain=SimpleNamespace(wf_sub_terrains={"platform_block": {"platform_width": 5.}})),
        dof_names=[name + "_joint" for name in names],
        wheel_body_indices=torch.tensor([1, 4, 3, 0]),
        rigid_body_states=torch.arange(130).reshape(10, 13).float(),
        contact_forces=torch.arange(30).reshape(2, 5, 3).float() + 100.,
        joint_pos_target=torch.arange(16).reshape(2, 8).float() + 10.,
        dof_pos=torch.arange(16).reshape(2, 8).float() + 20.,
        delayed_actions=torch.arange(32).reshape(2, 2, 8).float(),
        root_states=torch.zeros(2, 13), base_ang_vel=torch.tensor([[0., 0., 3.], [0., 0., 0.]]),
    )


def test_trace_aligns_commands_world_contacts_and_last_applied_targets(tmp_path):
    env = trace_env()
    set_orientation(env, math.pi - .02)
    path = tmp_path / "run.csv"
    with open_platform_trace(path, env, (.8, 0., 0.), "block", .50) as trace:
        set_orientation(env, -math.pi + .02, pitch=-.25)
        env.root_states[0, :3] = torch.tensor([3., 4., 5.])
        trace.record(1, torch.tensor([False, True]))
        set_orientation(env, .4)  # state after auto-reset belongs to the next attempt
        trace.record(2, torch.tensor([True, False]))
        set_orientation(env, .42)
        trace.record(3, torch.tensor([False, False]))
    with path.open() as stream:
        first, terminal, next_trial = csv.DictReader(stream)
    expected = {
        "time_s": .02, "cmd_vx_body_m_s": .8, "cmd_yaw_rate_rad_s": 0,
        "platform_width_m": 5., "platform_height_m": .5, "base_z_world_m": 5.,
        "base_yaw_world_rad": -math.pi + .02, "base_pitch_rad": -.25,
        "base_yaw_rate_world_rad_s": 2., "base_omega_z_body_rad_s": 3.,
        "FL_wheel_x_world_m": 0, "FL_wheel_force_z_world_N": 102,
        "FR_wheel_z_world_m": 54, "FR_wheel_force_x_world_N": 112,
        "FL_thigh_target_rad": 13, "FL_thigh_actual_rad": 23,
        "FR_calf_target_rad": 11, "FR_calf_actual_rad": 21,
        "FL_wheel_target_rad_s": 130, "FR_wheel_target_rad_s": 120,
        "RL_wheel_target_rad_s": 100, "RR_wheel_target_rad_s": 80,
    }
    for key, value in expected.items():
        assert float(first[key]) == pytest.approx(value, abs=1e-5)
    assert first["attempt"] == terminal["attempt"] == "1"
    assert terminal["terminal"] == "1" and terminal["base_z_world_m"] == ""
    assert terminal["FL_wheel_force_z_world_N"] == ""
    assert next_trial["attempt"] == "2" and float(next_trial["attempt_time_s"]) == .02
    assert float(next_trial["base_yaw_rate_world_rad_s"]) == pytest.approx(1., abs=1e-5)
    original = path.read_bytes()
    with pytest.raises(FileExistsError), open_platform_trace(path, env, (.8, 0., 0.), "block", .50):
        pass
    assert path.read_bytes() == original


def test_trace_requires_fixed_platform_and_accepts_disabled_mode(tmp_path):
    path = tmp_path / "invalid.csv"
    with open_platform_trace(None, None, None, None, None) as trace:
        assert trace is None
    with pytest.raises(ValueError, match="fixed"), open_platform_trace(path, None, None, None, None):
        pass
    assert not path.exists()
