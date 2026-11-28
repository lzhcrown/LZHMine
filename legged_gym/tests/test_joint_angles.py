"""Control-cycle joint snapshots, including independent terminal/reset pairs."""

import csv
from types import SimpleNamespace

import pytest
import torch

from legged_gym.scripts._joint_angles import open_joint_angles


def make_env():
    # The simulator's DOF ordering can differ from the requested leg ordering.
    names = [f"{leg}_{joint}_joint" for joint in ("foot", "thigh", "calf", "hip")
             for leg in ("FR", "RL", "RR", "FL")]
    return SimpleNamespace(
        num_envs=2, dof_names=names, dt=.02, episode_length_buf=torch.zeros(2, dtype=torch.long),
        dof_pos=torch.arange(-10., 22.).view(2, 16), joint_pos_target=torch.zeros(2, 16),
        common_step_counter=0, commands=torch.tensor([[.8, .3, 1., 2.], [-.1, -.2, -.3, 2.]]),
        base_lin_vel=torch.tensor([[1., 2., 12.], [-33., -2., 8.]]),
        base_ang_vel=torch.tensor([[12., 15., -5.], [3., 22., -4.]]),
        cfg=SimpleNamespace(asset=SimpleNamespace(feet_names=["RR_foot", "FL_foot", "RL_foot", "FR_foot"]),
                            control=SimpleNamespace(control_type="P")),
        wheel_body_indices=torch.tensor([5, 9, 2, 6]),
        contact_forces=torch.arange(-31., 29.).reshape(2, 10, 3),
    )


def test_clock_pose_and_terminal_sample_across_independent_resets(tmp_path):
    env = make_env()
    path = tmp_path / "run" / "angles.csv"
    initial = env.dof_pos.clone()
    student = torch.tensor([[.2, .8], [.7, .3]])
    actor = torch.tensor([[.1, .9], [.8, .2]])
    before_step = env.commands.clone()
    first_action = torch.arange(-16., 16.).reshape(2, 16)
    second_action = first_action * -.4
    first_target = initial * .125 + .5
    second_target = initial + 1.
    with open_joint_angles(path, env, "g12_v3", ["flat", "platform_100cm"],
                           student_experts=2, actor_experts=2) as trace:
        trace.start()
        action = first_action.clone()
        trace.begin_step(student, actor, action)
        action.clamp_(-1., 1.)
        env.joint_pos_target.copy_(first_target)
        # The callback changes these before termination. The action was
        # generated with the previous commands.
        env.commands += 1.
        env.dof_pos += .25
        env.episode_length_buf[:] = 1
        env.common_step_counter += 1
        trace.sample(None, {1: "stable_landing"})
        env.dof_pos[1] = -1.
        env.episode_length_buf[1] = 0
        trace.start(torch.tensor([1]))
        trace.begin_step(actor, student, second_action)
        env.joint_pos_target.copy_(second_target)
        env.dof_pos += .125
        env.episode_length_buf += 1
        env.common_step_counter += 1
        trace.sample([0], {0: "time_limit"})
        trace.sample(torch.tensor([1]))
        trace.sample([])
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        actual_fields = [key for key in reader.fieldnames if key.endswith("_joint_rad")]
        target_fields = [key for key in reader.fieldnames if key.endswith("_target_rad")]
        action_fields = [key for key in reader.fieldnames if key.endswith("_actor_raw_action")]
    assert set(actual_fields) == {name + "_rad" for name in env.dof_names}
    assert len(actual_fields) == 16
    assert set(target_fields) == {name + "_target_rad" for name in env.dof_names if "foot" not in name}
    assert set(action_fields) == {name + "_actor_raw_action" for name in env.dof_names}
    assert [(r["env_id"], r["attempt"], r["attempt_step"], r["termination_reason"]) for r in rows] == [
        ("0", "1", "0", ""), ("1", "1", "0", ""),
        ("0", "1", "1", ""), ("1", "1", "1", "stable_landing"), ("1", "2", "0", ""),
        ("0", "1", "2", "time_limit"), ("1", "2", "1", ""),
    ]
    for row in rows:
        assert row["group"] == "g12_v3"
        assert row["terrain"] == ["flat", "platform_100cm"][int(row["env_id"])]
        assert float(row["attempt_time_s"]) == pytest.approx(int(row["attempt_step"]) * .02)
        assert bool(int(row["terminal"])) == bool(row["termination_reason"])
    for i, name in enumerate(env.dof_names):
        assert float(rows[0][name + "_rad"]) == initial[0, i].item()
        assert float(rows[3][name + "_rad"]) == initial[1, i].item() + .25
        assert float(rows[4][name + "_rad"]) == -1.

    command_fields = ["command_vx_body_m_s", "command_vy_body_m_s", "command_yaw_rate_rad_s"]
    velocity_fields = ["base_vx_body_m_s", "base_vy_body_m_s", "base_yaw_rate_body_rad_s"]
    force_fields = [leg + "_wheel_force_z_world_N" for leg in ("FL", "FR", "RL", "RR")]
    gate_fields = [f"{module}_expert_{i:02d}_weight" for module in ("student", "actor") for i in range(2)]
    assert [int(row["sim_step"]) for row in rows] == [0, 0, 1, 1, 1, 2, 2]
    for row in rows:
        assert float(row["sim_time_s"]) == pytest.approx(int(row["sim_step"]) * .02)
        env_id = int(row["env_id"])
        assert [float(row[name]) for name in velocity_fields] == [
            *env.base_lin_vel[env_id, :2].tolist(), env.base_ang_vel[env_id, 2].item()]
        if row["attempt_step"] == "0":
            assert all(row[name] == "" for name in [*force_fields, "wheel_force_z_sum_world_N", *gate_fields,
                                                    *target_fields, *action_fields])
        else:
            forces = env.contact_forces[env_id, [9, 6, 2, 5], 2]
            assert [float(row[name]) for name in force_fields] == forces.tolist()
            assert float(row["wheel_force_z_sum_world_N"]) == pytest.approx(forces.sum().item())
            first_step = row["sim_step"] == "1"
            cmd = before_step if first_step else env.commands
            assert [float(row[name]) for name in command_fields] == cmd[env_id, :3].tolist()
            student_row, actor_row = (student, actor) if first_step else (actor, student)
            assert [float(row[name]) for name in gate_fields] == [
                *student_row[env_id].tolist(), *actor_row[env_id].tolist()]
            sent = first_action if first_step else second_action
            applied = first_target if first_step else second_target
            for i, name in enumerate(env.dof_names):
                assert float(row[name + "_actor_raw_action"]) == sent[env_id, i].item()
                if "foot" not in name:
                    assert float(row[name + "_target_rad"]) == applied[env_id, i].item()
    assert [float(rows[4][name]) for name in command_fields] == env.commands[1, :3].tolist()


@pytest.mark.parametrize("student_n,actor_n", [(0, 0), (0, 3), (2, 0)])
def test_trace_only_exports_real_moe_columns(tmp_path, student_n, actor_n):
    env = make_env()
    path = tmp_path / "angles.csv"
    with open_joint_angles(path, env, "test", ["flat"] * 2,
                           student_experts=student_n, actor_experts=actor_n) as trace:
        trace.start()
        student = torch.empty(2, student_n).fill_(1. / max(1, student_n))
        actor = torch.empty(2, actor_n).fill_(1. / max(1, actor_n))
        trace.begin_step(student, actor, torch.zeros_like(env.dof_pos))
        env.common_step_counter += 1
        env.episode_length_buf += 1
        trace.sample()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        gate_fields = [name for name in reader.fieldnames if name.endswith("_weight")]
        last = list(reader)[-1]
    assert len(gate_fields) == student_n + actor_n
    assert "actor_skill_id" not in last
    assert all(float(last[name]) > 0. for name in gate_fields)


def test_legacy_trace_records_newest_skill_from_the_actual_actor_input(tmp_path):
    env = make_env()
    env.observation_config = {
        "history_length": 2,
        "proprioceptive_terms": [{"name": "base", "dim": 2}, {"name": "skill_id", "dim": 1}],
    }
    observations = torch.zeros(2, 6)
    observations[:, -2:] = torch.tensor([[1., 0.], [0., 1.]])
    gates = torch.empty(2, 0)
    path = tmp_path / "skills.csv"
    with open_joint_angles(path, env, "legacy_labelled_student", ["stairs_convex_35cm", "platform_100cm"]) as trace:
        trace.start()
        trace.begin_step(gates, gates, torch.zeros_like(env.dof_pos), observations)
        observations.zero_()  # env.step can replace the observation, including on reset
        env.common_step_counter += 1
        env.episode_length_buf += 1
        trace.sample(terminal_reasons={1: "fall"})
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["actor_skill_id"] for row in rows] == ["", "", "0.0", "1.0"]


@pytest.mark.parametrize("mode", ["V", "T"])
def test_non_position_control_targets_are_blank_and_raw_actions_remain_available(tmp_path, mode):
    env = make_env()
    env.cfg.control.control_type = mode
    env.joint_pos_target.fill_(1.5)  # Also populated by the V/T torque implementation.
    path = tmp_path / "angles.csv"
    actions = torch.full_like(env.dof_pos, 3.)
    with open_joint_angles(path, env, "test", ["flat"] * 2) as trace:
        trace.begin_step(torch.empty(2, 0), torch.empty(2, 0), actions)
        env.common_step_counter += 1
        env.episode_length_buf += 1
        trace.sample()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            assert [value for key, value in row.items() if key.endswith("_target_rad")] == [""] * 12
            assert [float(value) for key, value in row.items() if key.endswith("_actor_raw_action")] == [3.] * 16


def test_disabled_trace_and_existing_file(tmp_path):
    with open_joint_angles(None, None, None, None) as trace:
        assert trace is None
    path = tmp_path / "angles.csv"
    path.write_text("previous capture\n")
    with pytest.raises(FileExistsError), open_joint_angles(path, make_env(), "g12_v3", ["flat"] * 2):
        pass
    assert path.read_text() == "previous capture\n"


def test_missing_joint_reports_urdf_name_before_creating_file(tmp_path):
    env = make_env()
    env.dof_names.remove("RR_hip_joint")
    path = tmp_path / "angles.csv"
    with pytest.raises(ValueError, match="RR_hip_joint"), open_joint_angles(path, env, "g12_v3", ["flat"] * 2):
        pass
    assert not path.exists()
