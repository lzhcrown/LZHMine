"""The same raw sensor sequence must produce the same training/deployment action."""

import json
from argparse import Namespace

import pytest
import torch

from deploy.deploy_mujoco.runtime import NezhaCTSPolicyRuntime
from legged_gym.scripts.export_nezha_moe_cts import export
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.utils.cts_checkpoint import (
    load_cts_checkpoint, normalize_policy_config, policy_from_checkpoint, validate_checkpoint_policy,
)
from rsl_rl.utils.cts_export import CTSStudentExporter
from rsl_rl.utils.cts_observations import (
    CTSObservationHistory, build_proprioception, default_observation_config,
)


def make_policy(history_length=4, expert_num=8, actor_type="mlp"):
    return ActorCriticMoECTS(
        num_actor_obs=57 * history_length, num_critic_obs=295,
        num_one_step_obs=57, num_actions=16, history_length=history_length,
        actor_hidden_dims=[24], critic_hidden_dims=[24],
        teacher_encoder_hidden_dims=[24], student_encoder_hidden_dims=[24, 16],
        expert_num=expert_num,
        actor_type=actor_type, actor_expert_num=8,
    ).eval()


def make_runtime(tmp_path, history_length=4, expert_num=8, control_config=None, actor_type="mlp"):
    policy = make_policy(history_length, expert_num, actor_type)
    schema = default_observation_config(history_length)
    path = tmp_path / "policy.pt"
    CTSStudentExporter(policy, schema, control_config).export(path)
    runtime = NezhaCTSPolicyRuntime(
        str(path), default_dof_pos=[0.0] * 16,
        p_gains=[10.0] * 16, d_gains=[2.0] * 16,
        torque_limits=[100.0] * 16,
    )
    return policy, schema, runtime, path


@pytest.mark.parametrize("history_length,expert_num", [(4, 8), (2, 3)])
@pytest.mark.parametrize("actor_type", ["mlp", "moe"])
def test_export_is_student_only_and_matches_full_policy(tmp_path, history_length, expert_num, actor_type):
    policy, schema, runtime, path = make_runtime(tmp_path, history_length, expert_num, actor_type=actor_type)
    probe = torch.randn(5, 57 * history_length)
    with torch.inference_mode():
        torch.testing.assert_close(runtime.policy(probe), policy.act_inference(probe))
        torch.testing.assert_close(runtime.policy.get_expert_probabilities(probe), policy.get_expert_probabilities(probe))
        actor_weights = runtime.policy.get_actor_expert_probabilities(probe)
        torch.testing.assert_close(actor_weights, policy.get_actor_expert_probabilities(probe))
    keys = runtime.policy.state_dict().keys()
    assert all("teacher" not in key and "critic" not in key for key in keys)
    assert runtime.num_experts == expert_num
    assert runtime.expert_names == tuple("expert_{}".format(i) for i in range(expert_num))
    assert json.loads(runtime.policy.get_observation_config_json()) == schema
    actor_experts = 8 if actor_type == "moe" else 0
    assert runtime.policy.get_actor_type() == actor_type
    assert runtime.policy.get_num_actor_experts() == actor_experts
    assert actor_weights.shape == (5, actor_experts)
    if actor_experts:
        torch.testing.assert_close(actor_weights.sum(-1), torch.ones(5))
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["num_experts"] == expert_num
    assert metadata["actor_type"] == actor_type
    assert metadata["actor_expert_num"] == actor_experts


@pytest.mark.parametrize("actor_type", ["mlp", "moe"])
def test_sensor_sequence_and_reset_match_training(tmp_path, actor_type):
    torch.manual_seed(7)
    policy, schema, runtime, _ = make_runtime(tmp_path, actor_type=actor_type)
    training_history = CTSObservationHistory(1, schema)
    previous_actions = torch.zeros(1, 16)
    for step in range(8):
        if step == 4:
            runtime.reset()
            training_history.reset()
            previous_actions.zero_()
        raw = {
            "base_ang_vel": torch.randn(1, 3),
            "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
            "velocity_commands": torch.tensor([[0.7, -0.2, 0.1]]),
            "joint_pos": torch.randn(1, 16),
            "joint_vel": torch.randn(1, 16) * 8,
            "actions": previous_actions,
        }
        frame = build_proprioception(raw, schema, add_noise=False)
        history = training_history.update(frame)
        with torch.inference_mode():
            expected = policy.act_inference(history).squeeze(0).clamp(-100, 100)
        actual, torques = runtime.step(
            raw["base_ang_vel"][0], raw["projected_gravity"][0],
            raw["velocity_commands"][0], raw["joint_pos"][0], raw["joint_vel"][0],
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(runtime.observations.buffer, training_history.buffer)
        assert torch.isfinite(torques).all()
        if step in (0, 4):
            torch.testing.assert_close(runtime.observations.buffer, frame[:, None].repeat(1, 4, 1))
        # Wheel angle fields are zero; measured wheel rates remain observable.
        wheel_ids = [3, 7, 11, 15]
        assert torch.equal(frame[0, [9 + i for i in wheel_ids]], torch.zeros(4))
        torch.testing.assert_close(frame[0, [25 + i for i in wheel_ids]], raw["joint_vel"][0, wheel_ids] * 0.05)
        previous_actions.copy_(actual)


def test_hybrid_pd_uses_wheel_velocity_and_applies_limits(tmp_path):
    _, _, runtime, _ = make_runtime(tmp_path)
    actions = torch.ones(16)
    position = torch.full((16,), 0.1)
    velocity = torch.full((16,), 3.0)
    actual = runtime.compute_torques(actions, position, velocity)
    expected = torch.full((16,), 10 * (0.25 - 0.1) - 2 * 3)
    expected[[3, 7, 11, 15]] = 2 * (10 - 3)
    torch.testing.assert_close(actual, expected)
    assert runtime.compute_torques(actions * 1e6, position, velocity).abs().max() <= 100


@pytest.mark.parametrize("actor_type", ["mlp", "moe"])
def test_clipped_actions_feed_the_next_observation(tmp_path, actor_type):
    _, _, runtime, _ = make_runtime(tmp_path, control_config={"clip_actions": 0.01}, actor_type=actor_type)
    inputs = ([0.0] * 3, [0.0, 0.0, -1.0], [0.5, 0.0, 0.0], [0.0] * 16, [0.0] * 16)
    actions, _ = runtime.step(*inputs)
    assert actions.abs().max() <= 0.01
    torch.testing.assert_close(runtime.previous_actions, actions)
    runtime.step(*inputs)
    torch.testing.assert_close(runtime.observations.buffer[0, -1, -16:], actions)


def test_runtime_rejects_history_or_control_metadata_mismatch(tmp_path):
    _, _, _, path = make_runtime(tmp_path, control_config={"action_scale": 0.25})
    kwargs = dict(policy_path=str(path), default_dof_pos=[0.0] * 16,
                  p_gains=[10.0] * 16, d_gains=[2.0] * 16)
    with pytest.raises(ValueError, match="history_steps"):
        NezhaCTSPolicyRuntime(history_steps=2, **kwargs)
    with pytest.raises(ValueError, match="action_scale"):
        NezhaCTSPolicyRuntime(action_scale=0.5, **kwargs)


def test_runtime_accepts_legacy_student_without_training_head_count(monkeypatch):
    model = make_policy()
    student = CTSStudentExporter(model, default_observation_config())
    monkeypatch.delattr(CTSStudentExporter, "get_critic_task_count")
    NezhaCTSPolicyRuntime(
        policy_path=None, policy=student, default_dof_pos=[0.] * 16, p_gains=[10.] * 16,
        d_gains=[2.] * 16, expected_policy={"critic_task_count": 2},
    )


def test_export_rejects_different_term_layout(tmp_path):
    schema = default_observation_config()
    schema["proprioceptive_terms"][0]["dim"] = 4
    with pytest.raises(ValueError, match="terms"):
        CTSStudentExporter(make_policy(), schema)


@pytest.mark.parametrize("group", ["g13_v1", "g13_v2"])
@pytest.mark.parametrize("legacy_dense_critic", [False, True])
def test_g13_export_replays_with_training_student_inputs(tmp_path, group, legacy_dense_critic):
    from experiments.registry import get_experiment
    from rsl_rl.utils.cts_observations import add_skill_id, flatten_history, observation_dimensions

    schema = add_skill_id(default_observation_config(), privileged_only=True)
    policy = get_experiment(group).POLICY
    current, history, privileged = observation_dimensions(schema)
    config = dict(policy, num_actor_obs=history, num_one_step_obs=current, num_critic_obs=privileged,
                  history_length=4, num_actions=16, actor_hidden_dims=[24], critic_hidden_dims=[24],
                  teacher_encoder_hidden_dims=[24], student_encoder_hidden_dims=[24, 16])
    if legacy_dense_critic:
        config.update(critic_type="moe", critic_expert_num=8)
    model = ActorCriticMoECTS(**config, observation_config=schema).eval()
    checkpoint_path = tmp_path / "model_1.pt"
    torch.save({"architecture": "nezha_moe_cts", "format_version": 1,
                "observation_config": schema, "policy_config": config,
                "model_state_dict": model.state_dict()}, checkpoint_path)
    if legacy_dense_critic:
        with pytest.raises(ValueError, match="critic_type|critic_expert_num"):
            validate_checkpoint_policy(load_cts_checkpoint(checkpoint_path), policy)
    policy_path = export(Namespace(checkpoint=str(checkpoint_path), run=None, output=str(tmp_path / "student.pt")),
                         expected_policy=policy)
    runtime = NezhaCTSPolicyRuntime(policy_path, default_dof_pos=[0.] * 16,
                                    p_gains=[10.] * 16, d_gains=[2.] * 16, expected_policy=policy)
    assert runtime.policy.get_teacher_map_attention() == (group == "g13_v2")
    assert runtime.policy.get_teacher_frozen_terrain() == (group == "g13_v1")
    assert runtime.policy.get_critic_type() == config["critic_type"]
    assert runtime.policy.get_num_critic_experts() == config["critic_expert_num"]
    assert runtime.policy.get_critic_task_count() == 2
    assert runtime.student_frozen_terrain
    assert not any("teacher" in name for name, _ in runtime.policy.named_parameters())
    sensors = ([0.] * 3, [0., 0., -1.], [.8, 0., 0.], [0.] * 16, [0.] * 16)
    for bad_scan in (None, torch.zeros(57), torch.full((187,), float("nan"))):
        with pytest.raises(ValueError, match="height_scan"):
            runtime.step(*sensors, height_scan=bad_scan)
    assert not runtime.observations.initialized.any()
    assert runtime.policy.get_num_actor_obs() == 228
    scan = torch.linspace(-1.5, 1.5, 187)
    outputs = []
    for raw_scan in (scan, -scan):
        runtime.reset()  # Identical proprioception/history in the two trials.
        actions, _ = runtime.step(*sensors, height_scan=raw_scan)
        frames = runtime.observations.buffer
        assert frames.shape == (1, 4, 57)
        history = flatten_history(frames, model.observation_dims)
        with torch.inference_mode():
            normalized = raw_scan.clamp(-1., 1.)[None] * 2.5
            mean, gates, _, _, _ = model.act_inference_with_diagnostics(history, normalized)
            torch.testing.assert_close(actions, mean[0])
            torch.testing.assert_close(runtime.expert_probabilities(), gates[0])
        outputs.append(actions)
    assert not torch.allclose(*outputs)


@pytest.mark.parametrize("group", ["g13_v1", "g13_v2", None])
def test_runtime_rejects_the_earlier_export_with_student_id(tmp_path, group):
    from experiments.registry import get_experiment
    from rsl_rl.utils.cts_observations import add_skill_id

    schema = add_skill_id(default_observation_config())
    legacy = ActorCriticMoECTS(
        **get_experiment("g12_v5").POLICY, num_actor_obs=232, num_one_step_obs=58,
        num_critic_obs=296, history_length=4, num_actions=16, observation_config=schema,
        observation_dims=[term["dim"] for term in schema["proprioceptive_terms"]],
        teacher_map_attention=(group == "g13_v2"), actor_hidden_dims=[24], critic_hidden_dims=[24],
        teacher_encoder_hidden_dims=[24], student_encoder_hidden_dims=[24, 16],
    )
    path = CTSStudentExporter(legacy, schema).export(tmp_path / "legacy_student.pt")
    with pytest.raises(ValueError, match="observation_dims|skill_id"):
        NezhaCTSPolicyRuntime(
            path, default_dof_pos=[0.] * 16, p_gains=[10.] * 16, d_gains=[2.] * 16,
            expected_policy=get_experiment(group).POLICY if group else None,
        )


@pytest.mark.parametrize("actor_type", [None, "mlp", "moe"])
def test_checkpoint_export_restores_saved_architecture_and_schema(tmp_path, actor_type):
    config = dict(
        num_actor_obs=114, num_critic_obs=295, num_one_step_obs=57,
        num_actions=16, history_length=2, observation_dims=[3, 3, 3, 16, 16, 16],
        actor_hidden_dims=[24], critic_hidden_dims=[24],
        teacher_encoder_hidden_dims=[24], student_encoder_hidden_dims=[24, 16],
        expert_num=5,
    )
    if actor_type is not None:
        config.update(actor_type=actor_type, actor_expert_num=8)
    model = ActorCriticMoECTS(**config).eval()
    schema = default_observation_config(history_length=2)
    schema["proprioceptive_terms"][0]["scale"] = 0.125
    checkpoint_path = tmp_path / "model_7.pt"
    torch.save({
        "architecture": "nezha_moe_cts", "format_version": 1,
        "policy_config": config, "observation_config": schema,
        "model_state_dict": model.state_dict(), "control_config": {"control_dt": 0.02},
    }, checkpoint_path)
    exported_path = export(Namespace(checkpoint=str(checkpoint_path), run=None,
                                     output=str(tmp_path / "student.pt"), experiment="nezha_moe_cts"))
    scripted = torch.jit.load(exported_path)
    assert scripted.get_num_experts() == 5
    assert scripted.get_actor_type() == (actor_type or "mlp")
    assert scripted.get_num_actor_experts() == (8 if actor_type == "moe" else 0)
    assert json.loads(scripted.get_observation_config_json()) == schema
    assert json.loads(scripted.get_control_config_json()) == {"control_dt": 0.02}
    probe = torch.randn(3, 114)
    with torch.inference_mode():
        torch.testing.assert_close(scripted(probe), model.act_inference(probe))
        restored = policy_from_checkpoint(load_cts_checkpoint(checkpoint_path))
        torch.testing.assert_close(restored.act_inference(probe), model.act_inference(probe))

    # A MoE checkpoint cannot masquerade as the old MLP architecture, or vice versa.
    malformed = torch.load(checkpoint_path, weights_only=False)
    if actor_type == "moe":
        malformed["policy_config"].pop("actor_type")
        malformed["policy_config"].pop("actor_expert_num")
    else:
        malformed["policy_config"].update(actor_type="moe", actor_expert_num=8)
    torch.save(malformed, checkpoint_path)
    with pytest.raises(ValueError, match="actor_type"):
        load_cts_checkpoint(checkpoint_path)


def test_policy_config_normalization_does_not_mutate_legacy_config():
    legacy = {"actor_hidden_dims": [24]}
    restored = normalize_policy_config(legacy)
    assert restored["actor_type"] == "mlp"
    assert restored["actor_expert_num"] == 0
    assert "actor_type" not in legacy
    assert normalize_policy_config({"actor_type": "moe"})["actor_expert_num"] == 8
    with pytest.raises(ValueError, match="actor_expert_num"):
        normalize_policy_config({"actor_type": "moe", "actor_expert_num": 0})
