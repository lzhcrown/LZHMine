"""Check the current geometric input through CTS learning, routing and deployment."""

import pytest
import torch

from experiments.registry import get_experiment
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.utils.cts_checkpoint import policy_from_checkpoint, validate_checkpoint_policy, validate_cts_checkpoint
from rsl_rl.utils.cts_export import CTSStudentExporter
from rsl_rl.utils.cts_observations import add_skill_id, default_observation_config


def checkpoint_for(key, geometry):
    schema = add_skill_id(default_observation_config(), privileged_only=True)
    # A change in privileged term order must not move oracle state into the student.
    fields = schema["privileged_terms"]
    fields.insert(0, fields.pop(-2))
    config = dict(get_experiment(key).POLICY)
    config.update(num_actor_obs=228, num_critic_obs=296, num_one_step_obs=57, history_length=4,
                  num_actions=16, actor_hidden_dims=[16], critic_hidden_dims=[16],
                  teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16],
                  student_frozen_terrain=geometry)
    policy = ActorCriticMoECTS(**config, observation_config=schema)
    return {"architecture": "nezha_moe_cts", "format_version": 1, "policy_config": config,
            "observation_config": schema, "model_state_dict": policy.state_dict()}


@pytest.mark.parametrize("key", ["g13_v1", "g13_v2"])
def test_only_current_geometry_changes_student_actions_and_gates(key, tmp_path):
    torch.manual_seed(17)
    checkpoint = checkpoint_for(key, True)
    policy = policy_from_checkpoint(checkpoint)
    history, frame = torch.randn(5, 228), torch.randn(5, 296)
    scan = frame[:, :187]
    expected = policy.act_inference_with_diagnostics(history, scan)
    eager_ppo = policy.action_mean_with_diagnostics(history, frame)
    torch.testing.assert_close(eager_ppo[0], expected[0])
    torch.testing.assert_close(eager_ppo[1], expected[2])
    oracle_changed = frame.clone()
    oracle_changed[:, 187:] += 10
    assert torch.equal(policy.action_mean_with_diagnostics(history, oracle_changed)[0], expected[0])

    changed = policy.act_inference_with_diagnostics(history, -scan)
    for output, alternate in zip(expected[:3], changed[:3]):
        assert (output - alternate).abs().max() > 1e-6

    student = policy.student_moe_encoder
    scan = scan.detach().requires_grad_()
    student(history, scan)[0][:, 0].sum().backward()
    assert scan.grad is None and all(p.grad is None for p in student.terrain_encoder.parameters())
    for layer in (student.moe.experts.backbone, student.moe.gating_network[0]):
        assert layer.network[0].weight.grad[:, 228:].abs().sum() > 0

    path = tmp_path / "student.pt"
    CTSStudentExporter(policy).export(path)
    exported = torch.jit.load(str(path))
    torch.testing.assert_close(exported.forward_with_diagnostics(history, scan), expected)
    torch.testing.assert_close(exported.get_expert_probabilities(history, scan), expected[1])
    torch.testing.assert_close(exported.get_actor_expert_probabilities(history, scan), expected[2])
    assert exported.get_student_frozen_terrain()
    with pytest.raises(ValueError, match="height_scan"):
        policy.act_inference(history)
    with pytest.raises(torch.jit.Error, match="height_scan"):
        exported(history)
    with pytest.raises(torch.jit.Error, match="187"):
        exported(history, frame)
    if key == "g13_v1":
        state = checkpoint["model_state_dict"]
        name = "student_moe_encoder.terrain_encoder.position"
        state[name] = state[name] + 1
        with pytest.raises(ValueError, match="conflicting weights"):
            validate_cts_checkpoint(checkpoint)
    del checkpoint["policy_config"]["student_frozen_terrain"]
    with pytest.raises(ValueError, match="student_frozen_terrain disagrees"):
        validate_cts_checkpoint(checkpoint)


@pytest.mark.parametrize("key", ["g13_v1", "g13_v2"])
def test_blind_checkpoint_keeps_history_only_and_rejects_geometry_resume(key):
    checkpoint = checkpoint_for(key, False)
    del checkpoint["policy_config"]["student_frozen_terrain"]
    blind = policy_from_checkpoint(checkpoint)
    assert blind.act_inference(torch.randn(3, 228)).shape == (3, 16)
    with pytest.raises(ValueError, match="student_frozen_terrain"):
        validate_checkpoint_policy(checkpoint, get_experiment(key).POLICY)
