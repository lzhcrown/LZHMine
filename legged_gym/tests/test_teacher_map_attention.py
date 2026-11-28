"""Regressions for using the original G13 height and skill fields in the teacher."""

import pytest
import torch

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.modules.teacher_map_attention import TeacherMapAttention
from rsl_rl.utils.cts_observations import add_skill_id, default_observation_config


def skill_schema():
    return add_skill_id(default_observation_config(), privileged_only=True)


def make_policy(enabled):
    return ActorCriticMoECTS(
        228, 296, 57, 16,
        teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16],
        actor_hidden_dims=[16], critic_hidden_dims=[16], actor_type="moe",
        teacher_map_attention=enabled, observation_config=skill_schema(),
    )


def test_named_height_and_skill_separation_and_xy_order():
    schema = skill_schema()
    # Move height_scan to the front to catch accidental fixed-offset slicing.
    fields = schema["privileged_terms"]
    fields.insert(0, fields.pop(-2))
    branch = TeacherMapAttention(schema, 296, 32)
    assert branch.height_start == 0
    assert len(branch.query_indices) == 58
    assert branch.query_indices.min() == 190  # after map and privileged base linear velocity
    assert branch.query_indices[-1] == 295  # actual teacher skill
    assert torch.allclose(branch.xy[0, [0, 1, 17, -1]].cpu(), torch.tensor([
        [-0.8, -0.5], [-0.7, -0.5], [-0.8, -0.4], [0.8, 0.5],
    ]))
    points = torch.zeros(2, 187, 3)
    points[:, :, :2] = branch.xy
    branch.validate_grid(points)
    with pytest.raises(ValueError, match="0.1 m grid"):
        branch.validate_grid(points.flip(1))
    schema["height_grid_order"] = "yx"
    with pytest.raises(ValueError, match="x-fastest"):
        TeacherMapAttention(schema, 296, 32)


def test_teacher_residual_matches_baseline_then_learns_without_changing_student():
    torch.manual_seed(250)
    baseline = make_policy(False)
    experimental = make_policy(True)
    missing, unexpected = experimental.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all(name.startswith("teacher_encoder.map_attention.") for name in missing)
    assert not unexpected

    history = torch.randn(3, 228)
    privileged = torch.randn(3, 296)
    teacher = experimental.teacher_encoder
    for is_teacher in (False, True):
        original = baseline.action_mean_with_diagnostics(history, privileged, is_teacher)[0]
        actual = experimental.action_mean_with_diagnostics(history, privileged, is_teacher)[0]
        assert torch.equal(actual, original)
    assert torch.equal(teacher(privileged), baseline.teacher_encoder(privileged))

    # One policy update trains the zero-start projection; subsequent updates
    # must propagate back into both the map CNN and proprioceptive query.
    optimizer = torch.optim.SGD(teacher.map_attention.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        means = experimental.action_mean_with_diagnostics(history, privileged, is_teacher=True)[0]
        means[:, 0].sum().backward()
        assert teacher.map_attention.to_latent.weight.grad.abs().sum() > 0
        optimizer.step()
    assert teacher.map_attention.query.weight.grad.abs().sum() > 0
    assert teacher.map_attention.height_features[0].weight.grad.abs().sum() > 0
    assert torch.equal(baseline.act_inference(history), experimental.act_inference(history))
    # Hold every common physical field and the task fixed, change the map.
    moved_step = privileged.clone()
    start = teacher.map_attention.height_start
    moved_step[:, start:start + 187] = privileged[:, start:start + 187].flip(1)
    assert not torch.equal(teacher.map_attention(privileged), teacher.map_attention(moved_step))


def test_only_the_teacher_can_condition_actions_on_the_oracle_id():
    torch.manual_seed(63)
    policy = make_policy(True).eval()
    history = torch.randn(5, 228)
    ordinary = torch.randn(5, 296)
    ordinary[:, -1] = 0
    platform = ordinary.clone()
    platform[:, -1] = 1
    with torch.inference_mode():
        assert not torch.equal(policy.teacher_encoder(ordinary), policy.teacher_encoder(platform))
        # Even during PPO rollouts, changing only the privileged ID cannot
        # change the student's means or actor expert weights.
        mean0, gate0, _, _ = policy.action_mean_with_diagnostics(history, ordinary, is_teacher=False)
        mean1, gate1, _, _ = policy.action_mean_with_diagnostics(history, platform, is_teacher=False)
        assert torch.equal(mean0, mean1) and torch.equal(gate0, gate1)
        assert torch.equal(mean0, policy.act_inference(history))
