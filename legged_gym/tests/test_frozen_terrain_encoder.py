import pytest
import torch

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.modules.frozen_terrain_encoder import TerrainEncoder, terrain_encoder_checkpoint
from rsl_rl.utils.cts_observations import add_skill_id, default_observation_config


def make_policy(schema=None):
    return ActorCriticMoECTS(
        228, 296, 57, 16,
        teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16],
        actor_hidden_dims=[16], critic_hidden_dims=[16],
        teacher_frozen_terrain=True,
        observation_config=schema or add_skill_id(default_observation_config(), privileged_only=True),
    )


def test_map_pretraining_hides_masked_cells_and_trains_attention():
    torch.manual_seed(21)
    encoder = TerrainEncoder()
    scan = torch.randn(4, 187)
    mask = torch.rand_like(scan) > 0.6
    code = encoder(scan, mask)
    assert code.shape == (4, 16)
    torch.testing.assert_close(code.norm(dim=-1), torch.ones(4))
    assert torch.equal(code, encoder(scan.masked_fill(mask, 1000), mask))
    torch.testing.assert_close(encoder(scan), encoder(scan, torch.zeros_like(mask)))
    code[:, 0].sum().backward()
    for p in encoder.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_stage_two_teacher_replaces_the_named_scan_and_freezes_only_geometry(tmp_path):
    torch.manual_seed(34)
    schema = add_skill_id(default_observation_config(), privileged_only=True)
    fields = schema["privileged_terms"]
    fields.insert(0, fields.pop(-2))  # catch accidental fixed-offset extraction
    policy = make_policy(schema)
    encoder = TerrainEncoder()
    path = tmp_path / "geometry.pt"
    torch.save(terrain_encoder_checkpoint(encoder), path)
    teacher = policy.teacher_encoder
    teacher.load_pretrained(path)
    policy.train()
    assert all(not p.requires_grad for p in teacher.terrain_encoder.parameters())
    assert not teacher.terrain_encoder.training
    privileged = torch.randn(4, 296, requires_grad=True)
    terrain = encoder(privileged[:, :187])
    expected = teacher[1](teacher[0](torch.cat((privileged[:, 187:], terrain), dim=1)))
    torch.testing.assert_close(teacher(privileged), expected)
    assert teacher[0].network[0].in_features == 125
    assert not torch.equal(teacher(privileged), teacher(privileged.index_fill(1, torch.arange(187), 0)))
    assert not torch.equal(teacher(privileged), teacher(privileged.index_fill(1, torch.tensor([295]), 0)))

    history = torch.randn(4, 228)
    student = policy.action_mean_with_diagnostics(history, privileged)[0]
    assert torch.equal(student, policy.action_mean_with_diagnostics(history, -privileged)[0])
    policy.action_mean_with_diagnostics(history, privileged, is_teacher=True)[0].sum().backward()
    assert torch.count_nonzero(privileged.grad[:, :187]) == 0
    assert torch.count_nonzero(privileged.grad[:, 187:]) > 0
    assert teacher[0].network[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.terrain_encoder.parameters())


def test_frozen_terrain_rejects_student_skill_and_incompatible_artifacts(tmp_path):
    with pytest.raises(ValueError, match="blind student"):
        make_policy(add_skill_id(default_observation_config()))
    schema = add_skill_id(default_observation_config(), privileged_only=True)
    schema["privileged_terms"][-2]["scale"] = 1.0
    with pytest.raises(ValueError, match="scaling"):
        make_policy(schema)
    teacher = make_policy().teacher_encoder
    valid = terrain_encoder_checkpoint(TerrainEncoder())
    path = tmp_path / "damaged.pt"
    torch.save(dict(valid, scan_config={}), path)
    with pytest.raises(ValueError, match="convention"):
        teacher.load_pretrained(path)
    valid["encoder_state_dict"]["to_latent.weight"].fill_(float("nan"))
    torch.save(valid, path)
    with pytest.raises(ValueError, match="weights"):
        teacher.load_pretrained(path)
