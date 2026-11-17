"""Residual policies preserve CTS ownership and use one deployed action mean."""

import copy
import json

import pytest
import torch
from torch.distributions import Normal

from deploy.deploy_mujoco.runtime import NezhaCTSPolicyRuntime
from rsl_rl.algorithms.moe_cts import MoECTS
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.modules.moe import MoE, MoEActor
from rsl_rl.utils.cts_checkpoint import (
    normalize_policy_config, policy_from_checkpoint, validate_cts_checkpoint,
)
from rsl_rl.utils.cts_export import CTSStudentExporter
from rsl_rl.utils.cts_observations import (
    CTSObservationHistory, build_proprioception, default_observation_config,
)


SCALES = [0.05 if index in (3, 7, 11, 15) else 0.1 for index in range(16)]


def policy_config(residual=False):
    config = dict(
        num_actor_obs=228, num_critic_obs=295, num_one_step_obs=57,
        num_actions=16, history_length=4, observation_dims=[3, 3, 3, 16, 16, 16],
        actor_hidden_dims=[24, 16], critic_hidden_dims=[24, 16],
        teacher_encoder_hidden_dims=[24], student_encoder_hidden_dims=[24, 16],
        student_encoder_type="moe", actor_type="moe", critic_type="moe",
        expert_num=8, actor_expert_num=8, critic_expert_num=8,
        latent_dim=32,
    )
    if residual:
        config.update(residual_enabled=True, residual_hidden_dims=[16, 8],
                      residual_action_scales=SCALES)
    return config


def make_checkpoint(policy, config):
    return dict(architecture="nezha_moe_cts", format_version=1,
                model_state_dict=copy.deepcopy(policy.state_dict()),
                policy_config=copy.deepcopy(config),
                observation_config=default_observation_config(4))


def has_gradient(module):
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in module.parameters())


def test_disabled_actor_retains_legacy_parameters_and_rng_sequence():
    torch.manual_seed(103)
    old = MoE(8, 89, [24, 16], 16)
    old_rng = torch.rand(7)
    torch.manual_seed(103)
    actor = MoEActor(8, 89, [24, 16], 16)
    new_rng = torch.rand(7)
    assert torch.equal(old_rng, new_rng)
    assert set(actor.state_dict()) == {"moe." + name for name in old.state_dict()}
    inputs = torch.randn(4, 89)
    assert torch.equal(old(inputs)[0], actor(inputs))


def test_zero_initialized_residual_exactly_preserves_base_policy():
    base = ActorCriticMoECTS(**policy_config())
    residual = ActorCriticMoECTS(**policy_config(True))
    missing, unexpected = residual.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(name.startswith("actor.residual.") or
                           name == "actor.residual_action_scales" for name in missing)
    history, privileged = torch.randn(6, 228), torch.randn(6, 295)
    for is_teacher in (False, True):
        expected = base.action_mean_with_diagnostics(history, privileged, is_teacher)
        actual = residual.action_mean_with_diagnostics(history, privileged, is_teacher)
        for index in (0, 1, 2):
            assert torch.equal(expected[index], actual[index])
        assert torch.equal(actual[3], torch.zeros(6, 16))
    assert torch.equal(base.act_inference(history), residual.act_inference(history))


def test_residual_initialization_preserves_shared_weights_and_environment_rng():
    torch.manual_seed(42)
    base = ActorCriticMoECTS(**policy_config())
    base_rng = torch.rand(32)
    torch.manual_seed(42)
    residual = ActorCriticMoECTS(**policy_config(True))
    assert torch.equal(torch.rand(32), base_rng)
    for name, tensor in base.state_dict().items():
        assert torch.equal(residual.state_dict()[name], tensor), name


def test_nonzero_residual_is_bounded_and_all_policy_paths_use_final_mean():
    policy = ActorCriticMoECTS(**policy_config(True))
    with torch.no_grad():
        policy.actor.residual.network[-1].bias.copy_(torch.linspace(-3, 3, 16))
    history, privileged = torch.randn(5, 228), torch.randn(5, 295)
    final, student_gate, actor_gate, coarse, correction = policy.act_inference_with_diagnostics(history)
    assert torch.all(correction.abs() <= torch.tensor(SCALES))
    assert bool(correction.abs().sum() > 0)
    torch.testing.assert_close(final, coarse + correction)
    torch.testing.assert_close(policy.act_inference(history), final)
    torch.testing.assert_close(policy.get_expert_probabilities(history), student_gate)
    torch.testing.assert_close(policy.get_actor_expert_probabilities(history), actor_gate)
    actions = policy.act(history, privileged)
    torch.testing.assert_close(policy.action_mean, final)
    torch.testing.assert_close(policy.coarse_action_mean, coarse)
    torch.testing.assert_close(policy.action_residual, correction)
    expected = Normal(final, policy.std.clamp_min(1e-6).expand_as(final)).log_prob(actions).sum(-1)
    torch.testing.assert_close(policy.get_actions_log_prob(actions), expected)

    # A temporal regularizer may query earlier means after the current Normal
    # has been built. Such queries must not overwrite PPO's distribution/cache.
    distribution = policy.distribution
    cached_mean = policy.action_mean.clone()
    policy.action_mean_with_diagnostics(history + 1, privileged, is_teacher=True)
    assert policy.distribution is distribution
    assert torch.equal(policy.action_mean, cached_mean)


def test_wheel_and_leg_bounds_follow_action_indices_not_contiguous_blocks():
    policy = ActorCriticMoECTS(**policy_config(True))
    with torch.no_grad():
        policy.actor.residual.network[-1].bias.fill_(100)
    correction = policy.act_inference_with_diagnostics(torch.randn(2, 228))[-1]
    torch.testing.assert_close(correction, torch.tensor(SCALES).expand(2, -1))
    torch.testing.assert_close(correction[:, [3, 7, 11, 15]], torch.full((2, 4), 0.05))
    torch.testing.assert_close(correction[:, [0, 1, 2, 4]], torch.full((2, 4), 0.1))


@pytest.mark.parametrize("is_teacher", [False, True])
def test_residual_policy_gradient_ownership_and_optimizer_membership(is_teacher):
    policy = ActorCriticMoECTS(**policy_config(True))
    algorithm = MoECTS(policy, num_envs=4, schedule="fixed")
    residual_ids = {id(parameter) for parameter in policy.actor.residual.parameters()}
    ppo_ids = {id(parameter) for parameter in algorithm.ppo_parameters}
    assert residual_ids <= ppo_ids
    history, privileged = torch.randn(4, 228), torch.randn(4, 295)
    before = policy.actor.residual.network[-1].weight.detach().clone()
    mean = policy.action_mean_with_diagnostics(history, privileged, is_teacher)[0]
    mean.square().mean().backward()
    assert has_gradient(policy.actor.residual)
    assert has_gradient(policy.actor.moe)
    assert has_gradient(policy.teacher_encoder) == is_teacher
    assert not has_gradient(policy.student_moe_encoder)
    assert not has_gradient(policy.critic)
    algorithm.optimizer.step()
    assert not torch.equal(before, policy.actor.residual.network[-1].weight)

    policy.zero_grad(set_to_none=True)
    policy.evaluate(history, privileged, is_teacher).square().mean().backward()
    assert has_gradient(policy.critic)
    assert not has_gradient(policy.actor)
    assert not has_gradient(policy.teacher_encoder)
    assert not has_gradient(policy.student_moe_encoder)


def test_residual_export_matches_actions_diagnostics_and_metadata(tmp_path):
    config = policy_config(True)
    policy = ActorCriticMoECTS(**config).eval()
    with torch.no_grad():
        policy.actor.residual.network[-1].weight.normal_(std=0.1)
        policy.actor.residual.network[-1].bias.fill_(0.2)
    path = tmp_path / "residual.pt"
    CTSStudentExporter(policy, default_observation_config(4)).export(path)
    exported = torch.jit.load(str(path))
    history = torch.randn(4, 228)
    expected = policy.act_inference_with_diagnostics(history)
    actual = exported.forward_with_diagnostics(history)
    for want, have in zip(expected, actual):
        torch.testing.assert_close(have, want)
    torch.testing.assert_close(exported(history), expected[0])
    assert exported.get_residual_enabled()
    torch.testing.assert_close(exported.get_residual_action_scales(), torch.tensor(SCALES))
    assert all("critic" not in name and "teacher" not in name for name in exported.state_dict())
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["residual_enabled"] is True
    assert metadata["residual_action_scales"] == SCALES
    assert metadata["residual_hidden_dims"] == [16, 8]
    exported_config = json.loads(exported.get_residual_config_json())
    assert exported_config["residual_action_scales"] == SCALES


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_and_residual_checkpoints_strictly_roundtrip(enabled):
    config = policy_config(enabled)
    policy = ActorCriticMoECTS(**config)
    checkpoint = make_checkpoint(policy, config)
    if not enabled:
        assert not any(name.startswith("residual_") for name in checkpoint["policy_config"])
    restored = policy_from_checkpoint(checkpoint)
    assert restored.residual_enabled == enabled
    assert policy.state_dict().keys() == restored.state_dict().keys()
    history = torch.randn(4, 228)
    assert torch.equal(policy.act_inference(history), restored.act_inference(history))


def test_checkpoint_rejects_residual_weight_and_scale_metadata_mismatches():
    config = policy_config(True)
    policy = ActorCriticMoECTS(**config)
    checkpoint = make_checkpoint(policy, config)
    checkpoint["policy_config"]["residual_action_scales"][3] = 0.2
    with pytest.raises(ValueError, match="scales disagree"):
        validate_cts_checkpoint(checkpoint)
    checkpoint = make_checkpoint(policy, config)
    checkpoint["policy_config"].update(residual_enabled=False, residual_action_scales=None)
    with pytest.raises(ValueError, match="residual configuration"):
        validate_cts_checkpoint(checkpoint)


@pytest.mark.parametrize("scales", [None, [0.1], [0.0] * 16, [-0.1] * 16,
                                   [float("nan")] * 16, [float("inf")] * 16])
def test_invalid_enabled_residual_scales_fail_before_training(scales):
    config = policy_config(True)
    config["residual_action_scales"] = scales
    with pytest.raises(ValueError):
        normalize_policy_config(config)
    with pytest.raises(ValueError):
        ActorCriticMoECTS(**config)


def runtime_from_export(path, expected_policy=None, policy=None):
    return NezhaCTSPolicyRuntime(
        str(path), default_dof_pos=[0.] * 16,
        p_gains=[10.] * 16, d_gains=[2.] * 16,
        torque_limits=[100.] * 16, expected_policy=expected_policy, policy=policy,
    )


def test_g9_runtime_sensor_sequence_and_residual_structure_validation(tmp_path):
    from experiments.g9_all_moe_residual.config import POLICY

    config = policy_config()
    config.update(copy.deepcopy(POLICY))
    policy = ActorCriticMoECTS(**config).eval()
    with torch.no_grad():
        policy.actor.residual.network[-1].bias.fill_(0.4)
    schema = default_observation_config(4)
    path = tmp_path / "g9.pt"
    CTSStudentExporter(policy, schema).export(path)
    runtime = runtime_from_export(path, expected_policy=POLICY)
    assert runtime.policy_config["residual_action_scales"] == SCALES
    assert runtime.policy_config["residual_hidden_dims"] == [128, 64]
    history = CTSObservationHistory(1, schema)
    previous = torch.zeros(1, 16)
    for step in range(6):
        if step == 3:
            runtime.reset()
            history.reset()
            previous.zero_()
        raw = dict(
            base_ang_vel=torch.randn(1, 3),
            projected_gravity=torch.tensor([[0., 0., -1.]]),
            velocity_commands=torch.tensor([[0.8, 0., 0.]]),
            joint_pos=torch.randn(1, 16), joint_vel=torch.randn(1, 16),
            actions=previous.clone(),
        )
        observation = history.update(build_proprioception(raw, schema, add_noise=False))
        expected = policy.act_inference(observation).squeeze(0).clamp(-100, 100)
        action, _ = runtime.step(raw["base_ang_vel"][0], raw["projected_gravity"][0],
                                 raw["velocity_commands"][0], raw["joint_pos"][0],
                                 raw["joint_vel"][0])
        torch.testing.assert_close(action, expected)
        previous.copy_(action.unsqueeze(0))

    wrong = copy.deepcopy(POLICY)
    wrong.update(residual_enabled=False, residual_action_scales=None)
    with pytest.raises(ValueError, match="residual_enabled"):
        runtime_from_export(path, expected_policy=wrong)
    wrong = copy.deepcopy(POLICY)
    wrong["residual_action_scales"][3] = 0.2
    with pytest.raises(ValueError, match="residual_action_scales"):
        runtime_from_export(path, expected_policy=wrong)
    wrong = copy.deepcopy(POLICY)
    wrong["residual_hidden_dims"] = [128, 32]
    with pytest.raises(ValueError, match="residual_hidden_dims"):
        runtime_from_export(path, expected_policy=wrong)


def test_runtime_rejects_export_residual_metadata_disagreement(tmp_path):
    policy = ActorCriticMoECTS(**policy_config(True))
    exporter = CTSStudentExporter(policy, default_observation_config(4))
    metadata = json.loads(exporter.residual_config_json)
    metadata["residual_action_scales"][0] = 0.2
    exporter.residual_config_json = json.dumps(metadata)
    with pytest.raises(ValueError, match="scales disagree"):
        runtime_from_export(tmp_path / "unused.pt", policy=exporter)
    metadata.update(residual_enabled=False, residual_action_scales=None)
    exporter.residual_config_json = json.dumps(metadata)
    with pytest.raises(ValueError, match="flag disagrees"):
        runtime_from_export(tmp_path / "unused.pt", policy=exporter)


def test_runtime_defaults_legacy_export_to_residual_disabled(tmp_path):
    class LegacyInterface:
        """Present only the interfaces available before residual exports."""
        def __init__(self, exported):
            self.exported = exported

        def to(self, device):
            self.exported.to(device)
            return self

        def eval(self):
            self.exported.eval()
            return self

        def __getattr__(self, name):
            if name.startswith("get_residual_"):
                raise AttributeError(name)
            return getattr(self.exported, name)

        def __call__(self, observations):
            return self.exported(observations)

    policy = ActorCriticMoECTS(**policy_config())
    legacy = LegacyInterface(CTSStudentExporter(policy, default_observation_config(4)))
    runtime = runtime_from_export(tmp_path / "unused.pt", policy=legacy,
                                  expected_policy=policy_config())
    assert runtime.policy_config["residual_enabled"] is False
    assert runtime.policy_config["residual_action_scales"] is None
