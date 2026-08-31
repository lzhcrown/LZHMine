import torch

from deploy.nezha.policy_runtime import NezhaObservationHistory
from rsl_rl.algorithms import MINEPPO
from rsl_rl.modules import MINEActorCritic, MINEEstimator, MINEPolicyExporter


def make_policy():
    return MINEActorCritic(
        num_actor_obs=20,
        num_critic_obs=24,
        num_one_step_obs=10,
        num_one_step_privileged_obs=12,
        estimation_target_indices=[10, 11],
        num_actions=4,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        enc_hidden_dims=[32, 16],
        tar_hidden_dims=[32, 16],
        latent_dim=6,
        num_modes=3,
    )


def test_mine_forward_update_and_gate_probabilities():
    policy = make_policy()
    observations = torch.randn(16, 20)
    privileged = torch.randn(16, 24)

    actions = policy.act_inference(observations)
    losses = policy.estimator.update(observations, privileged)
    mode_probabilities = policy.estimator.mode_probabilities(observations)

    assert actions.shape == (16, 4)
    assert len(losses) == 13
    assert all(torch.isfinite(torch.tensor(losses)))
    assert mode_probabilities.shape == (16, 3)
    assert torch.allclose(
        mode_probabilities.sum(dim=-1), torch.ones(16), atol=1e-6
    )


def test_scripted_export_matches_online_policy():
    policy = make_policy().eval()
    observations = torch.randn(8, 20)
    exporter = MINEPolicyExporter(policy).eval()
    scripted = torch.jit.script(exporter)

    assert torch.allclose(
        policy.act_inference(observations),
        scripted(observations),
        atol=1e-6,
    )
    assert scripted.get_mode_probabilities(observations).shape == (8, 3)


def test_nezha_observation_history_is_newest_first():
    builder = NezhaObservationHistory([0.0] * 16)
    first = torch.ones(65)
    second = torch.full((65,), 2.0)
    builder.push(first)
    history = builder.push(second).reshape(2, 65)

    assert torch.equal(history[0], second)
    assert torch.equal(history[1], first)


def test_semantic_labels_follow_fixed_wheel_leg_hybrid_indices():
    semantic_cfg = {
        "enabled": True,
        "loss_coef": 0.1,
        "mode_loss_coef": 0.5,
        "command_slice": [6, 9],
        "dof_vel_slice": [21, 37],
        "action_slice": [49, 65],
        "wheel_dof_indices": [3, 7, 11, 15],
        "leg_dof_indices": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        "wheel_activity_low": 0.05,
        "wheel_activity_high": 0.20,
        "leg_activity_low": 0.03,
        "leg_activity_high": 0.12,
        "static_command_threshold": 0.05,
        "leg_action_delta_weight": 0.70,
        "leg_velocity_weight": 0.30,
    }
    estimator = MINEEstimator(
        temporal_steps=2,
        num_one_step_obs=65,
        num_one_step_privileged_obs=375,
        estimation_target_indices=[371, 372, 373, 374],
        enc_hidden_dims=[32],
        tar_hidden_dims=[32],
        latent_dim=8,
        mode_semantic_cfg=semantic_cfg,
    )
    history = torch.zeros(4, 130)
    current = history[:, :65]
    current[:3, 6] = 0.5
    wheel_ids = torch.tensor([3, 7, 11, 15]) + 49
    leg_ids = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]) + 49
    current[0, wheel_ids] = 0.30
    current[1, leg_ids] = 0.30
    current[2, wheel_ids] = 0.30
    current[2, leg_ids] = 0.30

    targets, _, valid, _, _ = estimator._semantic_mode_targets(history)
    assert torch.equal(torch.argmax(targets[:3], dim=-1), torch.tensor([0, 1, 2]))
    assert torch.equal(valid, torch.tensor([True, True, True, False]))


def test_mine_ppo_update_uses_separate_estimator_optimizer():
    policy = make_policy()
    algorithm = MINEPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    algorithm.init_storage(4, 2, [20], [24], [4])
    observations = torch.randn(4, 20)
    privileged = torch.randn(4, 24)
    for _ in range(2):
        algorithm.act(observations, privileged)
        next_privileged = torch.randn(4, 24)
        algorithm.process_env_step(
            torch.randn(4), torch.zeros(4), {}, next_privileged
        )
        observations = torch.randn(4, 20)
        privileged = next_privileged
    algorithm.compute_returns(privileged)
    losses = algorithm.update()

    estimator_ids = {id(parameter) for parameter in policy.estimator.parameters()}
    policy_optimizer_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    assert estimator_ids.isdisjoint(policy_optimizer_ids)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
