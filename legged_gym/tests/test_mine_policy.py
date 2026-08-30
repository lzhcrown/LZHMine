import torch

from deploy.nezha.policy_runtime import NezhaObservationHistory
from rsl_rl.algorithms import MINEPPO
from rsl_rl.modules import MINEActorCritic, MINEPolicyExporter


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
