"""CPU checks for CTS data alignment, gradient ownership and update order."""

import pytest
import torch

from rsl_rl.algorithms.moe_cts import MoECTS
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS


def make_policy(actor_type="mlp"):
    return ActorCriticMoECTS(
        num_actor_obs=24, num_critic_obs=11, num_one_step_obs=6,
        num_actions=3, history_length=4, observation_dims=[2, 1, 3],
        actor_hidden_dims=[16], critic_hidden_dims=[16],
        teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16, 8],
        latent_dim=8, expert_num=4,
        actor_type=actor_type, actor_expert_num=8,
    )


def has_gradient(module):
    return any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in module.parameters())


def fill_rollout(algorithm, steps=3):
    algorithm.init_storage(algorithm.num_envs, steps, [24], [11], [3])
    for step in range(steps):
        history = torch.randn(algorithm.num_envs, 24)
        privileged = torch.randn(algorithm.num_envs, 11)
        marker = torch.arange(algorithm.num_envs) + 100 * step
        history[:, 0] = marker
        privileged[:, 0] = marker
        algorithm.act(history, privileged)
        algorithm.process_env_step(torch.randn(algorithm.num_envs),
                                   torch.zeros(algorithm.num_envs, dtype=torch.bool), {})
    algorithm.compute_returns(torch.randn(algorithm.num_envs, 24),
                              torch.randn(algorithm.num_envs, 11))


def test_wf_policy_shapes_and_term_major_current_frame():
    policy = ActorCriticMoECTS(228, 295, 57, 16)
    history = torch.arange(228, dtype=torch.float32).reshape(1, -1).repeat(2, 1)
    current = policy.get_current_observations(history)
    expected = torch.cat((history[:, 9:12], history[:, 21:24], history[:, 33:36],
                          history[:, 84:100], history[:, 148:164], history[:, 212:228]), dim=-1)
    assert torch.equal(current, expected)
    latent, weights = policy.student_moe_encoder(history)
    assert latent.shape == (2, 32)
    assert torch.allclose(latent.norm(dim=-1), torch.ones(2), atol=1.0e-6)
    assert weights.shape == (2, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1.0e-6)
    assert policy.act_inference(history).shape == (2, 16)
    assert policy.evaluate(history, torch.randn(2, 295), is_teacher=True).shape == (2, 1)
    assert policy.actor.network[0].in_features == 89
    assert policy.critic.network[0].in_features == 327


@pytest.mark.parametrize("actor_type", ["mlp", "moe"])
def test_gradient_boundaries_and_positive_action_scale(actor_type):
    policy = make_policy(actor_type)
    obs, privileged = torch.randn(5, 24), torch.randn(5, 11)
    policy.act(obs, privileged, is_teacher=True)
    policy.action_mean.square().mean().backward()
    assert has_gradient(policy.teacher_encoder)
    assert has_gradient(policy.actor)
    assert not has_gradient(policy.student_moe_encoder)

    policy.zero_grad(set_to_none=True)
    policy.act(obs, privileged, is_teacher=False)
    policy.action_mean.square().mean().backward()
    assert has_gradient(policy.actor)
    assert not has_gradient(policy.teacher_encoder)
    assert not has_gradient(policy.student_moe_encoder)

    for is_teacher in (True, False):
        policy.zero_grad(set_to_none=True)
        policy.evaluate(obs, privileged, is_teacher=is_teacher).square().mean().backward()
        assert has_gradient(policy.critic)
        assert not has_gradient(policy.actor)
        assert not has_gradient(policy.teacher_encoder)
        assert not has_gradient(policy.student_moe_encoder)

    with torch.no_grad():
        policy.std.fill_(-0.1)
    policy.act(obs)
    assert torch.all(policy.action_std > 0)
    assert torch.isfinite(policy.get_actions_log_prob(policy.action_mean)).all()


def test_environment_order_and_timeout_are_kept_together():
    torch.manual_seed(13)
    algorithm = MoECTS(make_policy(), 10, num_learning_epochs=1, num_mini_batches=1)
    algorithm.init_storage(10, 1, [24], [11], [3])
    assert algorithm.student_env_idxs.tolist() == [0, 3, 6]
    assert sorted(algorithm.env_order.tolist()) == list(range(10))
    observations = torch.arange(10, dtype=torch.float32).reshape(-1, 1).repeat(1, 24)
    privileged = observations[:, :11] * 10
    actions = algorithm.act(observations, privileged)
    expected_actions = algorithm.transition.actions.clone()
    old_values = algorithm.transition.values.squeeze(-1).clone()
    assert torch.equal(actions[algorithm.env_order], expected_actions)
    assert torch.equal(algorithm.transition.observations, observations[algorithm.env_order])
    assert torch.equal(algorithm.transition.critic_observations, privileged[algorithm.env_order])
    # Simulators can reuse/mutate input buffers during step; stored input must
    # still be the state which produced this action.
    saved_obs = observations[algorithm.env_order].clone()
    observations.fill_(-1)
    rewards = torch.arange(10, dtype=torch.float32)
    dones = torch.arange(10) % 2 == 0
    timeouts = torch.arange(10) % 3 == 0
    algorithm.process_env_step(rewards, dones, {"time_outs": timeouts})
    storage = algorithm.storage
    expected_rewards = rewards[algorithm.env_order] + algorithm.gamma * old_values * timeouts[algorithm.env_order]
    assert torch.equal(storage.observations[0], saved_obs)
    assert torch.equal(storage.actions[0], expected_actions)
    assert torch.equal(storage.dones[0, :, 0], dones[algorithm.env_order])
    assert torch.allclose(storage.rewards[0, :, 0], expected_rewards)


def test_minibatches_cover_uneven_groups_and_preserve_same_time_pairs():
    # 21 teacher / 9 student samples split into two batches have remainders.
    storage = RolloutStorageCTS(10, 7, 3, [1], [1], [1])
    for step in range(3):
        storage.observations[step, :, 0] = torch.arange(10) + 100 * step
    storage.privileged_observations.copy_(storage.observations + 1000)
    storage.actions.copy_(storage.observations + 2000)
    storage.step = 3
    batches = list(storage.mini_batch_generator(2, 2))
    assert len(batches) == 4
    for epoch in range(2):
        seen = []
        for batch in batches[epoch * 2:(epoch + 1) * 2]:
            teacher_count = batch["teacher_count"]
            ids = batch["observations"].squeeze(-1)
            assert (ids[:teacher_count] % 100 < 7).all()
            assert (ids[teacher_count:] % 100 >= 7).all()
            assert torch.equal(batch["critic_observations"], batch["observations"] + 1000)
            assert torch.equal(batch["actions"], batch["observations"] + 2000)
            seen.extend(ids.tolist())
        assert sorted(seen) == [100 * step + env for step in range(3) for env in range(10)]
    student_seen = []
    for obs, privileged in storage.student_mini_batch_generator(2):
        assert torch.equal(privileged, obs + 1000)
        student_seen.extend(obs.squeeze(-1).tolist())
    assert sorted(student_seen) == [100 * step + env for step in range(3) for env in (7, 8, 9)]
    with pytest.raises(ValueError, match="both groups"):
        list(storage.mini_batch_generator(10))


def test_return_computation_stops_at_episode_boundary():
    storage = RolloutStorageCTS(2, 1, 2, [1], [1], [1])
    storage.step = 2
    storage.rewards.fill_(1)
    storage.values.zero_()
    storage.dones[0, 0] = True
    storage.compute_returns(torch.full((2, 1), 10.0), gamma=0.9, lam=1.0,
                            normalize_advantage=False)
    assert torch.allclose(storage.returns[:, :, 0], torch.tensor([[1.0, 10.0], [10.0, 10.0]]))


@pytest.mark.parametrize("actor_type", ["mlp", "moe"])
def test_update_finishes_ppo_before_student_distillation(monkeypatch, actor_type):
    torch.manual_seed(7)
    policy = make_policy(actor_type)
    algorithm = MoECTS(policy, 6, num_learning_epochs=2, num_mini_batches=2, schedule="fixed")
    ppo_ids = {id(p) for group in algorithm.optimizer.param_groups for p in group["params"]}
    student_ids = {id(p) for group in algorithm.optimizer_stu_enc.param_groups for p in group["params"]}
    assert not ppo_ids.intersection(student_ids)
    assert ppo_ids.union(student_ids) == {id(p) for p in policy.parameters()}
    fill_rollout(algorithm)
    before_teacher = [p.detach().clone() for p in policy.teacher_encoder.parameters()]
    before_student = [p.detach().clone() for p in policy.student_moe_encoder.parameters()]
    before_actor = {name: p.detach().clone() for name, p in policy.actor.named_parameters()}
    actor_after_ppo = {}
    phase_steps = []
    original_ppo_step = algorithm.optimizer.step
    original_student_step = algorithm.optimizer_stu_enc.step
    student_markers, target_markers = [], []

    def student_inputs(module, inputs):
        if phase_steps.count("ppo") == 4:
            student_markers.append(inputs[0][:, 0].clone())

    def teacher_inputs(module, inputs):
        if phase_steps.count("ppo") == 4:
            target_markers.append(inputs[0][:, 0].clone())

    def ppo_step(*args, **kwargs):
        phase_steps.append("ppo")
        assert all(p.grad is None for p in policy.student_moe_encoder.parameters())
        assert all(torch.equal(old, new) for old, new in zip(before_student, policy.student_moe_encoder.parameters()))
        return original_ppo_step(*args, **kwargs)

    def student_step(*args, **kwargs):
        phase_steps.append("student")
        assert phase_steps.count("ppo") == 4
        assert all(p.grad is None for p in algorithm.ppo_parameters)
        if not actor_after_ppo:
            actor_after_ppo.update({name: p.detach().clone()
                                    for name, p in policy.actor.named_parameters()})
        assert all(torch.equal(actor_after_ppo[name], p)
                   for name, p in policy.actor.named_parameters())
        return original_student_step(*args, **kwargs)

    monkeypatch.setattr(algorithm.optimizer, "step", ppo_step)
    monkeypatch.setattr(algorithm.optimizer_stu_enc, "step", student_step)
    handles = [policy.student_moe_encoder.register_forward_pre_hook(student_inputs),
               policy.teacher_encoder.register_forward_pre_hook(teacher_inputs)]
    try:
        losses = algorithm.update()
    finally:
        for handle in handles:
            handle.remove()
    assert phase_steps == ["ppo"] * 4 + ["student"] * 4
    assert len(student_markers) == len(target_markers) == 4
    for history_markers, privileged_markers in zip(student_markers, target_markers):
        assert torch.equal(history_markers, privileged_markers)
        assert set((history_markers % 100).tolist()).issubset(set(algorithm.student_env_idxs.tolist()))
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert abs(sum(losses["expert_%d_usage" % i] for i in range(4)) - 1) < 1.0e-6
    assert losses["surrogate"] == pytest.approx(losses["teacher_surrogate"] + losses["student_surrogate"])
    assert any(not torch.equal(old, new) for old, new in zip(before_teacher, policy.teacher_encoder.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(before_student, policy.student_moe_encoder.parameters()))
    assert any(not torch.equal(before_actor[name], p) for name, p in policy.actor.named_parameters())
    if actor_type == "moe":
        assert any(not torch.equal(before_actor[name], p)
                   for name, p in policy.actor.named_parameters() if "gating_network" in name)
        # Grouped Conv1d stores one contiguous action-head block per expert.
        head_name = "moe.experts.experts.weight"
        old_heads = before_actor[head_name].reshape(8, policy.num_actions, -1)
        new_heads = dict(policy.actor.named_parameters())[head_name].detach().reshape_as(old_heads)
        assert torch.all((old_heads != new_heads).flatten(1).any(dim=1))
        assert not hasattr(policy.critic, "moe")
    assert algorithm.storage.step == 0


def test_small_environment_counts_and_incomplete_rollouts_fail_clearly():
    with pytest.raises(ValueError, match="at least two"):
        MoECTS(make_policy(), 1)
    algorithm = MoECTS(make_policy(), 2, num_mini_batches=4)
    with pytest.raises(ValueError, match="reduce num_mini_batches"):
        algorithm.init_storage(2, 1, [24], [11], [3])
    algorithm.init_storage(2, 4, [24], [11], [3])
    with pytest.raises(RuntimeError, match="complete CTS rollout"):
        algorithm.compute_returns(torch.randn(2, 24), torch.randn(2, 11))
