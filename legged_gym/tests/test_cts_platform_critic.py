"""Exercise training-only terrain labels through G7 initialization and PPO."""

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint, policy_from_checkpoint
from rsl_rl.utils.cts_export import CTSStudentExporter
from test_cts_initialization import all_moe_config
from test_cts_runner import FakeVectorEnv


def test_task_episode_metrics_use_completed_terrain_before_reset(tmp_path):
    class ResettingTerrainEnv(FakeVectorEnv):
        def __init__(self):
            super().__init__()
            self.high_platform_mask = torch.arange(self.num_envs) % 2 == 0

        def step(self, actions):
            rewards = torch.where(self.high_platform_mask, 4., 1.)
            obs, privileged, _, dones, *rest = super().step(actions)
            self.high_platform_mask[dones] = ~self.high_platform_mask[dones]
            return obs, privileged, rewards, dones, *rest

    cfg = all_moe_config()
    cfg["policy"]["critic_task_count"] = 2
    env = ResettingTerrainEnv()
    runner = MoECTSOnPolicyRunner(env, cfg, log_dir=str(tmp_path))
    # Partial first episodes have only one observed step and must be excluded.
    env.episode_length_buf[:2] = env.max_episode_length - 1
    runner.learn(1)
    events = EventAccumulator(str(tmp_path)).Reload()
    for group, reward in (("ordinary", 3.), ("high_platform", 12.)):
        assert events.Scalars("Train/" + group + "_reward")[-1].value == reward
        assert events.Scalars("Train/" + group + "_length")[-1].value == 3.
    assert not any(tag.startswith("Platform/") for tag in events.Tags()["scalars"])


def test_platform_outcome_rates_count_each_full_episode_once_before_reset(tmp_path):
    class PlatformTrialEnv(FakeVectorEnv):
        def __init__(self):
            super().__init__()
            self.ids = torch.arange(self.num_envs)
            self.high_platform_mask = self.ids < 7
            self.is_block = torch.tensor([True, False, True, False, False, True, True, True])

        def step(self, actions):
            result = super().step(actions)
            dones, infos = result[3:5]
            # Include stale rows and an ordinary environment. Only finished
            # full-length platform trials are legitimate samples this step.
            infos["platform_outcomes"] = {
                "env_ids": self.ids,
                "is_block": self.is_block.clone(),
                "success": (self.ids == 0) | (self.ids == 3) | (self.ids == 7),
                "bypass": (self.ids == 2) | (self.ids == 4),
                "peak_wheel_fraction": torch.tensor([1., .5, 0., 1., .25, .25, .5, 1.]),
            }
            self.is_block[dones] = ~self.is_block[dones]
            return result

    env = PlatformTrialEnv()
    cfg = all_moe_config()
    cfg["policy"]["critic_task_count"] = 2
    cfg["runner"]["num_steps_per_env"] = 2
    runner = MoECTSOnPolicyRunner(env, cfg, log_dir=str(tmp_path))
    env.episode_length_buf[:2] = env.max_episode_length - 1
    runner.learn(2)
    events = EventAccumulator(str(tmp_path)).Reload()
    # In iteration 1, envs 2--6 finish in one batch and 0--1 in another.
    # Blocks: 1,2,5,6. Pits: 0,3,4. Iteration 0 has no full trials.
    for shape, count, successes, bypasses, mean_peak in (
            ("block", 4, 0, 1, 1.25 / 4), ("pit", 3, 2, 1, 2.25 / 3)):
        for name, value in (("episodes", count), ("success_rate", successes / count),
                            ("bypass_rate", bypasses / count), ("mean_peak_wheel_fraction", mean_peak)):
            samples = events.Scalars("Platform/{}/{}".format(shape, name))
            assert [sample.step for sample in samples] == [1]
            assert samples[0].value == pytest.approx(value)


@pytest.mark.parametrize("critic_type", ["moe", "task_mlp"])
def test_platform_labels_follow_pre_step_samples_through_ppo(tmp_path, critic_type):
    env = FakeVectorEnv()
    env.high_platform_mask = torch.arange(env.num_envs) % 3 == 0
    cfg = all_moe_config()
    cfg["policy"].update(critic_type=critic_type, critic_task_count=2,
                         critic_expert_num=8 if critic_type == "moe" else 0)
    cfg["runner"]["num_steps_per_env"] = 1
    runner = MoECTSOnPolicyRunner(env, cfg)
    alg = runner.alg
    policy = alg.actor_critic
    with torch.no_grad():
        if critic_type == "task_mlp":
            for network, value in zip(policy.critic.heads, (1., 5.)):
                network.network[-1].weight.zero_()
                network.network[-1].bias.fill_(value)
        else:
            heads = policy.critic.moe.experts.experts
            heads.weight.zero_()
            heads.bias.view(-1, 2).copy_(torch.tensor([1., 5.]))

    # Half the envs time out; simulate an in-place change of their terrain at reset.
    labels = env.high_platform_mask.clone()
    obs, privileged = env.observations, env.privileged
    alg.act(obs, privileged, env.high_platform_mask)
    timeout = torch.arange(env.num_envs) % 2 == 0
    env.high_platform_mask[timeout] = ~env.high_platform_mask[timeout]
    alg.process_env_step(torch.ones(env.num_envs), timeout, {"time_outs": timeout})
    alg.compute_returns(obs, privileged, env.high_platform_mask)
    ordered = labels[alg.env_order]
    values = torch.where(ordered, 5., 1.)
    torch.testing.assert_close(alg.storage.platform_mask[0], ordered)
    torch.testing.assert_close(alg.storage.values[0, :, 0], values)
    torch.testing.assert_close(alg.storage.returns[0, :, 0], 1 + alg.gamma * values)
    for batch in alg.storage.mini_batch_generator(2, 1):
        new_values = alg._batch_results(batch)[1][:, 0]
        torch.testing.assert_close(new_values, torch.where(batch["platform_mask"], 5., 1.))
    alg.update()

    # A saved, trained two-head checkpoint still deploys without a terrain label.
    path = tmp_path / "platform.pt"
    runner.save(path)
    loaded = policy_from_checkpoint(load_cts_checkpoint(path))
    exported = torch.jit.script(CTSStudentExporter(loaded, env.observation_config))
    with torch.no_grad():
        torch.testing.assert_close(exported(obs), policy.act_inference(obs))


@pytest.mark.parametrize("active_task", [0, 1], ids=["only_ordinary", "only_high_platform"])
def test_task_critics_update_only_the_present_task_after_adam_warmup(active_task, tmp_path):
    torch.manual_seed(19)
    env = FakeVectorEnv()
    env.high_platform_mask = torch.arange(env.num_envs) % 2 == 0
    cfg = all_moe_config()
    cfg["policy"].update(critic_type="task_mlp", critic_expert_num=0, critic_task_count=2)
    cfg["algorithm"]["weight_decay"] = .01
    runner = MoECTSOnPolicyRunner(env, cfg, log_dir=str(tmp_path))
    model = runner.alg.actor_critic
    heads = model.critic.heads

    ordinary, platform = (set(head.parameters()) for head in heads)
    assert ordinary.isdisjoint(platform)
    assert set(model.critic.parameters()) == ordinary | platform
    # Differences in Adam moments are observable only after both heads have been updated.
    runner.learn(1)
    for params in (ordinary, platform):
        assert all(runner.alg.optimizer.state[p]["step"].item() > 0 for p in params)
    before = {p: p.detach().clone() for p in model.critic.parameters()}
    steps = {p: runner.alg.optimizer.state[p]["step"].item() for p in model.critic.parameters()}
    env.high_platform_mask.fill_(bool(active_task))
    runner.learn(1)
    for p in heads[1 - active_task].parameters():
        assert torch.equal(p, before[p])
        assert runner.alg.optimizer.state[p]["step"].item() == steps[p]
    assert any(not torch.equal(p, before[p]) for p in heads[active_task].parameters())
    events = EventAccumulator(str(tmp_path)).Reload()
    for task, tag in enumerate(("Loss/value_ordinary", "Loss/value_high_platform")):
        assert [s.step for s in events.Scalars(tag)] == ([0, 1] if task == active_task else [0])
    assert not any(tag.startswith("CriticExperts/") for tag in events.Tags()["scalars"])


def test_g7_warm_start_initializes_both_value_heads(tmp_path):
    env = FakeVectorEnv()
    source = MoECTSOnPolicyRunner(env, all_moe_config())
    path = tmp_path / "g7.pt"
    source.save(path)
    cfg = all_moe_config()
    cfg["policy"]["critic_task_count"] = 2
    env.high_platform_mask = torch.arange(env.num_envs) % 2 == 0
    target = MoECTSOnPolicyRunner(env, cfg)
    target.initialize_from(path)
    obs = torch.randn_like(env.observations)
    privileged = torch.randn_like(env.privileged)
    with torch.no_grad():
        expected = source.alg.actor_critic.evaluate(obs, privileged)
        for labels in (env.high_platform_mask, ~env.high_platform_mask):
            actual = target.alg.actor_critic.evaluate(obs, privileged, platform_mask=labels)
            torch.testing.assert_close(actual, expected)
    target.learn(1)
    trained = tmp_path / "trained.pt"
    target.save(trained)
    restored = MoECTSOnPolicyRunner(env, cfg)
    restored.load(trained)
    restored.learn(1)
