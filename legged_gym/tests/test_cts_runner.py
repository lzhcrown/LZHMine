"""Exercise the real CTS runner without loading an Isaac Gym simulator."""

import copy

import pytest
import torch

from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint, policy_from_checkpoint
from rsl_rl.utils.cts_observations import default_observation_config


class FakeVectorEnv:
    num_envs = 8
    num_obs = 228
    num_privileged_obs = 295
    num_one_step_obs = 57
    num_actions = 16
    max_episode_length = 3

    def __init__(self):
        self.observation_config = default_observation_config()
        self.observations = torch.zeros(self.num_envs, self.num_obs)
        self.privileged = torch.zeros(self.num_envs, self.num_privileged_obs)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0
        self.reset_calls = 0
        self.action_batches = []

    def reset(self):
        self.reset_calls += 1
        self.observations.zero_()
        self.privileged.zero_()
        self.episode_length_buf.zero_()
        return self.observations, self.privileged

    def get_observations(self):
        return self.observations

    def get_privileged_observations(self):
        return self.privileged

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        assert torch.isfinite(actions).all()
        self.action_batches.append(actions.clone())
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.observations.add_(0.01)
        self.observations[:, -16:] = actions
        self.privileged[:, :57] = self.observations[:, -57:]
        rewards = 1 - 0.05 * actions.square().mean(dim=-1)
        dones = self.episode_length_buf >= self.max_episode_length
        ids = dones.nonzero(as_tuple=False).flatten()
        # This sentinel would destroy the learning inputs if the runner used
        # terminal observations as the next episode's current observations.
        terminal_privileged = torch.full((len(ids), self.num_privileged_obs), 1.0e12)
        infos = {"time_outs": dones.clone()}
        if ids.numel():
            infos["episode"] = {"rew_tracking": rewards[ids].mean()}
            self.observations[ids] = 0
            self.privileged[ids] = 0
            self.episode_length_buf[ids] = 0
        # Match Nezha's seven-result Gym step API.
        return (self.observations, self.privileged, rewards, dones, infos,
                ids, terminal_privileged)


def runner_config():
    return {
        "policy": {
            "actor_hidden_dims": [16], "critic_hidden_dims": [16],
            "teacher_encoder_hidden_dims": [16],
            "student_encoder_hidden_dims": [16, 8],
            "latent_dim": 8, "expert_num": 8,
        },
        "algorithm": {
            "num_learning_epochs": 1, "num_mini_batches": 2,
            "learning_rate": 3.0e-4, "student_encoder_learning_rate": 7.0e-4,
            "schedule": "fixed",
        },
        "runner": {
            "policy_class_name": "ActorCriticMoECTS", "algorithm_class_name": "MoECTS",
            "num_steps_per_env": 3, "save_interval": 1, "wandb_enabled": False,
        },
    }


def assert_nested_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual):
            assert_nested_equal(expected_item, actual_item)
    else:
        assert expected == actual


def test_runner_learns_saves_and_resumes_both_optimizers(tmp_path):
    torch.manual_seed(31)
    env = FakeVectorEnv()
    runner = MoECTSOnPolicyRunner(env, runner_config(), log_dir=str(tmp_path))
    before = {name: parameter.detach().clone()
              for name, parameter in runner.alg.actor_critic.named_parameters()}
    try:
        runner.learn(2)
    finally:
        if runner.writer is not None:
            runner.writer.close()
    assert runner.current_learning_iteration == 2
    assert runner.tot_timesteps == 48
    assert len(env.action_batches) == 6
    assert (tmp_path / "model_1.pt").is_file()
    assert (tmp_path / "model_2.pt").is_file()
    assert (tmp_path / "config.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))
    changed = {name for name, parameter in runner.alg.actor_critic.named_parameters()
               if not torch.equal(before[name], parameter)}
    assert any(name.startswith("teacher_encoder.") for name in changed)
    assert any(name.startswith("student_moe_encoder.") for name in changed)
    assert any(name.startswith("actor.") for name in changed)
    assert any(name.startswith("critic.") for name in changed)

    checkpoint = load_cts_checkpoint(tmp_path / "model_2.pt")
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["student_optimizer_state_dict"]["state"]
    assert checkpoint["next_iter"] == 2
    restored = MoECTSOnPolicyRunner(FakeVectorEnv(), runner_config())
    restored.load(tmp_path / "model_2.pt")
    assert restored.current_learning_iteration == 2
    assert restored.tot_timesteps == 48
    assert restored.env.common_step_counter == env.common_step_counter
    assert restored.env.reset_calls == 2
    assert_nested_equal(runner.alg.actor_critic.state_dict(), restored.alg.actor_critic.state_dict())
    assert_nested_equal(runner.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
    assert_nested_equal(runner.alg.optimizer_stu_enc.state_dict(), restored.alg.optimizer_stu_enc.state_dict())
    assert restored.alg.learning_rate == 3.0e-4

    policy = policy_from_checkpoint(checkpoint)
    observations = torch.randn(3, 228)
    with torch.no_grad():
        expected = runner.get_inference_policy()(observations)
        assert torch.allclose(restored.get_inference_policy()(observations), expected)
        assert torch.allclose(policy.act_inference(observations), expected)
    restored.learn(1)
    assert restored.current_learning_iteration == 3
    assert restored.tot_timesteps == 72
    # Restoring Adam's step matters: both optimizers must continue at step 6.
    for optimizer in (restored.alg.optimizer, restored.alg.optimizer_stu_enc):
        assert all(state["step"].item() == 6 for state in optimizer.state.values())


def test_checkpoint_rejects_other_architectures_and_inconsistent_dimensions(tmp_path):
    runner = MoECTSOnPolicyRunner(FakeVectorEnv(), runner_config())
    original = tmp_path / "valid.pt"
    runner.save(original)
    valid = load_cts_checkpoint(original)
    old_path = tmp_path / "old_architecture.pt"
    torch.save({"model_state_dict": valid["model_state_dict"], "iter": 4}, old_path)
    with pytest.raises(ValueError, match="other architectures"):
        runner.load(old_path)
    for field, value in (("num_actor_obs", 229), ("history_length", 3),
                         ("num_critic_obs", 296), ("observation_dims", [57])):
        changed = copy.deepcopy(valid)
        changed["policy_config"][field] = value
        path = tmp_path / (field + ".pt")
        torch.save(changed, path)
        with pytest.raises(ValueError, match="schema disagrees"):
            load_cts_checkpoint(path)


def test_checkpoint_rejects_unsupported_schema_and_changed_preprocessing(tmp_path):
    runner = MoECTSOnPolicyRunner(FakeVectorEnv(), runner_config())
    valid_path = tmp_path / "valid.pt"
    runner.save(valid_path)
    valid = load_cts_checkpoint(valid_path)
    changed = copy.deepcopy(valid)
    changed["observation_config"]["schema_version"] = 2
    path = tmp_path / "unsupported_schema.pt"
    torch.save(changed, path)
    with pytest.raises(ValueError, match="schema version"):
        load_cts_checkpoint(path)

    # Width alone cannot detect changes to scale, noise or sensor semantics.
    changed = copy.deepcopy(valid)
    changed["observation_config"]["proprioceptive_terms"][0]["scale"] = 1.0
    path = tmp_path / "changed_preprocessing.pt"
    torch.save(changed, path)
    with pytest.raises(ValueError, match="Observation configuration differs"):
        runner.load(path)
