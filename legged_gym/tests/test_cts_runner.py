"""Exercise the real CTS runner without loading an Isaac Gym simulator."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint, policy_from_checkpoint
from rsl_rl.utils.cts_observations import default_observation_config
from rsl_rl.utils.wf_curriculum import WFCurriculum, default_wf_command_config


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


class CurriculumVectorEnv(FakeVectorEnv):
    """Use the real range curriculum while keeping simulator state minimal."""

    def __init__(self, range_profile="nezha_reward"):
        super().__init__()
        self.device = "cpu"
        self.dt = 0.02
        self.command_ranges = {
            "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.6, 0.6],
            "ang_vel_yaw": [-1.0, 1.0], "heading": [-3.14, 3.14],
        }
        settings = default_wf_command_config()
        settings["range_curriculum"] = range_profile
        self.cfg = SimpleNamespace(
            commands=SimpleNamespace(
                curriculum=True, wf_settings=settings,
                ranges=SimpleNamespace(**copy.deepcopy(self.command_ranges)),
            ),
            terrain=SimpleNamespace(
                mesh_type="trimesh", terrain_length=9.0,
                sub_terrain_border_width=0.5,
            ),
        )
        self.terrain = SimpleNamespace(column_types=["flat", "stairs_up"])
        self.terrain_types = torch.arange(self.num_envs) % 2
        self.wf_curriculum = WFCurriculum(self)
        self.ranges_at_reset = []

    def reset(self):
        self.wf_curriculum.update_command_ranges()
        self.ranges_at_reset.append((
            self.common_step_counter, copy.deepcopy(self.command_ranges),
            self.wf_curriculum.env_command_ranges["lin_vel_x"].clone(),
        ))
        return super().reset()


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


def test_reward_command_range_checkpoint_restores_expansion_before_reset(tmp_path):
    env = CurriculumVectorEnv()
    runner = MoECTSOnPolicyRunner(env, runner_config())
    env.command_ranges["lin_vel_x"] = [-1.4, 1.4]
    env.command_ranges["ang_vel_yaw"] = [-1.3, 1.3]
    env.wf_curriculum.update_command_ranges()
    env.common_step_counter = 5800 * runner.num_steps_per_env
    expanded = copy.deepcopy(env.command_ranges)
    path = tmp_path / "expanded.pt"
    runner.save(path)
    checkpoint = load_cts_checkpoint(path)
    assert checkpoint["command_curriculum_state"] == {
        "profile": "nezha_reward", "ranges": expanded,
    }
    # Saving runtime state must not rewrite the documented initial settings.
    assert checkpoint["environment_config"]["commands"]["ranges"]["lin_vel_x"] == [-1.0, 1.0]

    resumed = MoECTSOnPolicyRunner(CurriculumVectorEnv(), runner_config())
    resumed.load(path)
    clock, ranges, per_terrain = resumed.env.ranges_at_reset[-1]
    assert clock == env.common_step_counter
    assert ranges == expanded
    assert resumed.env.command_ranges == expanded
    assert torch.allclose(per_terrain[::2], torch.tensor([[-1.4, 1.4]]).expand(4, -1))
    # WF still caps the effective speed on stairs after a global expansion.
    assert torch.equal(per_terrain[1::2], torch.tensor([[-1.0, 1.0]]).expand(4, -1))


def test_reward_command_range_load_accepts_older_checkpoint_without_state(tmp_path):
    runner = MoECTSOnPolicyRunner(CurriculumVectorEnv(), runner_config())
    path = tmp_path / "older.pt"
    runner.save(path)
    checkpoint = load_cts_checkpoint(path)
    checkpoint.pop("command_curriculum_state")
    checkpoint["env_common_step_counter"] = 100000 * runner.num_steps_per_env
    torch.save(checkpoint, path)

    resumed = MoECTSOnPolicyRunner(CurriculumVectorEnv(), runner_config())
    initial = copy.deepcopy(resumed.env.command_ranges)
    resumed.load(path)
    assert resumed.env.command_ranges == initial
    assert resumed.env.ranges_at_reset[-1][1] == initial


def test_iteration_command_curriculum_does_not_save_or_load_reward_ranges(tmp_path):
    runner = MoECTSOnPolicyRunner(CurriculumVectorEnv("iteration"), runner_config())
    path = tmp_path / "iteration.pt"
    runner.save(path)
    checkpoint = load_cts_checkpoint(path)
    assert "command_curriculum_state" not in checkpoint
    checkpoint["command_curriculum_state"] = {
        "profile": "nezha_reward",
        "ranges": {"lin_vel_x": [-1.4, 1.4], "ang_vel_yaw": [-1.3, 1.3]},
    }
    checkpoint["env_common_step_counter"] = 75000 * runner.num_steps_per_env
    torch.save(checkpoint, path)

    runner.load(path)
    assert runner.env.command_ranges["lin_vel_x"] == [-3.0, 3.0]
    assert runner.env.command_ranges["ang_vel_yaw"] == [-3.14, 3.14]


def test_tensorboard_starts_before_rollout_and_covers_repeated_learn_calls(tmp_path, monkeypatch):
    env = FakeVectorEnv()
    env.max_episode_length = 100
    env.cfg = SimpleNamespace(terrain=SimpleNamespace(generator_profile="legbot_wf"))
    runner = MoECTSOnPolicyRunner(env, runner_config(), log_dir=str(tmp_path))
    original_step = env.step
    observed_writers = []

    def check_live_logs(actions):
        if env.common_step_counter == 0:
            assert runner.writer is not None
            observed_writers.append(runner.writer)
            assert list(tmp_path.glob("events.out.tfevents.*"))
            # Creating the run must not manufacture loss/reward samples.
            assert not EventAccumulator(str(tmp_path)).Reload().Tags()["scalars"]
        elif env.common_step_counter == 3:
            events = EventAccumulator(str(tmp_path)).Reload()
            assert [event.step for event in events.Scalars("Loss/value")] == [0]
            assert "Train/teacher_reward" not in events.Tags()["scalars"]
        return original_step(actions)

    monkeypatch.setattr(env, "step", check_live_logs)
    runner.learn(2)
    assert runner.writer is None
    assert observed_writers[0].file_writer is None
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [event.step for event in events.Scalars("Loss/value")] == [0, 1]
    assert [event.value for event in events.Scalars("Perf/total_timesteps")] == [24, 48]
    with (tmp_path / "config.json").open() as stream:
        assert json.load(stream)["environment"]["terrain"]["generator_profile"] == "legbot_wf"

    runner.learn(1)
    assert runner.writer is None
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [event.step for event in events.Scalars("Loss/value")] == [0, 1, 2]
    assert [event.value for event in events.Scalars("Perf/total_timesteps")] == [24, 48, 72]
    assert (tmp_path / "model_3.pt").is_file()


@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt])
def test_tensorboard_persists_completed_iterations_on_interruption(tmp_path, monkeypatch, interruption):
    env = FakeVectorEnv()
    runner = MoECTSOnPolicyRunner(env, runner_config(), log_dir=str(tmp_path))
    original_step = env.step
    observed_writers = []

    def interrupted_step(actions):
        if env.common_step_counter == 0:
            observed_writers.append(runner.writer)
        if env.common_step_counter == 6:
            raise interruption("interrupted during third rollout")
        return original_step(actions)

    monkeypatch.setattr(env, "step", interrupted_step)
    with pytest.raises(interruption, match="interrupted during third rollout"):
        runner.learn(3)
    assert runner.writer is None
    assert observed_writers[0].file_writer is None
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [event.step for event in events.Scalars("Loss/value")] == [0, 1]
    assert [event.value for event in events.Scalars("Perf/total_timesteps")] == [24, 48]
    assert (tmp_path / "model_2.pt").is_file()
    assert not (tmp_path / "model_3.pt").exists()


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
