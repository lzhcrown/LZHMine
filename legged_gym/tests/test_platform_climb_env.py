"""G13 control-step rewards, terminal flags and reset on CPU simulator tensors."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from legged_gym.terrains.platform_rewards import platform_climb_settings
from legged_gym.terrains.wf import high_platform_wf_sub_terrains
from test_platform_rewards import robot


@pytest.fixture
def env(robot, monkeypatch):
    # Keep the real Nezha termination, reward dispatch and terrain move masks.
    # Replace the parent's simulator allocation and native reset only.
    class CpuNezha(type(robot)):
        def _init_buffers(self):
            pass

        def reset_idx(self, ids):
            self.spawn_yaws = self._sample_spawn_yaws(ids)
            self.episode_length_buf[ids] = 0

    parent = ModuleType("legged_gym.envs.nezha.nezha_moe_cts_env")
    parent.NezhaMoECTSEnv = CpuNezha
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    path = Path(__file__).parents[1] / "envs/nezha/platform_climb_env.py"
    spec = importlib.util.spec_from_file_location("cpu_platform_climb_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.NezhaPlatformClimbEnv
    env = cls.__new__(cls)
    env.__dict__.update(robot.__dict__)
    env.cfg.env = SimpleNamespace(fail_to_terminal_time_s=.6)
    env.cfg.terrain = SimpleNamespace(wf_sub_terrains=high_platform_wf_sub_terrains(),
                                      terrain_length=9., terrain_width=9., yaw_init_range=3.14, curriculum=True)
    env.cfg.rewards.platform_climb = True
    env.cfg.rewards.platform_climb_settings = platform_climb_settings()
    # Slots are ordinary, pit and block. Fixed RNG places the pit on +Y.
    globals_ = type(robot)._sample_spawn_yaws.__globals__
    monkeypatch.setitem(globals_, "torch_rand_float", lambda lo, hi, size, device: torch.full(size, 1.4))
    env.block_tiles = torch.as_tensor(env.terrain.tile_types == "platform_block")
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    env.reset_buf = torch.zeros_like(env.episode_length_buf)
    env.time_out_buf = torch.zeros(3, dtype=torch.bool)
    env.fail_buf = torch.zeros_like(env.episode_length_buf)
    env.rew_buf = torch.zeros(3)
    env.root_states = torch.zeros(3, 13)
    centers = torch.stack((env.terrain_levels + .5, env.terrain_types + .5), dim=1) * 9.
    env.root_states[:, :2] = centers
    env.root_states[2, 0] -= 3.
    env.num_bodies = 5
    env.wheel_body_indices = torch.arange(4)
    env.termination_contact_indices = [4]
    env.rigid_body_states = torch.zeros(3, 5, 13)
    env.rigid_body_states[:, :, :2] = env.root_states[:, None, :2]
    env.contact_forces = torch.zeros(3, 5, 3)
    env.projected_gravity = torch.tensor([[0., 0., -1.]] * 3)
    env.commands = torch.tensor([[1., 0., 0.]] * 3)
    env.base_lin_vel = env.commands.clone()
    env.reward_scales = {"tracking_lin_vel": 3., "termination": -.8}
    env.platform_scales = {name: scale * env.dt for name, scale in env.reward_scales.items()}
    env.max_episode_length = 1250
    env.wf_curriculum = SimpleNamespace(
        update_reward_curriculum=lambda: None,
        terrain_move_masks=lambda ids: (torch.ones_like(ids, dtype=torch.bool),
                                       torch.zeros_like(ids, dtype=torch.bool)),
    )
    env.extras = {}
    env._init_buffers()
    env._prepare_reward_function()
    env.reset_idx(torch.arange(3))
    return env


def test_route_at_spawn_and_forward_credit_is_world_aligned_and_single_use(env):
    torch.testing.assert_close(env.climb_axis[1:], torch.tensor([[0., 1.], [1., 0.]]))
    assert env.spawn_yaws[:, 0].tolist() == pytest.approx([1.4, torch.pi / 2, 0.])
    assert env.best_forward[2] == -3.
    # The body reports perfect forward tracking in all slots, including the
    # block robot now turned 90 degrees and driving along the face.
    env.root_states[2, 1] += env.dt
    env.root_states[2, 8] = 1.
    env.check_termination()
    assert env._reward_tracking_lin_vel().tolist() == [1., 0., 0.]
    env.root_states[2, 0] += env.dt
    env.root_states[2, 7:9] = torch.tensor([1., 0.])
    env.check_termination()
    assert env._reward_tracking_lin_vel()[2].item() == pytest.approx(1., abs=2e-5)
    for dx in (-env.dt, env.dt):
        env.root_states[2, 0] += dx
        env.check_termination()
        assert env.forward_tracking[2] == 0.


def test_stage_reward_dwell_completion_and_reset_snapshot(env):
    # Land the front pair first. Repeating the same support earns nothing.
    env.rigid_body_states[2, :2, :2] = env.climb_center[2]
    env.contact_forces[2, :4, 2] = 20.
    env.check_termination()
    env.compute_reward()
    assert env.rew_buf[2] == 2.
    env.check_termination()
    env.compute_reward()
    assert env.rew_buf[2] == 0.
    env.rigid_body_states[2, 2:4, :2] = env.climb_center[2]
    env.check_termination()
    env.compute_reward()
    assert env.rew_buf[2] == 2.
    env.contact_forces[2, :2, 2] = 0.  # Break the dwell counter.
    env.check_termination()
    assert env.settled_steps[2] == 0
    env.contact_forces[2, :2, 2] = 20.
    for i in range(env.settle_steps):
        env.check_termination()
        env.compute_reward()
        assert env.rew_buf[2] == (6. if i == env.settle_steps - 1 else 0.)
    assert env.reset_buf[2] and not env.time_out_buf[2] and not env._reward_termination()[2]
    assert env.episode_sums["platform_upper_support"][2] == 4.
    assert env.episode_sums["platform_completion"][2] == 6.
    promote, demote = env._terrain_move_masks(torch.arange(3))
    assert promote.tolist() == [True, False, True] and not demote.any()
    env.reset_idx(torch.tensor([2]))
    result = env.extras["platform_outcomes"]
    assert result["env_ids"].tolist() == [2] and result["is_block"].item()
    assert result["success"].item() and not result["bypass"].item() and result["peak_wheel_fraction"] == 1.
    assert not env.climb_finished.any() and not env.best_support.any()
    env.reset_idx(torch.empty(0, dtype=torch.long))
    assert "platform_outcomes" not in env.extras


@pytest.mark.parametrize("failure", ["left_lane", "timeout", "physical_fall"])
def test_unsuccessful_pit_episode_demotes_with_correct_terminal_reason(env, failure):
    # On +Y approaches, world X is transverse. A simultaneous fall takes
    # precedence over bypass and timeout for the outcome and the bootstrap.
    if failure != "timeout":
        env.root_states[1, 0] += 1.5
    if failure != "left_lane":
        env.episode_length_buf[1] = env.max_episode_length
    if failure == "physical_fall":
        env.projected_gravity[1, 2] = 1.
        env.fail_buf[1] = 31
    env.check_termination()
    assert env.reset_buf[1] and not env.climb_success[1]
    assert env.time_out_buf[1].item() == (failure == "timeout")
    assert env._reward_termination()[1].item() == (failure != "timeout")
    up, down = env._terrain_move_masks(torch.tensor([1, 2, 0]))
    assert up.tolist() == [False, False, True] and down.tolist() == [True, False, False]
    env.reset_idx(torch.tensor([1]))
    outcome = env.extras["platform_outcomes"]
    assert not outcome["is_block"].item() and not outcome["success"].item()
    assert outcome["bypass"].item() == (failure == "left_lane")
