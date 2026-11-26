"""A fixed student replay must reproduce both platform shapes on old and new WF maps."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from legged_gym.scripts._platform_play import configure_fixed_platform, wf_landing_distance, wf_obstacle_bounds
from legged_gym.terrains.wf import WFTerrain
from test_wf_terrain import terrain_cfg


@pytest.mark.parametrize("shape", ["block", "pit"])
@pytest.mark.parametrize("legacy", [False, True])
def test_fixed_play_builds_the_requested_terrain(shape, legacy):
    cfg = terrain_cfg(generator_profile="legbot_wf")
    if legacy:
        cfg.wf_sub_terrains = {"obstacles": {"proportion": 1.0}}
    else:
        cfg.wf_sub_terrains["platform_" + shape]["platform_width"] = 3.6
    original = cfg.wf_sub_terrains
    saved = copy.deepcopy(original)
    height = configure_fixed_platform(cfg, shape, 0.234, num_envs=2)
    assert height == pytest.approx(.235)
    assert cfg.num_rows == 1 and cfg.num_cols == 2
    assert cfg.yaw_init_range == 0 and cfg.spawn_xy_range == 0
    assert cfg.wf_sub_terrains["platform_" + shape]["platform_width"] == (3.2 if legacy else 3.6)
    terrain = WFTerrain(cfg, 2, mesh_converter=None)
    assert np.all(terrain.tile_types == "platform_" + shape)
    assert set(np.unique(terrain.height_field_raw)) == {0, 47 if shape == "block" else -47}
    np.testing.assert_allclose(terrain.env_origins[..., 2], 0 if shape == "block" else -height)
    assert original == saved


@pytest.mark.parametrize("shape,height,mesh", [
    ("block", None, "trimesh"), (None, .3, "trimesh"), ("slope", .3, "trimesh"),
    ("pit", -.1, "heightfield"), ("block", .5, "plane"), ("pit", float("nan"), "trimesh"),
    ("block", .001, "trimesh"),
])
def test_fixed_play_rejects_invalid_scenarios(shape, height, mesh):
    cfg = terrain_cfg(generator_profile="legbot_wf", mesh_type=mesh)
    with pytest.raises(ValueError):
        configure_fixed_platform(cfg, shape, height, num_envs=1)


def test_play_without_fixed_platform_preserves_the_restored_environment():
    cfg = terrain_cfg(generator_profile="legbot_wf")
    saved = copy.deepcopy(vars(cfg))
    assert configure_fixed_platform(cfg, None, None, num_envs=1) is None
    assert vars(cfg) == saved


@pytest.mark.parametrize("shape", ["block", "pit"])
def test_fixed_play_can_match_a_five_metre_square(shape):
    cfg = terrain_cfg(generator_profile="legbot_wf", border_size=0.)
    configure_fixed_platform(cfg, shape, .5, num_envs=1, width=5.)
    terrain = WFTerrain(cfg, 1, mesh_converter=None)
    assert (terrain.height_field_raw != 0).sum() == 50 ** 2
    assert terrain.env_origins[0, 0, 0] == (1.0 if shape == "block" else 4.5)


@pytest.mark.parametrize("shape,mesh,outer,central,height", [
    ("block", "trimesh", (2.0, 6.9), (2.0, 6.9), .5),
    ("pit", "trimesh", (1.9, 7.0), (1.9, 7.0), -.5),
    ("block", "heightfield", (1.9, 7.0), (2.0, 6.9), .5),
    ("pit", "heightfield", (1.9, 7.0), (2.0, 6.9), -.5),
])
def test_traversal_uses_the_physical_wf_face_on_each_side(shape, mesh, outer, central, height):
    cfg = terrain_cfg(generator_profile="legbot_wf", border_size=0.)
    configure_fixed_platform(cfg, shape, .5, num_envs=2, width=5.)
    terrain = WFTerrain(cfg, 2, mesh_converter=None)
    cfg.mesh_type = mesh
    outer_xy, central_xy, middle_height = wf_obstacle_bounds(terrain, cfg, 0, 1)
    for actual, limits in ((outer_xy, outer), (central_xy, central)):
        lo, hi = limits
        assert actual == pytest.approx((lo, hi, lo + 9., hi + 9.))
    assert middle_height == pytest.approx(height)


@pytest.mark.parametrize("name,sign", [("stairs_up", -1), ("stairs_down", 1)])
def test_stair_play_distinguishes_outermost_riser_from_central_landing(name, sign):
    cfg = terrain_cfg(border_size=0., num_rows=1, num_cols=1)
    cfg.wf_sub_terrains = {name: dict(cfg.wf_sub_terrains[name], proportion=1.)}
    terrain = WFTerrain(cfg, 1, mesh_converter=None)
    outer, middle, height = wf_obstacle_bounds(terrain, cfg, 0, 0)
    assert sign * height > cfg.wf_sub_terrains[name]["step_height_range"][0]
    assert outer[0] < middle[0] < middle[1] < outer[1]
    assert outer[:2] == outer[2:] and middle[:2] == middle[2:]
    assert (middle[0] - outer[0]) >= .35


@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1)])
def test_available_landing_ground_ends_before_the_adjacent_obstacle(axis, sign):
    cfg = terrain_cfg(generator_profile="legbot_wf", border_size=2.)
    configure_fixed_platform(cfg, "block", .5, num_envs=3, width=5.)
    cfg.num_rows = 3
    terrain = WFTerrain(cfg, 9, mesh_converter=None)
    outer, _, _ = wf_obstacle_bounds(terrain, cfg, 1, 1)
    # Two adjoining flat aprons are available between the heightfield ramps.
    assert wf_landing_distance(terrain, outer, axis, sign) == pytest.approx(3.9)


@pytest.mark.parametrize("shape,height,width", [
    (None, None, 5.), ("block", .5, 9.), ("pit", .5, .03), ("block", .5, float("nan")),
])
def test_fixed_play_rejects_unsupported_platform_width(shape, height, width):
    cfg = terrain_cfg(generator_profile="legbot_wf")
    with pytest.raises(ValueError):
        configure_fixed_platform(cfg, shape, height, num_envs=1, width=width)


def test_play_samples_all_robots_before_the_terminal_pose_is_reset(monkeypatch):
    pytest.importorskip("isaacgym")
    import torch
    from legged_gym.envs.nezha.nezha_moe_cts_env import NezhaMoECTSEnv
    from legged_gym.scripts._play_env import StudentPlayEnv

    monkeypatch.setattr(NezhaMoECTSEnv, "check_termination", lambda self: None)
    env = StudentPlayEnv.__new__(StudentPlayEnv)
    env.cfg = SimpleNamespace(env=SimpleNamespace(fail_to_terminal_time_s=.6))
    env.dt, env.play_step_limit, env.traversal = .02, 1500, None
    env.fail_buf = torch.zeros(2)
    env.episode_length_buf = torch.tensor([1500, 32])
    env.play_success = torch.zeros(2, dtype=torch.bool)
    env.reset_buf = torch.zeros_like(env.play_success)
    env.time_out_buf = torch.zeros_like(env.play_success)
    env.dof_pos = torch.arange(32).reshape(2, 16).float()

    samples = []

    def sample(terminal_reasons):
        samples.append((terminal_reasons.copy(), env.dof_pos.clone()))

    env.joint_angles = SimpleNamespace(sample=sample)
    env.check_termination()
    # Automatic reset replaces the finished robot's active DOF buffer immediately.
    env.dof_pos[env.reset_buf] = 0.
    reasons, angles = samples[0]
    assert reasons == {0: "time_limit"}
    torch.testing.assert_close(angles, torch.arange(32).reshape(2, 16).float())
