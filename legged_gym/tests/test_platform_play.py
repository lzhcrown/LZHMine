"""A fixed student replay must reproduce both platform shapes on old and new WF maps."""

import copy

import numpy as np
import pytest

from legged_gym.scripts._platform_play import configure_fixed_platform
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


@pytest.mark.parametrize("shape,height,width", [
    (None, None, 5.), ("block", .5, 9.), ("pit", .5, .03), ("block", .5, float("nan")),
])
def test_fixed_play_rejects_unsupported_platform_width(shape, height, width):
    cfg = terrain_cfg(generator_profile="legbot_wf")
    with pytest.raises(ValueError):
        configure_fixed_platform(cfg, shape, height, num_envs=1, width=width)
