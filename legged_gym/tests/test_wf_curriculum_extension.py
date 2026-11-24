"""Appended training rows and checkpoint compatibility, without Isaac Gym."""

from copy import deepcopy

import numpy as np
import pytest

from experiments.registry import PLATFORM_EXPERIMENT, PLATFORM_V1_EXPERIMENT, clone_config, get_experiment
from legged_gym.terrains.wf import PLATFORM_TYPES, WFTerrain
from test_cts_experiments import actual_config_module, config_dict
from test_cts_runner import cpu_task_registry
from test_wf_terrain import preserve_numpy_random_state


def test_dreamwaq_progression_preserves_every_g12_terrain_height(actual_config_module):
    def generate(key):
        cfg = clone_config(actual_config_module.NezhaMoECTSCfg())
        get_experiment(key).configure_environment(cfg)
        cfg.terrain.mesh_type, cfg.terrain.seed = "heightfield", 17
        np.random.seed(17)
        return WFTerrain(cfg.terrain, 1, None)

    original, revised = generate(PLATFORM_EXPERIMENT), generate(PLATFORM_V1_EXPERIMENT)
    assert original.tile_types.shape == (20, 20)
    np.testing.assert_array_equal(revised.tile_types, original.tile_types)
    np.testing.assert_array_equal(revised.difficulties, original.difficulties)
    np.testing.assert_array_equal(revised.height_field_raw, original.height_field_raw)


def test_extended_map_keeps_seeded_lower_rows_and_appends_each_terrain(actual_config_module):
    cfg = clone_config(actual_config_module.NezhaMoECTSCfg()).terrain
    assert (cfg.num_rows, cfg.difficulty_reference_rows) == (20, 10)
    cfg.mesh_type, cfg.seed = "heightfield", 17
    original_cfg = deepcopy(cfg)
    original_cfg.num_rows = 10
    np.random.seed(17)
    original = WFTerrain(original_cfg, 1, None)
    np.random.seed(17)
    extended = WFTerrain(cfg, 1, None)
    np.testing.assert_array_equal(extended.tile_types[:10], original.tile_types)
    np.testing.assert_array_equal(extended.difficulties[:10], original.difficulties)
    end = extended.border + original_cfg.num_rows * extended.length_per_env_pixels + 1
    np.testing.assert_array_equal(extended.height_field_raw[:end], original.height_field_raw[:end])
    expected_jitter = np.random.default_rng(17).uniform(size=(2, 20, 10))
    for block in range(2):
        rows = np.arange(block * 10, (block + 1) * 10)
        expected = (rows[:, None] + expected_jitter[block].T) / 10
        np.testing.assert_array_equal(extended.difficulties[rows], expected)
    stairs_col = extended.column_types.index("stairs_up")
    for row in (0, 9, 10, 19):
        x = extended.border + row * extended.length_per_env_pixels
        y = extended.border + stairs_col * extended.width_per_env_pixels
        cx, cy = extended.length_per_env_pixels // 2, extended.width_per_env_pixels // 2
        profile = extended.height_field_raw[x:x + cx + 1, y + cy]
        nonzero_steps = np.diff(profile)[np.diff(profile) != 0]
        d = extended.difficulties[row, stairs_col]
        expected_step = -int((0.05 + d * (0.257 - 0.05)) / cfg.vertical_scale)
        np.testing.assert_array_equal(np.unique(nonzero_steps), [expected_step])


@pytest.mark.parametrize("warm_start", [False, True])
def test_legacy_wf_map_resume_or_new_run(actual_config_module, cpu_task_registry, warm_start):
    target = clone_config(actual_config_module.NezhaMoECTSCfg())
    old_terrain = dict(num_rows=7, generator_profile="legbot_wf", terrain_proportions=[1.0],
                       wf_sub_terrains={"obstacles": {"proportion": 1.0}})
    old_settings = config_dict(cpu_task_registry, target)["commands"]["wf_settings"]
    limits = old_settings["terrain_max_command_ranges"]
    for kind in PLATFORM_TYPES:
        del limits[kind]
    limits["obstacles"]["lin_vel_y"] = [-0.3, 0.3]
    source = dict(terrain=old_terrain, commands={"wf_settings": old_settings})
    cpu_task_registry.restore_environment_config(target, source, warm_start)
    if warm_start:
        assert (target.terrain.num_rows, target.terrain.difficulty_reference_rows) == (20, 10)
        for kind in PLATFORM_TYPES:
            assert target.terrain.wf_sub_terrains[kind]["proportion"] == 0.10
            assert target.commands.wf_settings["terrain_max_command_ranges"][kind]["lin_vel_y"] == [-0.3, 0.3]
        assert "obstacles" not in target.terrain.wf_sub_terrains
    else:
        assert (target.terrain.num_rows, target.terrain.difficulty_reference_rows) == (7, 7)
        assert target.terrain.wf_sub_terrains == old_terrain["wf_sub_terrains"]
    assert all(kind not in limits for kind in PLATFORM_TYPES)


@pytest.mark.parametrize("warm_start", [False, True])
def test_previous_20_row_checkpoint_replays_or_upgrades_stair_treads(
        actual_config_module, cpu_task_registry, warm_start):
    target = clone_config(actual_config_module.NezhaMoECTSCfg())
    saved_terrain = config_dict(cpu_task_registry, target)["terrain"]
    assert (saved_terrain["num_rows"], saved_terrain["difficulty_reference_rows"]) == (20, 10)
    for kind in ("stairs_up", "stairs_down"):
        stair = saved_terrain["wf_sub_terrains"][kind]
        stair.pop("horizontal_scale")
        stair["step_width"] = .31
        stair["step_height_range"] = [.06, .24]
    original = deepcopy(saved_terrain)
    cpu_task_registry.restore_environment_config(target, {"terrain": saved_terrain}, warm_start)
    assert (target.terrain.num_rows, target.terrain.difficulty_reference_rows) == (20, 10)
    for kind in ("stairs_up", "stairs_down"):
        stair = target.terrain.wf_sub_terrains[kind]
        assert stair["step_height_range"] == [.06, .24]
        if warm_start:
            assert (stair["step_width"], stair["horizontal_scale"]) == (.35, .05)
        else:
            assert stair == original["wf_sub_terrains"][kind]
    assert saved_terrain == original
