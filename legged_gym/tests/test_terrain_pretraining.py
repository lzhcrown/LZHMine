"""Ground offline G13 training data in the actual terrain and sensor configuration."""

import numpy as np
import torch

from experiments.g13_v1.pretrain import corrupt
from experiments.g13_v1.terrain_scans import GRID_X, GRID_Y, TERRAIN_NAMES, make_scan_dataset, robot_scans, terrain_config
from experiments.registry import clone_config, get_experiment
from test_cts_experiments import actual_config_module


def test_offline_recipe_matches_isaac_config(actual_config_module):
    online = clone_config(actual_config_module.NezhaMoECTSCfg())
    get_experiment("g13_v1").configure_environment(online)
    assert online.terrain.mesh_type == "trimesh"
    for field, value in vars(terrain_config()).items():
        assert value == getattr(online.terrain, field)
    x, y = np.meshgrid(online.terrain.measured_points_x, online.terrain.measured_points_y)
    np.testing.assert_allclose(GRID_X, x.ravel())
    np.testing.assert_allclose(GRID_Y, y.ravel())


def test_yaw_rotated_scan_uses_minimum_and_training_sign_clip_and_scale():
    cfg = terrain_config()
    raster = np.zeros((91, 91), dtype=np.int16)
    raster[40:, :] = -600  # 3 m drop; three-vertex minimum observes it from cell 39.
    centers = np.array([[3.85, 4.25], [3.85, 4.25]])
    actual = robot_scans(raster, centers, np.array([0, np.pi / 2]), np.array([0.60, 0.60]), cfg)
    # At yaw zero the drop lies ahead. At +90 degrees it is on the robot's right.
    np.testing.assert_allclose(actual[0], np.where(GRID_X >= 0.1, 2.5, 0.25))
    np.testing.assert_allclose(actual[1], np.where(GRID_Y <= -0.1, 2.5, 0.25))


def test_split_seed_reproduces_maps_and_masks_and_validation_uses_fresh_examples():
    a, ids = make_scan_dataset(180, 10)
    b, _ = make_scan_dataset(180, 10)
    c, val_ids = make_scan_dataset(180, 11)
    np.testing.assert_array_equal(a, b)
    assert set(ids) == set(val_ids) == set(range(len(TERRAIN_NAMES)))
    assert np.all(np.any(c != a, axis=1))
    assert np.all(np.ptp(a[ids == TERRAIN_NAMES.index("flat")], axis=1) == 0)
    assert np.ptp(a[ids == TERRAIN_NAMES.index("platform_block")], axis=1).max() > 1
    clean = torch.from_numpy(a)
    values, mask = corrupt(clean, torch.Generator().manual_seed(1), False)
    assert mask.any(1).all() and (~mask).any(1).all()
    assert values[mask].eq(0).all()
    assert (values[~mask] - clean[~mask]).abs().max() <= 0.050001
