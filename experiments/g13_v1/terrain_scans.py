"""Robot-centred privileged scans from the G13 WF terrain generator, without Isaac Gym."""

from types import SimpleNamespace

import numpy as np

from legged_gym.terrains.wf import (
    _rasterize_tile, high_platform_wf_sub_terrains, make_height_field, terrain_level_count,
)
from rsl_rl.utils.cts_observations import default_observation_config


def terrain_config():
    # Geometry consumed by make_height_field in Nezha's G13 trimesh task.
    return SimpleNamespace(terrain_length=9.0, terrain_width=9.0, horizontal_scale=0.1,
                           vertical_scale=0.005, sub_terrain_border_width=0.5, num_rows=20,
                           ordinary_max_level=9, difficulty_reference_rows=10,
                           wf_sub_terrains=high_platform_wf_sub_terrains())


_SCHEMA = default_observation_config()
_HEIGHT = next(term for term in _SCHEMA["privileged_terms"] if term["name"] == "height_scan")
HEIGHT_SCALE = _HEIGHT["scale"]
TERRAIN_NAMES = tuple(name for name, spec in high_platform_wf_sub_terrains().items() if spec["proportion"] > 0)
# NezhaMoECTSEnv._init_height_points uses x-fastest order.
GRID_X, GRID_Y = np.meshgrid(np.arange(-8, 9) / 10, np.arange(-5, 6) / 10)
GRID_X, GRID_Y = GRID_X.ravel(), GRID_Y.ravel()


def _sample_heights(field, x, y, cfg):
    ix = np.clip((x / cfg.horizontal_scale).astype(int), 0, field.shape[0] - 2)
    iy = np.clip((y / cfg.horizontal_scale).astype(int), 0, field.shape[1] - 2)
    height = np.minimum(np.minimum(field[ix, iy], field[ix + 1, iy]), field[ix, iy + 1])
    return height * cfg.vertical_scale


def robot_scans(field, centers, yaw, clearance, cfg):
    """Apply the simulator's yaw transform, three-vertex minimum and CTS normalization."""
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    x = centers[:, :1] + c * GRID_X - s * GRID_Y
    y = centers[:, 1:] + s * GRID_X + c * GRID_Y
    ground = _sample_heights(field, centers[:, 0], centers[:, 1], cfg)
    root_z = ground + clearance
    raw = root_z[:, None] - _SCHEMA["height_offset"] - _sample_heights(field, x, y, cfg)
    return (np.clip(raw, *_HEIGHT["clip"]) * HEIGHT_SCALE).astype(np.float32)


def _scan_centers(rng, count, field, cfg):
    # The 1 m margin keeps every yaw-rotated grid on the tile, including the +1 samples.
    bounds = np.array([cfg.terrain_length, cfg.terrain_width]) - 1.0
    centers = rng.uniform(1.0, bounds, size=(count, 2))
    # Include approaches and step lips. Uniform platform crops often see only level ground.
    rises = ((field[:-1, :-1] != field[1:, :-1]) | (field[:-1, :-1] != field[:-1, 1:]))
    edges = np.argwhere(rises)
    n = rng.binomial(count, 0.65)
    if len(edges) and n:
        edge_xy = edges[rng.integers(len(edges), size=n)] * cfg.horizontal_scale
        centers[:n] = np.clip(edge_xy + rng.uniform(-0.55, 0.55, size=(n, 2)), 1.0, bounds)
    return centers


def make_scan_dataset(samples, seed):
    """Balance active terrain families and cycle curriculum levels using split-local tiles and poses."""
    if samples < len(TERRAIN_NAMES):
        raise ValueError("Each split needs at least one scan per active terrain family")
    rng = np.random.default_rng(seed)
    cfg = terrain_config()
    scans = np.empty((samples, GRID_X.size), dtype=np.float32)
    ids = np.arange(samples) % len(TERRAIN_NAMES)
    # WF uses the legacy NumPy RNG for its roughness. Scope and restore it for library callers.
    state = np.random.get_state()
    try:
        for terrain_id, name in enumerate(TERRAIN_NAMES):
            indices = np.flatnonzero(ids == terrain_id)
            levels = terrain_level_count(cfg, name)
            tile_count = min(len(indices), max(levels, (len(indices) + 31) // 32))
            level_order = rng.permutation(levels)
            for i, subset in enumerate(np.array_split(indices, tile_count)):
                difficulty = (level_order[i % levels] + rng.random()) / cfg.difficulty_reference_rows
                np.random.seed(int(rng.integers(2**32)))
                native = make_height_field(name, difficulty, cfg)
                spacing = cfg.wf_sub_terrains[name].get("horizontal_scale", cfg.horizontal_scale)
                raster = _rasterize_tile(native, spacing, cfg.horizontal_scale)
                n = len(subset)
                scans[subset] = robot_scans(
                    raster, _scan_centers(rng, n, raster, cfg), rng.uniform(-np.pi, np.pi, n),
                    rng.uniform(0.35, 0.70, n), cfg,
                )
    finally:
        np.random.set_state(state)
    return scans, ids
