"""Fixed WF platform geometry for inspecting a deployed student in Isaac Gym."""

import math

import numpy as np

from legged_gym.terrains.wf import default_wf_sub_terrains, make_height_field


def wf_obstacle_bounds(terrain, cfg, row, col):
    """Return actual outer and central surface bounds (xmin, xmax, ymin, ymax) and height."""
    name = terrain.tile_types[row, col]
    if name not in ("stairs_up", "stairs_down", "platform_block", "platform_pit", "high_platform"):
        return None
    # These five WF geometries are deterministic. Read their native raster so
    # older checkpoints' floor quantization and the finer stair grid also work.
    samples = make_height_field(name, terrain.difficulties[row, col], cfg)
    spec = cfg.wf_sub_terrains[name]
    spacing = spec.get("horizontal_scale", cfg.horizontal_scale)
    threshold = spec.get("slope_threshold") if cfg.mesh_type == "trimesh" else None
    cx, cy = (size // 2 for size in samples.shape)
    level = samples[cx, cy]
    if level == 0:
        return None
    outer, middle = [], []
    for origin, line in ((row * cfg.terrain_length, samples[:, cy]),
                         (col * cfg.terrain_width, samples[cx, :])):
        for bounds, mask in ((outer, line != 0), (middle, line == level)):
            indices = np.flatnonzero(mask)
            jump = (int(line[indices[0]]) - int(line[indices[0] - 1])) * cfg.vertical_scale
            # Mesh conversion moves a vertical face to the higher endpoint.
            # A native heightfield keeps the ramp between adjacent samples.
            vertical = threshold is not None and abs(jump) > threshold * spacing
            expand = int(jump < 0 if vertical else bounds is outer)
            lower = indices[0] - expand
            upper = indices[-1] + expand
            bounds.extend((origin + lower * spacing, origin + upper * spacing))
    return outer, middle, level * cfg.vertical_scale


def wf_landing_distance(terrain, outer, axis, sign):
    """Continuous zero-height ground available beyond the final obstacle face."""
    bounds = np.asarray(outer).reshape(2, 2)
    edge = bounds[axis, int(sign > 0)]
    scale, border = terrain.cfg.horizontal_scale, terrain.cfg.border_size
    grid = terrain.height_field_raw if axis == 0 else terrain.height_field_raw.T
    along = (edge + border) / scale
    start = math.floor(along + 1e-7) + 1 if sign > 0 else math.ceil(along - 1e-7) - 1
    if not 0 <= start < grid.shape[0]:
        return 0.
    sides = (bounds[1 - axis] + border) / scale
    lo, hi = max(0, math.floor(sides[0])), min(grid.shape[1], math.ceil(sides[1]) + 1)
    ground = grid[start::sign, lo:hi]
    blocked = np.flatnonzero((ground != 0).any(axis=1))
    clear = blocked[0] if len(blocked) else len(ground)
    return max(0., sign * ((start + sign * (clear - 1)) * scale - border - edge))


def configure_fixed_platform(terrain, shape, height, num_envs, width=None):
    """Override the restored play terrain, returning its quantized height in metres."""
    if shape is None and height is None and width is None:
        return None
    if shape not in ("block", "pit") or height is None:
        raise ValueError("Supply both --platform_shape {block,pit} and --platform_height METRES")
    if terrain.mesh_type not in ("trimesh", "heightfield") or terrain.generator_profile != "legbot_wf":
        raise ValueError("Fixed platform play requires a WF trimesh or heightfield")
    if not math.isfinite(height) or height <= 0:
        raise ValueError("--platform_height must be finite and positive")
    samples = round(height / terrain.vertical_scale)
    if not 1 <= samples <= 32767:
        raise ValueError("--platform_height must fit a nonzero signed 16-bit terrain sample")
    kind = "platform_" + shape
    specs = {name: dict(spec, proportion=0.0) for name, spec in terrain.wf_sub_terrains.items()}
    # Older maps store scattered obstacles or a pit under 'high_platform'.
    if kind not in specs:
        specs[kind] = default_wf_sub_terrains()[kind]
    if width is not None:
        if not math.isfinite(width) or width <= 0:
            raise ValueError("--platform_width must be finite and positive")
        spacing = specs[kind].get("horizontal_scale", terrain.horizontal_scale)
        side_cells = round(width / spacing)
        usable_cells = (int(min(terrain.terrain_length, terrain.terrain_width) / spacing) + 1
                        - 2 * (int(terrain.sub_terrain_border_width / spacing) + 1))
        actual_width = side_cells * spacing
        if not 0 < side_cells < usable_cells - 1 or (terrain.terrain_length - actual_width) / 4 < 0.5:
            raise ValueError("--platform_width must fit the WF tile and leave a clear approach")
        if shape == "pit" and actual_width < 2 + 2 * spacing:
            raise ValueError("--platform_width must cover the WF pit spawn height sampling window")
        specs[kind]["platform_width"] = actual_width
    specs[kind].update(proportion=1.0, platform_height_range=[height, height], round_height=True)
    terrain.wf_sub_terrains = specs
    terrain.terrain_proportions = [spec["proportion"] for spec in specs.values()]
    terrain.curriculum = False
    terrain.num_rows = 1
    terrain.num_cols = min(terrain.num_cols, num_envs)
    terrain.max_init_terrain_level = 0
    terrain.yaw_init_range = 0.0
    terrain.spawn_xy_range = 0.0
    return samples * terrain.vertical_scale
