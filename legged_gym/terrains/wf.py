# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, LZHMine contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""WF-CTS-MOE terrain definitions and Isaac Lab 2.3.2 height-field semantics.

Reference: legbot_lab daff9fb, tasks/go2/mdp/terrains.py (TERRAIN_CFG).
Height-field equations follow IsaacLab v2.3.2 terrains/height_field/
hf_terrains.py and utils.py. No Isaac Lab installation is required.
The global map uses Gym's positive tile coordinates; local geometry is unchanged.
"""

from copy import copy

import numpy as np
from scipy.interpolate import RectBivariateSpline


PLATFORM_TYPES = ("platform_pit", "platform_block")


def default_wf_sub_terrains(*, platform_height_range=(0.05, 0.275), round_platform_height=False):
    """WF curriculum with the obstacle columns divided equally between pits and blocks."""
    roughness = dict(noise_range=[-0.05, 0.05], noise_step=0.005, downsampled_scale=0.2)
    return {
        "wave": dict(proportion=0.05, slope_threshold=10.0,
                     amplitude_range=[0.1, 0.28], num_waves=5, **roughness),
        "slope_up": dict(proportion=0.10, slope_threshold=10.0, inverted=True,
                         slope_range=[0.1, 0.568], platform_width=3.0),
        "slope_down": dict(proportion=0.10, slope_threshold=10.0, inverted=False,
                           slope_range=[0.1, 0.568], platform_width=3.0),
        "rough_slope": dict(proportion=0.05, slope_threshold=10.0, inverted=False,
                            slope_range=[0.1, 0.568], platform_width=3.0, **roughness),
        "stairs_up": dict(proportion=0.25, slope_threshold=0.25, inverted=True,
                          step_height_range=[0.05, 0.257], step_width=0.35, horizontal_scale=0.05, platform_width=3.0),
        "stairs_down": dict(proportion=0.10, slope_threshold=0.25, inverted=False,
                            step_height_range=[0.05, 0.257], step_width=0.35, horizontal_scale=0.05, platform_width=3.0),
        **{name: dict(proportion=0.10, slope_threshold=0.25, platform_width=3.2,
                      platform_height_range=list(platform_height_range), round_height=round_platform_height)
           for name in PLATFORM_TYPES},
        "stepping_stones": dict(proportion=0.0, slope_threshold=0.25,
                                stone_height_max=0.0, stone_width_range=[0.075, 1.575],
                                stone_distance_range=[0.05, 0.10], holes_depth=-10.0, platform_width=4.0),
        "gap": dict(proportion=0.0, gap_width_range=[0.0, 0.9], platform_width=3.0),
        "flat": dict(proportion=0.15, slope_threshold=None),
    }


def high_platform_wf_sub_terrains():
    """G12 uses its original platform height law and nearest-sample quantization."""
    return default_wf_sub_terrains(platform_height_range=(0.20, 0.70), round_platform_height=True)


def column_types(cfg):
    specs = cfg.wf_sub_terrains
    names = list(specs)
    proportions = np.array([specs[name]["proportion"] for name in names], dtype=float)
    if np.any(proportions < 0) or proportions.sum() <= 0:
        raise ValueError("WF terrain proportions must be nonnegative with positive sum")
    cumulative = np.cumsum(proportions / proportions.sum())
    return [names[int(np.where(col / cfg.num_cols + 0.001 < cumulative)[0][0])]
            for col in range(cfg.num_cols)]


def _interpolate(bounds, difficulty):
    return bounds[0] + difficulty * (bounds[1] - bounds[0])


def _whole_cells(distance, spacing):
    cells = round(distance / spacing)
    if cells < 1 or not np.isclose(cells * spacing, distance, rtol=0, atol=1e-8):
        raise ValueError(f"{distance} m is not a positive whole number of {spacing} m cells")
    return cells


def _rasterize_tile(heights, native_scale, map_scale):
    """Sample the native tile on the shared raster, retaining its outermost vertices."""
    if native_scale == map_scale:
        return heights
    if native_scale < map_scale:
        stride = _whole_cells(map_scale, native_scale)
        return heights[::stride, ::stride]
    stride = _whole_cells(native_scale, map_scale)
    # The source heightfield splits each quad along the (0,0)-(1,1) diagonal.
    # Interpolate the two planes and round once to the vertical grid.
    rows, cols = heights.shape
    x = np.arange((rows - 1) * stride + 1)[:, None] / stride
    y = np.arange((cols - 1) * stride + 1) / stride
    ix = np.minimum(x.astype(int), rows - 2)
    iy = np.minimum(y.astype(int), cols - 2)
    fx, fy = x - ix, y - iy
    values = (1 - np.maximum(fx, fy)) * heights[ix, iy]
    values += np.minimum(fx, fy) * heights[ix + 1, iy + 1]
    values += np.maximum(fx - fy, 0) * heights[ix + 1, iy]
    values += np.maximum(fy - fx, 0) * heights[ix, iy + 1]
    return np.rint(values).astype(heights.dtype)


def _roughness(size, shape, horizontal_scale, vertical_scale, spec):
    ds = spec["downsampled_scale"]
    low, high = [int(value / vertical_scale) for value in spec["noise_range"]]
    step = int(spec["noise_step"] / vertical_scale)
    sampled = np.random.choice(np.arange(low, high + step, step),
                               size=(int(size[0] / ds), int(size[1] / ds)))
    # Match Lab's cubic interpolation (Gym's legacy helper is linear).
    x = np.linspace(0, size[0] * horizontal_scale, sampled.shape[0])
    y = np.linspace(0, size[1] * horizontal_scale, sampled.shape[1])
    spline = RectBivariateSpline(x, y, sampled)
    heights = spline(np.linspace(0, size[0] * horizontal_scale, shape[0]),
                     np.linspace(0, size[1] * horizontal_scale, shape[1]))
    return np.rint(heights).astype(np.int16)


def make_height_field(name, difficulty, cfg):
    """Return the full tile, including Lab's extra edge sample and border."""
    spec = cfg.wf_sub_terrains[name]
    hs, vs = spec.get("horizontal_scale", cfg.horizontal_scale), cfg.vertical_scale
    shape = (int(cfg.terrain_length / hs) + 1, int(cfg.terrain_width / hs) + 1)
    result = np.zeros(shape, dtype=np.int16)
    if name == "flat":
        return result
    border = int(cfg.sub_terrain_border_width / hs) + 1
    width, length = shape[0] - 2 * border, shape[1] - 2 * border
    if min(width, length) <= 0:
        raise ValueError("WF terrain must be larger than its sub-terrain border")
    size = (width * hs, length * hs)
    heights = np.zeros((width, length))
    if name == "wave":
        amplitude = int(0.5 * _interpolate(spec["amplitude_range"], difficulty) / vs)
        frequency = 2 * np.pi / (length / spec["num_waves"])
        heights = np.rint(amplitude * (np.cos(np.arange(length)[None, :] * frequency)
                                      + np.sin(np.arange(width)[:, None] * frequency))).astype(np.int16)
        heights += _roughness(size, (width, length), hs, vs, spec)
    elif name in ("slope_up", "slope_down", "rough_slope"):
        slope = _interpolate(spec["slope_range"], difficulty) * (-1 if spec["inverted"] else 1)
        peak = int(slope * size[0] / 2 / vs)
        cx, cy = int(width / 2), int(length / 2)
        xx = (cx - np.abs(cx - np.arange(width))) / cx
        yy = (cy - np.abs(cy - np.arange(length))) / cy
        heights = peak * xx[:, None] * yy[None, :]
        platform = int(spec["platform_width"] / hs / 2)
        height = heights[width // 2 - platform, length // 2 - platform]
        heights = np.rint(np.clip(heights, min(0, height), max(0, height))).astype(np.int16)
        if name == "rough_slope":
            heights += _roughness(size, (width, length), hs, vs, spec)
    elif name in ("stairs_up", "stairs_down"):
        step = _interpolate(spec["step_height_range"], difficulty) * (-1 if spec["inverted"] else 1)
        step = int(step / vs)
        # Historical checkpoint specs used floor quantization (0.31 m on a 0.10 m grid).
        step_width = (_whole_cells(spec["step_width"], hs) if "horizontal_scale" in spec
                      else int(spec["step_width"] / hs))
        if step_width < 1:
            raise ValueError("Stair tread must span at least one cell")
        platform = int(spec["platform_width"] / hs)
        x0, x1, y0, y1, height = 0, width, 0, length, 0
        while x1 - x0 > platform and y1 - y0 > platform:
            x0, x1 = x0 + step_width, x1 - step_width
            y0, y1 = y0 + step_width, y1 - step_width
            height += step
            heights[x0:x1, y0:y1] = height
    elif name in PLATFORM_TYPES or name == "high_platform":
        # Old G12 checkpoints store the concave tile as high_platform / pit_width.
        legacy_pit = name == "high_platform"
        side = int(round(spec["pit_width" if legacy_pit else "platform_width"] / hs))
        if not 0 < side < min(width, length) - 1:
            raise ValueError("Platform must leave level ground on every side")
        x0, y0 = (width - side) // 2, (length - side) // 2
        samples = _interpolate(spec["platform_height_range"], difficulty) / vs
        height = int(round(samples)) if legacy_pit or spec["round_height"] else int(samples)
        heights[x0:x0 + side, y0:y0 + side] = height if name == "platform_block" else -height
    elif name == "obstacles":
        # Restore the original scattered obstacles only for older saved maps.
        peak = int(_interpolate(spec["obstacle_height_range"], difficulty) / vs)
        low, high = [int(value / hs) for value in spec["obstacle_width_range"]]
        for _ in range(spec["num_obstacles"]):
            height = (np.random.choice([-peak, -peak // 2, peak // 2, peak])
                      if spec["obstacle_height_mode"] == "choice" else peak)
            sx, sy = int(np.random.choice(np.arange(low, high, 4))), int(np.random.choice(np.arange(low, high, 4)))
            x = min(int(np.random.choice(np.arange(0, width, 4))), width - sx)
            y = min(int(np.random.choice(np.arange(0, length, 4))), length - sy)
            heights[x:x + sx, y:y + sy] = height
        platform = int(spec["platform_width"] / hs)
        heights[(width - platform) // 2:(width + platform) // 2,
                (length - platform) // 2:(length + platform) // 2] = 0
    else:
        raise ValueError("WF reference disables terrain type: " + name)
    result[border:-border, border:-border] = np.rint(heights).astype(np.int16)
    return result


def _plane(x0, x1, y0, y1):
    return (np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], dtype=np.float32),
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))


class WFTerrain:
    """Same tile distribution and geometry, assembled for Gym's mesh interface."""

    def __init__(self, cfg, num_robots, mesh_converter):
        self.num_robots, self.type = num_robots, cfg.mesh_type
        # Native heightfield collision requires one uniform raster. Triangle
        # meshes retain each tile's native spacing and the existing sensor grid.
        map_scale = cfg.horizontal_scale
        if self.type == "heightfield":
            map_scale = min([map_scale] + [spec.get("horizontal_scale", map_scale)
                            for spec in cfg.wf_sub_terrains.values() if spec["proportion"] > 0])
        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.cfg = copy(cfg)
        self.cfg.horizontal_scale = map_scale
        self.env_length, self.env_width = cfg.terrain_length, cfg.terrain_width
        self.length_per_env_pixels = int(self.env_length / map_scale)
        self.width_per_env_pixels = int(self.env_width / map_scale)
        self.border = int(cfg.border_size / map_scale)
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border + 1
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels + 2 * self.border + 1
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.difficulties = np.zeros((cfg.num_rows, cfg.num_cols))
        self.column_types = column_types(cfg)
        self.tile_types = np.empty((cfg.num_rows, cfg.num_cols), dtype=object)
        seed = getattr(cfg, "seed", None)
        self.np_rng = np.random.default_rng(np.random.get_state()[1][0] if seed is None else seed)
        meshes, vertex_count = [], 0
        names = list(cfg.wf_sub_terrains)
        probabilities = np.array([cfg.wf_sub_terrains[name]["proportion"] for name in names])
        probabilities /= probabilities.sum()
        reference_rows = getattr(cfg, "difficulty_reference_rows", cfg.num_rows)
        if reference_rows <= 0:
            raise ValueError("difficulty_reference_rows must be positive")
        lower, upper = cfg.difficulty_range
        if cfg.curriculum:
            # Finish the original map before generating the appended rows so
            # their random draws do not perturb the earlier levels.
            indices = ((row, col) for start in range(0, cfg.num_rows, reference_rows)
                       for col in range(cfg.num_cols)
                       for row in range(start, min(start + reference_rows, cfg.num_rows)))
        else:
            indices = ((row, col) for row in range(cfg.num_rows) for col in range(cfg.num_cols))
        for row, col in indices:
            name = self.column_types[col] if cfg.curriculum else names[self.np_rng.choice(len(names), p=probabilities)]
            jitter = self.np_rng.uniform()
            difficulty = ((row + jitter) / reference_rows if cfg.curriculum
                          else jitter * (cfg.num_rows / reference_rows))
            difficulty = lower + (upper - lower) * difficulty
            self.difficulties[row, col], self.tile_types[row, col] = difficulty, name
            heights = make_height_field(name, difficulty, cfg)
            native_scale = cfg.wf_sub_terrains[name].get("horizontal_scale", cfg.horizontal_scale)
            raster = _rasterize_tile(heights, native_scale, map_scale)
            ix = self.border + row * self.length_per_env_pixels
            iy = self.border + col * self.width_per_env_pixels
            self.height_field_raw[ix:ix + raster.shape[0], iy:iy + raster.shape[1]] = raster
            spawn_x = self.env_length / 2
            if name == "platform_block":
                # Start halfway across the level approach, facing the -X wall.
                spawn_x = (self.env_length - cfg.wf_sub_terrains[name]["platform_width"]) / 4
                spawn_z = 0.0
            else:
                x1, x2 = [int((spawn_x + delta) / native_scale) for delta in (-1, 1)]
                y1, y2 = [int((self.env_width / 2 + delta) / native_scale) for delta in (-1, 1)]
                spawn_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
            self.env_origins[row, col] = [row * self.env_length + spawn_x,
                                          (col + 0.5) * self.env_width, spawn_z]
            if self.type == "trimesh":
                if name == "flat":
                    vertices, faces = _plane(0, self.env_length, 0, self.env_width)
                else:
                    vertices, faces = mesh_converter(heights, native_scale, cfg.vertical_scale,
                                                     cfg.wf_sub_terrains[name]["slope_threshold"])
                vertices[:, 0] += cfg.border_size + row * self.env_length
                vertices[:, 1] += cfg.border_size + col * self.env_width
                meshes.append((vertices, faces + vertex_count))
                vertex_count += len(vertices)
        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            b = cfg.border_size
            x, y = cfg.num_rows * self.env_length, cfg.num_cols * self.env_width
            for bounds in ((0, b, 0, y + 2 * b), (b + x, x + 2 * b, 0, y + 2 * b),
                           (b, b + x, 0, b), (b, b + x, b + y, y + 2 * b)):
                if b > 0:
                    vertices, faces = _plane(*bounds)
                    meshes.append((vertices, faces + vertex_count))
                    vertex_count += len(vertices)
            self.vertices = np.concatenate([item[0] for item in meshes])
            self.triangles = np.concatenate([item[1] for item in meshes])
