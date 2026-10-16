# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, LZHMine contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""WF-CTS-MOE terrain definitions and Isaac Lab 2.3.2 height-field semantics.

Reference: legbot_lab daff9fb, tasks/go2/mdp/terrains.py (TERRAIN_CFG).
Height-field equations follow IsaacLab v2.3.2 terrains/height_field/
hf_terrains.py and utils.py. No Isaac Lab installation is required.
The global map uses Gym's positive tile coordinates; local geometry is unchanged.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def default_wf_sub_terrains():
    """Ordered definitions, including the reference's disabled terrain types."""
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
                          step_height_range=[0.05, 0.257], step_width=0.31, platform_width=3.0),
        "stairs_down": dict(proportion=0.10, slope_threshold=0.25, inverted=False,
                            step_height_range=[0.05, 0.257], step_width=0.31, platform_width=3.0),
        "obstacles": dict(proportion=0.20, slope_threshold=0.25, obstacle_height_mode="choice",
                          obstacle_width_range=[1.0, 2.0], obstacle_height_range=[0.05, 0.275],
                          num_obstacles=20, platform_width=3.0),
        "stepping_stones": dict(proportion=0.0, slope_threshold=0.25,
                                stone_height_max=0.0, stone_width_range=[0.075, 1.575],
                                stone_distance_range=[0.05, 0.10], holes_depth=-10.0, platform_width=4.0),
        "gap": dict(proportion=0.0, gap_width_range=[0.0, 0.9], platform_width=3.0),
        "flat": dict(proportion=0.15, slope_threshold=None),
    }


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
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
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
        step_width = int(spec["step_width"] / hs)
        platform = int(spec["platform_width"] / hs)
        x0, x1, y0, y1, height = 0, width, 0, length, 0
        while x1 - x0 > platform and y1 - y0 > platform:
            x0, x1 = x0 + step_width, x1 - step_width
            y0, y1 = y0 + step_width, y1 - step_width
            height += step
            heights[x0:x1, y0:y1] = height
    elif name == "obstacles":
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
        self.cfg, self.num_robots, self.type = cfg, num_robots, cfg.mesh_type
        self.env_length, self.env_width = cfg.terrain_length, cfg.terrain_width
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.border = int(cfg.border_size / cfg.horizontal_scale)
        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
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
        indices = ((row, col) for col in range(cfg.num_cols) for row in range(cfg.num_rows)) if cfg.curriculum else (
            (row, col) for row in range(cfg.num_rows) for col in range(cfg.num_cols))
        for row, col in indices:
            name = self.column_types[col] if cfg.curriculum else names[self.np_rng.choice(len(names), p=probabilities)]
            lower, upper = cfg.difficulty_range
            difficulty = ((row + self.np_rng.uniform()) / cfg.num_rows if cfg.curriculum else self.np_rng.uniform())
            difficulty = lower + (upper - lower) * difficulty
            self.difficulties[row, col], self.tile_types[row, col] = difficulty, name
            heights = make_height_field(name, difficulty, cfg)
            ix = self.border + row * self.length_per_env_pixels
            iy = self.border + col * self.width_per_env_pixels
            self.height_field_raw[ix:ix + heights.shape[0], iy:iy + heights.shape[1]] = heights
            x1, x2 = [int((self.env_length / 2 + delta) / cfg.horizontal_scale) for delta in (-1, 1)]
            y1, y2 = [int((self.env_width / 2 + delta) / cfg.horizontal_scale) for delta in (-1, 1)]
            self.env_origins[row, col] = [(row + 0.5) * self.env_length, (col + 0.5) * self.env_width,
                                          np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale]
            if self.type == "trimesh":
                if name == "flat":
                    vertices, faces = _plane(0, self.env_length, 0, self.env_width)
                else:
                    vertices, faces = mesh_converter(heights, cfg.horizontal_scale, cfg.vertical_scale,
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
