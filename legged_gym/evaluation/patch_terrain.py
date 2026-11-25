# Adapted from wheel_gym_CQ/wheel_legged_gym/evaluation/patch_terrain.py
"""Fixed evaluation regions with external support platforms.

The measurement ROI is distinct from the surrounding support corridor.
Concentric stairs use the tread, count and central-platform dimensions in the
shared presets. The narrow slope has its own rectangular ROI and equal-width
entrance/exit pads; ground outside its lateral edges stays at zero height.
"""
import copy
import math

import numpy as np

from .terrain import EvaluationTerrain, _grid_cells, _number
from .presets import (PATCH_TERRAIN_TYPES, PATCH_SUPPORTED_TERRAIN_TYPES, PATCH_SIZE_M,
                      PATCH_PLATFORM_LENGTH_M, PATCH_CORRIDOR_WIDTH_M,
                      PATCH_STAIR_TREAD_M, PATCH_STEPS_PER_SIDE, PATCH_STAIR_PLATFORM_M,
                      PATCH_STAIR_HEIGHTS_M, PATCH_HIGH_PLATFORM_SIZE_M, PATCH_HIGH_PLATFORM_HEIGHT_M,
                      PATCH_HIGH_PLATFORM_HEIGHTS_M,
                      PATCH_SLOPE_ANGLE_DEG, PATCH_SLOPE_LENGTH_M, PATCH_SLOPE_WIDTH_M)


class PatchEvaluationTerrain(EvaluationTerrain):
    """Use the regular Terrain mesh interface with per-terrain ROI bounds.

    ``terrain_length``/``terrain_width`` or ``patch_size_m`` describe the default
    square patch. The slope uses a narrower rectangle. ``env_length``/
    ``env_width`` describe the allocated map tile, while ``tile_bounds``
    describes each terrain's actual support corridor. ``patch_bounds`` and ``tile_bounds`` must not be confused.
    """

    def __init__(self, cfg, num_robots):
        self.cfg = copy.deepcopy(cfg)
        self.type = getattr(cfg, "mesh_type", "trimesh")
        if self.type not in ("trimesh", "heightfield"):
            raise ValueError("Local patch evaluation requires trimesh or heightfield")
        self.num_robots = int(num_robots)
        if self.num_robots < 1 or self.num_robots != num_robots:
            raise ValueError("num_robots must be a positive integer")
        self.type_names = tuple(getattr(cfg, "type_names", PATCH_TERRAIN_TYPES))
        if not self.type_names or len(set(self.type_names)) != len(self.type_names):
            raise ValueError("type_names must be nonempty and unique")
        if set(self.type_names) - set(PATCH_SUPPORTED_TERRAIN_TYPES):
            raise ValueError("Unsupported local patch terrain type")
        self.patch_size_m = float(getattr(cfg, "patch_size_m", getattr(cfg, "terrain_length", PATCH_SIZE_M)))
        length = float(getattr(cfg, "terrain_length", self.patch_size_m))
        width = float(getattr(cfg, "terrain_width", self.patch_size_m))
        if any(not math.isclose(value, PATCH_SIZE_M, rel_tol=0., abs_tol=1e-9)
               for value in (self.patch_size_m, length, width)):
            raise ValueError("This protocol requires an exact {0} x {0} metre patch".format(PATCH_SIZE_M))
        self.horizontal_scale = _number(cfg, "horizontal_scale", .02, strict=True)
        self.vertical_scale = _number(cfg, "vertical_scale", .005, strict=True)
        self.platform_length = _number(cfg, "platform_length", PATCH_PLATFORM_LENGTH_M, strict=True)
        self.env_width = _number(cfg, "corridor_width", PATCH_CORRIDOR_WIDTH_M, strict=True)
        if (not math.isclose(self.platform_length, PATCH_PLATFORM_LENGTH_M, rel_tol=0., abs_tol=1e-9)
                or not math.isclose(self.env_width, PATCH_CORRIDOR_WIDTH_M, rel_tol=0., abs_tol=1e-9)):
            raise ValueError("This protocol requires {} m end platforms and a {} m support width".format(
                PATCH_PLATFORM_LENGTH_M, PATCH_CORRIDOR_WIDTH_M))
        self.env_length = 2 * self.platform_length + self.patch_size_m
        border_size = _number(cfg, "border_size", 2., strict=True)
        gap_size = _number(cfg, "gap_size", 2., strict=True)
        self.border = _grid_cells(border_size, self.horizontal_scale, "border_size")
        patch_cells = _grid_cells(self.patch_size_m, self.horizontal_scale, "patch_size_m")
        platform_cells = _grid_cells(self.platform_length, self.horizontal_scale, "platform_length")
        self.length_per_env_pixels = _grid_cells(self.env_length, self.horizontal_scale, "corridor_length")
        self.width_per_env_pixels = _grid_cells(self.env_width, self.horizontal_scale, "corridor_width")
        gap_cells = _grid_cells(gap_size, self.horizontal_scale, "gap_size")
        side_padding_cells = _grid_cells((self.env_width - self.patch_size_m) / 2,
                                         self.horizontal_scale, "lateral_support_width")
        if patch_cells < 4 or self.border < 1 or gap_cells < 1 or side_padding_cells < 1:
            raise ValueError("Patch needs >=4 grid cells and positive surrounding support/gap/border cells")
        self.seed = int(getattr(cfg, "seed", 0))
        if self.seed < 0 or self.seed != getattr(cfg, "seed", 0):
            raise ValueError("seed must be a nonnegative integer")
        self.parameters = {
            "slope_angle_deg": _number(cfg, "slope_angle_deg", PATCH_SLOPE_ANGLE_DEG),
            "slope_length_m": _number(cfg, "slope_length_m", PATCH_SLOPE_LENGTH_M, strict=True),
            "slope_width_m": _number(cfg, "slope_width_m", PATCH_SLOPE_WIDTH_M, strict=True),
            "rough_height": _number(cfg, "rough_height", .04),
            "rough_cell_size": _number(cfg, "rough_cell_size", .08, minimum=self.horizontal_scale),
            "step_heights_m": list(PATCH_STAIR_HEIGHTS_M),
            "high_platform_size_m": PATCH_HIGH_PLATFORM_SIZE_M,
            "step_width": _number(cfg, "step_width", PATCH_STAIR_TREAD_M, minimum=self.horizontal_scale),
            "steps_per_side": PATCH_STEPS_PER_SIDE,
            "central_platform_size_m": PATCH_STAIR_PLATFORM_M,
        }
        platform_heights = {name: PATCH_HIGH_PLATFORM_HEIGHTS_M[name] for name in self.type_names
                            if name in PATCH_HIGH_PLATFORM_HEIGHTS_M}
        if len(platform_heights) > 1:
            self.parameters["high_platform_heights_m"] = platform_heights
        else:
            self.parameters["high_platform_height_m"] = next(
                iter(platform_heights.values()), PATCH_HIGH_PLATFORM_HEIGHT_M)
        if self.parameters["slope_angle_deg"] >= 90:
            raise ValueError("slope_angle_deg must be <90")
        if self.parameters["slope_length_m"] > self.patch_size_m:
            raise ValueError("slope_length_m must fit within the terrain tile")
        slope_length_cells = _grid_cells(self.parameters["slope_length_m"], self.horizontal_scale, "slope_length_m")
        if slope_length_cells < 1:
            raise ValueError("slope_length_m must span at least one grid cell")
        if self.parameters["slope_width_m"] > self.patch_size_m:
            raise ValueError("slope_width_m must fit within the terrain tile")
        slope_cells = _grid_cells(self.parameters["slope_width_m"], self.horizontal_scale, "slope_width_m")
        if slope_cells < 1:
            raise ValueError("slope_width_m must span at least one grid cell")
        if not math.isclose(self.parameters["step_width"], PATCH_STAIR_TREAD_M, rel_tol=0., abs_tol=1e-9):
            raise ValueError("The concentric stair presets require {} m treads".format(PATCH_STAIR_TREAD_M))
        stair_span = (2 * self.parameters["steps_per_side"] * self.parameters["step_width"]
                      + self.parameters["central_platform_size_m"])
        if not math.isclose(stair_span, self.patch_size_m, rel_tol=0., abs_tol=1e-9):
            raise ValueError("Stair treads and central platform must span the full ROI")
        _grid_cells(self.parameters["step_width"], self.horizontal_scale, "step_width")
        for name in self.type_names:
            if name.startswith("stairs_"):
                _grid_cells(int(name.rsplit('_', 1)[1][:-2]) / 100., self.vertical_scale, "stair height")
            elif name in PATCH_HIGH_PLATFORM_HEIGHTS_M:
                _grid_cells(PATCH_HIGH_PLATFORM_HEIGHTS_M[name], self.vertical_scale, "platform height")
                _grid_cells(PATCH_HIGH_PLATFORM_SIZE_M, self.horizontal_scale, "platform size")
                _grid_cells((self.patch_size_m - PATCH_HIGH_PLATFORM_SIZE_M) / 2,
                            self.horizontal_scale, "platform inset")

        geometry_defaults = dict(left_wheel_y=.2362, right_wheel_y=-.2604,
                                 front_wheel_x=.426, rear_wheel_x=-.436, wheel_radius=.11)
        self.wheel_geometry = {}
        fallback_fields = []
        for name, default in geometry_defaults.items():
            value = float(getattr(cfg, name, default))
            if not math.isfinite(value):
                raise ValueError(name + " must be finite")
            self.wheel_geometry[name] = value
            if not hasattr(cfg, name):
                fallback_fields.append(name)
        if (self.wheel_geometry["left_wheel_y"] <= self.wheel_geometry["right_wheel_y"]
                or self.wheel_geometry["front_wheel_x"] <= self.wheel_geometry["rear_wheel_x"]
                or self.wheel_geometry["wheel_radius"] <= 0):
            raise ValueError("Wheel geometry must use signed body-frame offsets and positive radius")
        geometry_source = getattr(cfg, "wheel_geometry_source", "explicit_config")
        if fallback_fields:
            geometry_source = "fallback_default_URDF_stance_offsets" if len(fallback_fields) == len(geometry_defaults) else str(geometry_source) + "; partial_fallback"
        for name, default in (
            ("wheel_x_min", self.wheel_geometry["rear_wheel_x"] - self.wheel_geometry["wheel_radius"]),
            ("wheel_x_max", self.wheel_geometry["front_wheel_x"] + self.wheel_geometry["wheel_radius"]),
            ("wheel_y_min", self.wheel_geometry["right_wheel_y"] - self.wheel_geometry["wheel_radius"]),
            ("wheel_y_max", self.wheel_geometry["left_wheel_y"] + self.wheel_geometry["wheel_radius"]),
        ):
            value = float(getattr(cfg, name, default))
            if not math.isfinite(value):
                raise ValueError(name + " must be finite")
            self.wheel_geometry[name] = value
        if (self.wheel_geometry["wheel_x_min"] >= self.wheel_geometry["wheel_x_max"]
                or self.wheel_geometry["wheel_y_min"] >= self.wheel_geometry["wheel_y_max"]):
            raise ValueError("Wheel collision bounds must be increasing")

        self.cfg.mesh_type = self.type
        self.cfg.terrain_length, self.cfg.terrain_width = self.env_length, self.env_width
        self.cfg.patch_size_m, self.cfg.corridor_width = self.patch_size_m, self.env_width
        self.cfg.horizontal_scale, self.cfg.vertical_scale = self.horizontal_scale, self.vertical_scale
        self.cfg.border_size, self.cfg.gap_size = border_size, gap_size
        self.cfg.platform_length = self.platform_length
        self.cfg.num_rows, self.cfg.num_cols = 1, len(self.type_names)
        self.cfg.num_sub_terrains, self.cfg.curriculum = len(self.type_names), False
        self.cfg.seed = self.seed
        for name, value in self.parameters.items():
            setattr(self.cfg, name, value)

        nx, ny = self.length_per_env_pixels, self.width_per_env_pixels
        core = np.zeros((nx + 1, len(self.type_names) * ny + (len(self.type_names) - 1) * gap_cells + 1))
        self.tile_bounds = np.zeros((len(self.type_names), 4), dtype=np.float64)
        self.corridor_bounds = self.tile_bounds
        self.patch_bounds = np.zeros_like(self.tile_bounds)
        details = []
        previous_right = None
        # Preserve the recorded seed order for maps created before taller platforms were added.
        last_seed_index = max(PATCH_SUPPORTED_TERRAIN_TYPES.index(name) for name in self.type_names)
        seed_order = PATCH_SUPPORTED_TERRAIN_TYPES[:max(len(PATCH_TERRAIN_TYPES), last_seed_index + 1)]
        for index, name in enumerate(self.type_names):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, seed_order.index(name)]))
            patch, description = self._build_patch(name, slope_length_cells if name == "slope" else patch_cells, rng)
            y_cell = index * (ny + gap_cells)
            tile = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            if name == "slope":
                # Align both edges to the height grid for an exact width.
                # An odd number of width cells shifts the center by <= half a cell.
                slope_start = (ny - slope_cells) // 2
                # Continue the exit pad through unused allocation to avoid a cliff.
                profile = np.pad(patch[:, slope_cells // 2],
                                 (platform_cells, nx - slope_length_cells - platform_cells), mode="edge")
                tile[:, slope_start:slope_start + slope_cells + 1] = profile[:, None]
            else:
                tile[platform_cells:platform_cells + patch_cells + 1,
                     side_padding_cells:side_padding_cells + patch_cells + 1] = patch
            if previous_right is not None:
                weights = np.linspace(0., 1., gap_cells + 1)[None, :]
                core[:, y_cell - gap_cells:y_cell + 1] = previous_right[:, None] * (1 - weights) + tile[:, :1] * weights
            core[:, y_cell:y_cell + ny + 1] = tile
            previous_right = tile[:, -1]
            y_origin = index * (self.env_width + gap_size)
            self.tile_bounds[index] = (0., self.env_length, y_origin, y_origin + self.env_width)
            patch_length = self.parameters["slope_length_m"] if name == "slope" else self.patch_size_m
            patch_width = self.parameters["slope_width_m"] if name == "slope" else self.patch_size_m
            patch_y = y_origin + (slope_start * self.horizontal_scale if name == "slope"
                                  else (self.env_width - self.patch_size_m) / 2)
            if name == "slope":
                self.tile_bounds[index, 1] = patch_length + 2 * self.platform_length
                self.tile_bounds[index, 2:] = (patch_y, patch_y + patch_width)
            self.patch_bounds[index] = (self.platform_length, self.platform_length + patch_length,
                                        patch_y, patch_y + patch_width)
            description.update(type=name, patch_bounds=self.patch_bounds[index].tolist(),
                               patch_length_m=patch_length, patch_width_m=patch_width,
                               corridor_length_m=float(self.tile_bounds[index, 1] - self.tile_bounds[index, 0]),
                               corridor_width_m=float(self.tile_bounds[index, 3] - self.tile_bounds[index, 2]),
                               corridor_bounds=self.tile_bounds[index].tolist(),
                               low_x_platform_height_m=float(np.rint(tile[0, ny // 2] / self.vertical_scale) * self.vertical_scale),
                               high_x_platform_height_m=float(np.rint(tile[-1, ny // 2] / self.vertical_scale) * self.vertical_scale))
            details.append(description)

        units = np.rint(core / self.vertical_scale)
        if np.max(np.abs(units)) > np.iinfo(np.int16).max:
            raise ValueError("Patch height exceeds the int16 heightfield range")
        # Continue the narrow exit pad into the longitudinal exterior border.
        self.height_field_raw = np.pad(units.astype(np.int16), self.border, mode="edge")
        self.heightsamples = self.height_field_raw
        self.tot_rows, self.tot_cols = self.height_field_raw.shape
        self.env_origins = np.zeros((1, len(self.type_names), 3))
        self.env_origins[0, :, 0] = self.patch_bounds[:, :2].mean(axis=1)
        self.env_origins[0, :, 1] = self.patch_bounds[:, 2:4].mean(axis=1)
        self.env_origins[0, :, 2] = self.ground_heights(self.env_origins[0, :, :2])
        self.metadata = {
            "protocol": "wheel_window_patch", "patch_size_m": self.patch_size_m,
            "type_names": list(self.type_names), "seed": self.seed, "mesh_type": self.type,
            "horizontal_scale": self.horizontal_scale, "vertical_scale": self.vertical_scale,
            "slope_threshold": getattr(cfg, "slope_treshold", .75),
            "allocated_tile_length": self.env_length, "allocated_tile_width": self.env_width,
            "corridor_lengths_m": (self.tile_bounds[:, 1] - self.tile_bounds[:, 0]).tolist(),
            "corridor_widths_m": (self.tile_bounds[:, 3] - self.tile_bounds[:, 2]).tolist(),
            "platform_length": self.platform_length, "border_size": border_size, "gap_size": gap_size,
            "curriculum": False, "type_seed_order": list(seed_order),
            "parameters": dict(self.parameters), "tiles": details,
            "wheel_geometry": dict(self.wheel_geometry), "wheel_geometry_source": geometry_source,
            "wheel_geometry_fallback_fields": fallback_fields,
            "heading_yaw_rad": 0., "motion": "signed world x; negative direction is backward motion",
            "spawn_alignment": "base 0.75 m in front of entry-edge midpoint; yaw=0; optional explicitly configured XY jitter",
            "spawn_xy_jitter_m": float(getattr(cfg, "spawn_xy_jitter", 0.)),
            "legacy_side_argument": "left/right are accepted as aliases of center; no wheel-track translation",
            "outside_support": "slope and end pads use slope_width_m with zero-height ground outside their sides; other tiles retain lateral support; exit-pad heights continue through unused tile allocation and into the longitudinal exterior border",
            "transition_note": "Concentric stairs have {} risers per side and return to zero at the ROI edge. Roughness has a zero boundary ring. The slope has no forced return: its high end continues onto an equal-height platform.".format(self.parameters["steps_per_side"]),
            "transition_in_measurement_window": True,
            "measurement_window": "first wheel enters ROI until completion, fall or timeout; stairs and square platforms require supported middle and stable far-side landing; other patches use opposite or lateral exits",
            "exit_edge_segments_xy": {
                name: [{"xmin": [[float(x0), float(y0)], [float(x0), float(y1)]],
                        "xmax": [[float(x1), float(y0)], [float(x1), float(y1)]],
                        "ymin": [[float(x0), float(y0)], [float(x1), float(y0)]],
                        "ymax": [[float(x0), float(y1)], [float(x1), float(y1)]]}[name]
                       for x0, x1, y0, y1 in self.patch_bounds]
                for name in ("xmin", "xmax", "ymin", "ymax")},
        }
        if self.type == "trimesh":
            from isaacgym import terrain_utils
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, self.horizontal_scale, self.vertical_scale,
                getattr(cfg, "slope_treshold", .75))

    def _build_patch(self, name, cells, rng):
        x = np.arange(cells + 1) * self.horizontal_scale
        width_cells = (_grid_cells(self.parameters["slope_width_m"], self.horizontal_scale, "slope_width_m")
                       if name == "slope" else cells)
        patch = np.zeros((cells + 1, width_cells + 1))
        description = {}
        if name == "slope":
            patch[:] = (x * math.tan(math.radians(self.parameters["slope_angle_deg"])))[:, None]
            description.update(profile="uniform rise along +x across ROI; equal-height external exit platform",
                               effective_ramp_length_m=self.parameters["slope_length_m"],
                               nominal_angle_deg=self.parameters["slope_angle_deg"],
                               nominal_height_difference_m=float(patch[-1, width_cells // 2]),
                               ramp_width_m=self.parameters["slope_width_m"],
                               lateral_profile_extension_m=0.,
                               forced_return_to_zero=False)
        elif name.startswith("stairs_"):
            height = int(name.rsplit('_', 1)[1][:-2]) / 100.
            sign = -1 if "concave" in name else 1
            # Distance to the nearest of all four ROI edges creates concentric
            # square terraces, not a one-dimensional staircase extruded in y.
            distance_x = np.minimum(x, self.patch_size_m - x)
            distance = np.minimum(distance_x[:, None], distance_x[None, :])
            # The mesh converter moves the lower endpoint onto the higher one.
            # Assign the nominal boundary to the higher terrace for both signs,
            # so convex and concave faces share the nominal tread boundaries.
            step_count = self.parameters["steps_per_side"]
            tread = self.parameters["step_width"]
            levels = np.clip(np.floor((distance + sign * 1e-9) / tread), 0, step_count)
            patch[:] = sign * height * levels
            centerline_levels = list(range(step_count + 1)) + list(range(step_count - 1, -1, -1))
            description.update(profile="concentric square terraces from distance to the four ROI edges",
                               shape="concave" if sign < 0 else "convex", step_height_m=height,
                               step_height_interpretation="per riser", steps_per_side=step_count,
                               tread_width_m=tread, central_platform_size_m=self.parameters["central_platform_size_m"],
                               central_platform_height_m=sign * step_count * height,
                               total_height_or_depth_m=step_count * height,
                               terrace_inset_edges_m=[round(tread * i, 12) for i in range(1, step_count + 1)],
                               heightfield_transition_width_m=self.horizontal_scale,
                               centerline_height_levels_m=[sign * height * level for level in centerline_levels])
        elif name in PATCH_HIGH_PLATFORM_HEIGHTS_M:
            # The mesh converter aligns the lower endpoint with the higher
            # one, giving vertical faces at the exact 5 m square footprint.
            inset = (self.patch_size_m - PATCH_HIGH_PLATFORM_SIZE_M) / 2
            inside = (x >= inset - 1e-9) & (x <= self.patch_size_m - inset + 1e-9)
            height = PATCH_HIGH_PLATFORM_HEIGHTS_M[name]
            patch[np.ix_(inside, inside)] = height
            description.update(profile="centered square platform with vertical {:g} cm faces".format(height * 100),
                               platform_size_xy_m=[PATCH_HIGH_PLATFORM_SIZE_M] * 2,
                               platform_height_m=height,
                               platform_bounds_local_xy=[inset, self.patch_size_m - inset] * 2,
                               approach_length_inside_roi_m=inset)
        elif name == "rough":
            coarse = np.arange(int(math.ceil(self.patch_size_m / self.parameters["rough_cell_size"])) + 1) * self.parameters["rough_cell_size"]
            noise = rng.uniform(-self.parameters["rough_height"], self.parameters["rough_height"], (len(coarse), len(coarse)))
            first = np.array([np.interp(x, coarse, column) for column in noise.T]).T
            patch[:] = np.array([np.interp(x, coarse, row) for row in first])
            patch[[0, -1], :] = 0.
            patch[:, [0, -1]] = 0.
            description.update(profile="bilinearly interpolated roughness with zero perimeter ring",
                               coarse_spacing_m=self.parameters["rough_cell_size"],
                               nominal_height_range_m=[-self.parameters["rough_height"], self.parameters["rough_height"]],
                               boundary_transition_width_m=self.horizontal_scale)
        else:
            description["profile"] = "flat reference patch and flat surrounding support"
        quantized = np.rint(patch / self.vertical_scale) * self.vertical_scale
        description.update(min_height_m=float(quantized.min()), max_height_m=float(quantized.max()),
                           boundary_height_zero=bool(np.all(quantized[[0, -1], :] == 0)
                                                     and np.all(quantized[:, [0, -1]] == 0)))
        return patch, description

    def sample_spawns(self, type_ids, rng, direction=1, side="center", xy_jitter=0.):
        """Place the whole robot on an external pad along the ROI centerline.

        Offsets use the heading-zero body frame, even when driving backward.
        Returned positions are ground heights; the environment adds base z.
        Goal positions are on the other external pad, not metric stop triggers.
        """
        ids = np.asarray(type_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("type_ids must be a one-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= len(self.type_names)):
            raise ValueError("Unknown terrain index")
        signs = np.broadcast_to(np.asarray(direction), (len(ids),)).copy()
        if not np.all(np.isin(signs, (-1, 1))):
            raise ValueError("direction must be +1 or -1")
        requested_sides = np.broadcast_to(np.asarray(side), (len(ids),)).copy()
        if not np.all(np.isin(requested_sides, ("center", "left", "right"))):
            raise ValueError("side must be center (legacy left/right are centered aliases)")
        if not math.isfinite(xy_jitter) or xy_jitter < 0 or xy_jitter >= self.patch_size_m / 2:
            raise ValueError("xy_jitter must be nonnegative and smaller than half the patch width")
        bounds, patch_bounds = self.tile_bounds[ids].copy(), self.patch_bounds[ids].copy()
        offsets = np.zeros(len(ids))
        jitter = rng.uniform(-xy_jitter, xy_jitter, (len(ids), 2))
        pad_x = self.platform_length / 2
        start_x = np.where(signs > 0, bounds[:, 0] + pad_x, bounds[:, 1] - pad_x) + jitter[:, 0]
        goal_x = np.where(signs > 0, bounds[:, 1] - pad_x, bounds[:, 0] + pad_x)
        y = (patch_bounds[:, 2] + patch_bounds[:, 3]) / 2 + jitter[:, 1]
        front = self.wheel_geometry["wheel_x_max"]
        rear = self.wheel_geometry["wheel_x_min"]
        # Validate the complete longitudinal wheel envelope, not only base or
        # wheel centers. Default 1.5 m pads leave ample clearance with +/-4 cm.
        if (np.any(start_x + rear < bounds[:, 0]) or np.any(start_x + front > bounds[:, 1])
                or np.any(goal_x + rear < bounds[:, 0]) or np.any(goal_x + front > bounds[:, 1])
                or np.any(np.where(signs > 0, start_x + front >= patch_bounds[:, 0],
                                   start_x + rear <= patch_bounds[:, 1]))
                or np.any(np.where(signs > 0, goal_x + rear <= patch_bounds[:, 1],
                                   goal_x + front >= patch_bounds[:, 0]))):
            raise ValueError("External platforms are too short for the robot wheel envelope and spawn jitter")
        lateral_low = y + self.wheel_geometry["wheel_y_min"]
        lateral_high = y + self.wheel_geometry["wheel_y_max"]
        if np.any(lateral_low < patch_bounds[:, 2]) or np.any(lateral_high > patch_bounds[:, 3]):
            raise ValueError("The full wheel envelope and spawn jitter must fit inside the ROI width")
        start_xy, goal_xy = np.column_stack((start_x, y)), np.column_stack((goal_x, y))
        return {
            "ground_positions": np.column_stack((start_xy, self.ground_heights(start_xy))),
            "goal_positions": np.column_stack((goal_xy, self.ground_heights(goal_xy))),
            "directions": np.column_stack((signs, np.zeros(len(ids)))),
            "bounds": bounds, "patch_bounds": patch_bounds,
            "heading_yaw": np.zeros(len(ids)), "quaternions": np.tile([0., 0., 0., 1.], (len(ids), 1)),
            "side": ["center"] * len(ids), "travel_sign": signs.astype(np.int64),
            "wheel_track_offset_y": offsets,
        }
