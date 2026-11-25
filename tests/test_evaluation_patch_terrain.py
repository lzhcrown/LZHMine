# Adapted from wheel_gym_CQ/tests/test_evaluation_patch_terrain.py
"""Verify square terrain ROIs, the narrow slope and their support geometry."""
import copy
import json
import math
from types import SimpleNamespace
import unittest

import isaacgym  # Import before torch, including when running the full suite.
import numpy as np

from legged_gym.evaluation.patch_terrain import PatchEvaluationTerrain
from legged_gym.evaluation.presets import (
    PATCH_TERRAIN_TYPES, PATCH_STAIR_TREAD_M, PATCH_STEPS_PER_SIDE, PATCH_STAIR_PLATFORM_M,
)


def config(**overrides):
    values = dict(mesh_type="heightfield", type_names=list(PATCH_TERRAIN_TYPES),
                  terrain_length=7.8, terrain_width=7.8, horizontal_scale=.02,
                  vertical_scale=.005, border_size=2., seed=10)
    values.update(overrides)
    return SimpleNamespace(**values)


def patch_grid(terrain, index):
    bounds = terrain.patch_bounds[index]
    lo = np.rint((bounds[[0, 2]] + terrain.cfg.border_size) / terrain.horizontal_scale).astype(int)
    hi = np.rint((bounds[[1, 3]] + terrain.cfg.border_size) / terrain.horizontal_scale).astype(int)
    return terrain.heightsamples[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]


class PatchTerrainTests(unittest.TestCase):
    def test_ten_exact_rois_and_support_corridors_do_not_modify_input(self):
        cfg = config()
        original = copy.deepcopy(vars(cfg))
        terrain = PatchEvaluationTerrain(cfg, 4096)
        self.assertEqual(vars(cfg), original)
        self.assertEqual(terrain.type_names, ("flat", "stairs_concave_35cm", "stairs_concave_25cm", "stairs_concave_30cm",
                                              "stairs_convex_35cm", "stairs_convex_25cm", "stairs_convex_30cm",
                                              "platform_45cm", "rough", "slope"))
        np.testing.assert_allclose(terrain.patch_bounds[:, 1] - terrain.patch_bounds[:, 0], [7.8] * 9 + [6.5])
        np.testing.assert_allclose(terrain.patch_bounds[:, 3] - terrain.patch_bounds[:, 2], [7.8] * 9 + [5.6])
        np.testing.assert_allclose(terrain.tile_bounds[:, 1] - terrain.tile_bounds[:, 0], [10.8] * 9 + [9.5])
        np.testing.assert_allclose(terrain.tile_bounds[:, 3] - terrain.tile_bounds[:, 2], [9.8] * 9 + [5.6])
        np.testing.assert_allclose(terrain.patch_bounds[:, 0] - terrain.tile_bounds[:, 0], 1.5)
        np.testing.assert_allclose(terrain.tile_bounds[:, 1] - terrain.patch_bounds[:, 1], 1.5)
        np.testing.assert_allclose(terrain.patch_bounds[:, 2] - terrain.tile_bounds[:, 2], [1.] * 9 + [0.])
        np.testing.assert_allclose(terrain.tile_bounds[:, 3] - terrain.patch_bounds[:, 3], [1.] * 9 + [0.])
        np.testing.assert_allclose(np.diff(terrain.tile_bounds[:-1, 2]), 11.8)
        self.assertIs(terrain.tile_bounds, terrain.corridor_bounds)
        self.assertEqual(terrain.env_origins.shape, (1, 10, 3))
        self.assertEqual(terrain.heightsamples.shape, (741, 6001))
        self.assertEqual(terrain.heightsamples.dtype, np.int16)
        self.assertEqual(terrain.metadata["protocol"], "wheel_window_patch")
        self.assertEqual(terrain.metadata["patch_size_m"], 7.8)
        np.testing.assert_allclose(terrain.metadata["corridor_lengths_m"], [10.8] * 9 + [9.5])
        self.assertEqual(terrain.metadata["type_seed_order"], list(PATCH_TERRAIN_TYPES))
        json.dumps(terrain.metadata, allow_nan=False)

    def test_concentric_stairs_have_eight_30cm_treads_and_three_metre_center(self):
        terrain = PatchEvaluationTerrain(config(), 7)
        self.assertEqual(PATCH_STAIR_TREAD_M, .3)
        self.assertEqual(PATCH_STEPS_PER_SIDE, 8)
        self.assertEqual(PATCH_STAIR_PLATFORM_M, 3.)
        self.assertEqual(terrain.metadata["parameters"]["step_width"], .3)
        self.assertEqual(terrain.metadata["parameters"]["steps_per_side"], 8)
        for index, sign, height in ((1, -1, .35), (2, -1, .25), (3, -1, .30),
                                    (4, 1, .35), (5, 1, .25), (6, 1, .30)):
            patch = patch_grid(terrain, index).astype(float) * terrain.vertical_scale
            self.assertEqual(patch.shape, (391, 391))
            np.testing.assert_allclose(patch, patch.T)
            np.testing.assert_allclose(patch, patch[::-1, :])
            np.testing.assert_allclose(patch, patch[:, ::-1])
            np.testing.assert_allclose(np.unique(patch), sorted(sign * height * np.arange(9)))
            # Sample tread interiors, avoiding the one-cell heightfield ramps.
            samples = zip((.15, .45, .75, 1.05, 1.35, 1.65, 1.95, 2.25, 3.9, 5.55, 5.85, 6.15, 6.45, 6.75, 7.05, 7.35, 7.65),
                          (0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0))
            for x, level in samples:
                self.assertAlmostEqual(patch[round(x / .02), 195], sign * height * level)
            # Corners depend on nearest y edge as well as x: this is not an
            # extruded 1-D stair profile.
            self.assertAlmostEqual(patch[195, 5], 0.)
            self.assertAlmostEqual(patch[195, 20], sign * height)
            self.assertAlmostEqual(patch[195, 110], sign * 7 * height)
            self.assertTrue(np.all(patch[[0, -1], :] == 0))
            self.assertTrue(np.all(patch[:, [0, -1]] == 0))
            centerline = patch[:, 195]
            changes = np.diff(centerline)
            self.assertEqual(np.count_nonzero(np.abs(changes) > 1e-9), 16)
            np.testing.assert_allclose(np.abs(changes[np.abs(changes) > 1e-9]), height)
            details = terrain.metadata["tiles"][index]
            self.assertEqual(details["step_height_interpretation"], "per riser")
            self.assertEqual(details["steps_per_side"], 8)
            self.assertEqual(details["tread_width_m"], .3)
            self.assertEqual(details["central_platform_size_m"], 3.)
            self.assertAlmostEqual(details["central_platform_height_m"], sign * 8 * height)
            self.assertAlmostEqual(details["total_height_or_depth_m"], 8 * height)
            self.assertEqual(details["terrace_inset_edges_m"], [.3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4])
            np.testing.assert_allclose(details["centerline_height_levels_m"],
                                       sign * height * np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

    def test_roughness_limits_coarse_scale_and_flat_reference(self):
        terrain = PatchEvaluationTerrain(config(), 7)
        self.assertFalse(patch_grid(terrain, 0).any())
        patch = patch_grid(terrain, 8) * terrain.vertical_scale
        self.assertGreater(np.std(patch), 0.)
        self.assertLessEqual(np.max(np.abs(patch)), .04)
        self.assertLess(patch.min(), 0.)
        self.assertGreater(patch.max(), 0.)
        self.assertTrue(np.all(patch[[0, -1], :] == 0))
        self.assertTrue(np.all(patch[:, [0, -1]] == 0))
        self.assertEqual(terrain.metadata["parameters"]["rough_cell_size"], .08)
        self.assertEqual(terrain.metadata["tiles"][8]["nominal_height_range_m"], [-.04, .04])

    def test_platforms_have_five_metre_square_tops_and_exact_vertical_mesh_faces(self):
        for height_cm in range(45, 111, 5):
            name = f"platform_{height_cm}cm"
            with self.subTest(name=name):
                self._check_platform_mesh(name, height_cm / 100)

    def _check_platform_mesh(self, name, height):
        terrain = PatchEvaluationTerrain(config(mesh_type="trimesh", type_names=[name]), 1)
        patch = patch_grid(terrain, 0) * terrain.vertical_scale
        np.testing.assert_allclose(np.unique(patch), [0., height])
        np.testing.assert_allclose(patch, patch.T)
        np.testing.assert_allclose(patch[70:321, 70:321], height)
        self.assertFalse(patch[:70].any())
        self.assertFalse(patch[321:].any())
        grid = terrain.vertices.reshape(terrain.tot_rows, terrain.tot_cols, 3)
        x0, x1, y0, y1 = terrain.patch_bounds[0]
        ix = round(((x0 + x1) / 2 + terrain.cfg.border_size) / terrain.horizontal_scale)
        iy = round(((y0 + y1) / 2 + terrain.cfg.border_size) / terrain.horizontal_scale)
        for line, axis, origin in ((grid[:, iy], 0, x0), (grid[ix, :], 1, y0)):
            faces = np.flatnonzero(np.abs(np.diff(line[:, 2])) > 1e-6)
            self.assertEqual(len(faces), 2)
            np.testing.assert_allclose(np.diff(line[:, 2])[faces], [height, -height])
            np.testing.assert_allclose(line[faces, axis] - terrain.cfg.border_size,
                                       [origin + 1.4, origin + 6.4], atol=2e-6)
            np.testing.assert_allclose(line[faces, axis], line[faces + 1, axis], atol=2e-6)
            self.assertAlmostEqual(float(line[faces[1], axis] - line[faces[0], axis]), 5., places=5)
        self.assertEqual(terrain.metadata["tiles"][0]["platform_size_xy_m"], [5., 5.])
        self.assertEqual(terrain.metadata["tiles"][0]["platform_height_m"], height)
        self.assertEqual(terrain.metadata["parameters"]["high_platform_height_m"], height)

    def test_narrow_slope_has_equal_width_end_pads_and_zero_ground_outside(self):
        terrain = PatchEvaluationTerrain(config(type_names=["slope"]), 1)
        patch = patch_grid(terrain, 0) * terrain.vertical_scale
        expected = np.rint(np.arange(326) * .02 * math.tan(math.radians(20.)) / .005) * .005
        np.testing.assert_allclose(patch[:, 140], expected)
        np.testing.assert_allclose(patch[:, 0], expected)
        np.testing.assert_allclose(patch[:, -1], expected)
        self.assertTrue(np.all(np.diff(expected) >= 0))
        self.assertAlmostEqual(expected[-1], 2.365)
        self.assertLessEqual(np.max(np.diff(expected)), .010 + 1e-9)
        self.assertEqual(patch.shape, (326, 281))
        self.assertAlmostEqual(terrain.patch_bounds[0, 3] - terrain.patch_bounds[0, 2], 5.6)
        # Both narrow pads and the longitudinal border continue the profile.
        x = np.array([-.5, .75, 1.5, 4.76, 8., 8.75, 9.5, 10.8, 11.3])
        profile = np.rint(np.clip(x - 1.5, 0., 6.5) * math.tan(math.radians(20.)) / .005) * .005
        y0, y1 = terrain.patch_bounds[0, 2:]
        for y in (y0 + .1, (y0 + y1) / 2, y1 - .1):
            np.testing.assert_allclose(terrain.ground_heights(np.column_stack((x, np.full(len(x), y)))), profile)
        for y in (y0 - .04, y1 + .04, .5, 9.3):
            np.testing.assert_allclose(terrain.ground_heights(np.column_stack((x, np.full(len(x), y)))), 0.)
        details = terrain.metadata["tiles"][0]
        self.assertEqual(details["low_x_platform_height_m"], 0.)
        self.assertAlmostEqual(details["high_x_platform_height_m"], 2.365)
        self.assertFalse(details["forced_return_to_zero"])
        self.assertEqual(details["lateral_profile_extension_m"], 0.)
        self.assertEqual(details["ramp_width_m"], 5.6)
        self.assertEqual(details["effective_ramp_length_m"], 6.5)
        self.assertEqual(details["patch_length_m"], 6.5)
        self.assertEqual(details["corridor_length_m"], 9.5)
        self.assertEqual(details["nominal_angle_deg"], 20.)
        self.assertAlmostEqual(details["nominal_height_difference_m"], 6.5 * math.tan(math.radians(20.)))

    def test_narrow_slope_mesh_edges_and_spawn_center_match_roi(self):
        terrain = PatchEvaluationTerrain(config(mesh_type="trimesh", type_names=["slope"]), 1)
        x0, x1, y0, y1 = terrain.patch_bounds[0]
        grid = terrain.vertices.reshape(terrain.tot_rows, terrain.tot_cols, 3)
        for x in ((x0 + x1) / 2, x1 + .75):
            ix = round((x + terrain.cfg.border_size) / terrain.horizontal_scale)
            line = grid[ix]
            edges = np.flatnonzero(np.abs(np.diff(line[:, 2])) > .1)
            self.assertEqual(len(edges), 2)
            np.testing.assert_allclose(line[edges, 1] - terrain.cfg.border_size, [y0, y1], atol=2e-6)
            np.testing.assert_allclose(line[edges, 1], line[edges + 1, 1], atol=2e-6)
            self.assertAlmostEqual(float(line[edges[1], 1] - line[edges[0], 1]), 5.6, places=5)
        for direction in (1, -1):
            spawn = terrain.sample_spawns(np.array([0]), np.random.default_rng(10), direction=direction)
            self.assertAlmostEqual(spawn["ground_positions"][0, 0], .75 if direction > 0 else 8.75)
            self.assertAlmostEqual(spawn["goal_positions"][0, 0], 8.75 if direction > 0 else .75)
            self.assertAlmostEqual(spawn["ground_positions"][0, 1], (y0 + y1) / 2)
            self.assertAlmostEqual(spawn["ground_positions"][0, 2], 0. if direction > 0 else 2.365)
        self.assertAlmostEqual(terrain.env_origins[0, 0, 0], (x0 + x1) / 2)
        self.assertAlmostEqual(terrain.env_origins[0, 0, 1], (y0 + y1) / 2)

    def test_centered_spawns_fit_whole_robot_for_both_directions_and_legacy_aliases(self):
        terrain = PatchEvaluationTerrain(config(), 112)
        ids = np.repeat(np.arange(10), 16)
        g = terrain.wheel_geometry
        for sign in (1, -1):
            plans = []
            for side in ("center", "left", "right"):
                plan = terrain.sample_spawns(ids, np.random.default_rng(18), direction=sign, side=side)
                plans.append(plan)
                start, goal, roi, bounds = (plan[key] for key in
                                           ("ground_positions", "goal_positions", "patch_bounds", "bounds"))
                middle_y = (roi[:, 2] + roi[:, 3]) / 2
                self.assertTrue(np.all(np.abs(start[:, 1] - middle_y) <= .04 + 1e-9))
                self.assertTrue(np.all(start[:, 1] + g["wheel_y_min"] > roi[:, 2]))
                self.assertTrue(np.all(start[:, 1] + g["wheel_y_max"] < roi[:, 3]))
                self.assertTrue(np.all(start[:, 0] + g["wheel_x_min"] >= bounds[:, 0]))
                self.assertTrue(np.all(start[:, 0] + g["wheel_x_max"] <= bounds[:, 1]))
                if sign > 0:
                    self.assertTrue(np.all(start[:, 0] + g["wheel_x_max"] < roi[:, 0]))
                    self.assertTrue(np.all(goal[:, 0] + g["wheel_x_min"] > roi[:, 1]))
                else:
                    self.assertTrue(np.all(start[:, 0] + g["wheel_x_min"] > roi[:, 1]))
                    self.assertTrue(np.all(goal[:, 0] + g["wheel_x_max"] < roi[:, 0]))
                np.testing.assert_array_equal(start[ids != 9, 2], 0.)
                np.testing.assert_allclose(start[ids == 9, 2], 0. if sign > 0 else 2.365)
                np.testing.assert_allclose(goal[ids == 9, 2], 2.365 if sign > 0 else 0.)
                np.testing.assert_array_equal(plan["directions"], np.tile([sign, 0.], (len(ids), 1)))
                np.testing.assert_array_equal(plan["quaternions"], np.tile([0., 0., 0., 1.], (len(ids), 1)))
                self.assertEqual(plan["side"], ["center"] * len(ids))
            for key in plans[0]:
                np.testing.assert_array_equal(plans[0][key], plans[1][key])
                np.testing.assert_array_equal(plans[0][key], plans[2][key])

    def test_type_seed_and_spawn_randomness_are_stable_under_selection_and_robot_count(self):
        terrain = PatchEvaluationTerrain(config(), 1)
        other = PatchEvaluationTerrain(config(), 500)
        np.testing.assert_array_equal(terrain.heightsamples, other.heightsamples)
        reordered = PatchEvaluationTerrain(config(type_names=["rough", "stairs_convex_35cm", "flat"]), 3)
        np.testing.assert_array_equal(patch_grid(terrain, 8), patch_grid(reordered, 0))
        np.testing.assert_array_equal(patch_grid(terrain, 4), patch_grid(reordered, 1))
        np.testing.assert_array_equal(patch_grid(terrain, 0), patch_grid(reordered, 2))
        ids = np.arange(10)
        first = terrain.sample_spawns(ids, np.random.default_rng(42), direction=[1, -1] * 5, xy_jitter=.04)
        second = terrain.sample_spawns(ids, np.random.default_rng(42), direction=[1, -1] * 5, xy_jitter=.04)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        changed = terrain.sample_spawns(ids, np.random.default_rng(43), xy_jitter=.04)
        self.assertFalse(np.array_equal(changed["ground_positions"], first["ground_positions"]))

    def test_default_start_is_exact_entry_midpoint_offset_for_every_seed_and_attempt(self):
        terrain = PatchEvaluationTerrain(config(), 7)
        ids = np.arange(10)
        for sign in (1, -1):
            reference = None
            for seed in (10, 100, 101):
                rng = np.random.default_rng(seed)
                for _ in range(3):
                    plan = terrain.sample_spawns(ids, rng, direction=sign)
                    start = plan["ground_positions"]
                    entry_x = terrain.patch_bounds[:, 0 if sign > 0 else 1]
                    np.testing.assert_allclose(start[:, 0], entry_x - sign * .75)
                    np.testing.assert_allclose(start[:, 1], terrain.patch_bounds[:, 2:4].mean(axis=1))
                    if reference is not None:
                        np.testing.assert_array_equal(start, reference)
                    reference = start

    def test_mesh_stair_faces_at_nominal_insets_and_no_slope_exit_cliff(self):
        terrain = PatchEvaluationTerrain(config(mesh_type="trimesh",
                                              type_names=["stairs_concave_25cm", "stairs_convex_30cm", "slope"]), 3)
        self.assertEqual(terrain.vertices.dtype, np.float32)
        self.assertEqual(terrain.triangles.dtype, np.uint32)
        self.assertEqual(terrain.vertices.shape, (terrain.tot_rows * terrain.tot_cols, 3))
        self.assertEqual(terrain.triangles.shape, (2 * (terrain.tot_rows - 1) * (terrain.tot_cols - 1), 3))
        np.testing.assert_allclose(terrain.vertices[:, 2], terrain.heightsamples.ravel() * .005)
        self.assertLess(terrain.triangles.max(), len(terrain.vertices))
        grid = terrain.vertices.reshape(terrain.tot_rows, terrain.tot_cols, 3)
        for index in (0, 1):
            x0, x1, y0, y1 = terrain.patch_bounds[index]
            iy = round(((y0 + y1) / 2 + terrain.cfg.border_size) / terrain.horizontal_scale)
            line = grid[:, iy]
            changed = np.flatnonzero(np.abs(np.diff(line[:, 2])) > 1e-6)
            self.assertEqual(len(changed), 16)
            nominal_faces = x0 + np.array([.3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4, 5.4, 5.7, 6., 6.3, 6.6, 6.9, 7.2, 7.5])
            np.testing.assert_allclose(line[changed, 0] - terrain.cfg.border_size, nominal_faces, atol=2e-6)
            np.testing.assert_allclose(line[changed, 0], line[changed + 1, 0], atol=2e-6)
            self.assertAlmostEqual(float(line[changed[8], 0] - line[changed[7], 0]), 3., places=6)
        x0, x1, y0, y1 = terrain.patch_bounds[2]
        ix = round((x1 + terrain.cfg.border_size) / terrain.horizontal_scale)
        iy = round(((y0 + y1) / 2 + terrain.cfg.border_size) / terrain.horizontal_scale)
        np.testing.assert_allclose(grid[ix:, iy, 2], 2.365, atol=1e-6)

    def test_explicit_wheel_envelope_metadata_and_unsafe_routes(self):
        terrain = PatchEvaluationTerrain(config(type_names=["flat"], left_wheel_y=.24, right_wheel_y=-.27,
                                              front_wheel_x=.48, rear_wheel_x=-.50, wheel_radius=.12,
                                              wheel_x_min=-.63, wheel_x_max=.62,
                                              wheel_y_min=-.31, wheel_y_max=.29,
                                              wheel_geometry_source="URDF/test_asset.urdf"), 1)
        self.assertEqual(terrain.metadata["wheel_geometry_fallback_fields"], [])
        self.assertEqual(terrain.metadata["wheel_geometry_source"], "URDF/test_asset.urdf")
        self.assertEqual(terrain.wheel_geometry["wheel_y_min"], -.31)
        plan = terrain.sample_spawns(np.array([0]), np.random.default_rng(1), xy_jitter=0.)
        np.testing.assert_allclose(plan["ground_positions"][0], [.75, 4.9, 0.])
        for changes in (dict(wheel_x_min=-1.), dict(wheel_y_min=-7.)):
            unsafe = PatchEvaluationTerrain(config(type_names=["flat"], **changes), 1)
            with self.assertRaises(ValueError):
                unsafe.sample_spawns(np.array([0]), np.random.default_rng(1), xy_jitter=0.)

    def test_old_patch_types_dimensions_and_incompatible_grids_are_rejected(self):
        for changes in (dict(terrain_length=.4), dict(terrain_length=3.), dict(terrain_width=40.), dict(patch_size_m=.4),
                        dict(horizontal_scale=.03), dict(type_names=[]), dict(type_names=["stairs_up"]), dict(type_names=["stairs_convex_15cm"]), dict(type_names=["stairs_convex_20cm"]),
                        dict(type_names=["stairs_concave_20cm"]), dict(type_names=["platform_115cm"]),
                        dict(type_names=["platform_50cm"], vertical_scale=.03),
                        dict(type_names=["discrete_obstacles"]), dict(step_width=.4), dict(step_width=.2), dict(seed=-1),
                        dict(vertical_scale=.03), dict(platform_length=.5), dict(corridor_width=4.), dict(slope_width_m=0),
                        dict(slope_width_m=1.51), dict(slope_width_m=10.), dict(slope_length_m=0),
                        dict(slope_length_m=6.51), dict(slope_length_m=8.)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PatchEvaluationTerrain(config(**changes), 1)
        terrain = PatchEvaluationTerrain(config(type_names=["flat"]), 1)
        for kwargs in (dict(direction=0), dict(side="both"), dict(xy_jitter=3.9)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                terrain.sample_spawns(np.array([0]), np.random.default_rng(1), **kwargs)


if __name__ == "__main__":
    unittest.main()
