"""Check the shipped training stair asset in physical MuJoCo coordinates."""

import hashlib
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np


class TrainingStairTests(unittest.TestCase):
    def test_every_stair_tile_has_35cm_riser_spacing_on_both_axes(self):
        scene = Path(__file__).resolve().parents[1] / "nezha/mjcf/scene_training.xml"
        metadata = json.loads((scene.parent / "assets/nezha_training_curriculum.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(scene))
        heights = model.hfield_data.reshape(model.hfield_nrow[0], model.hfield_ncol[0])
        self.assertEqual(hashlib.sha256(heights).hexdigest(), metadata["binary_data_sha256"])
        radii = model.hfield_size[0, :2]
        spacing = 2 * radii / (np.array(heights.shape[::-1]) - 1)
        lower = model.geom("training_terrain").pos[:2] - radii
        size = np.array([metadata["terrain_length_m"], metadata["terrain_width_m"]])
        columns = metadata["column_types"]
        self.assertIn("stairs_up", columns)
        self.assertIn("stairs_down", columns)
        for col, kind in enumerate(columns):
            if kind not in ("stairs_up", "stairs_down"):
                continue
            for row in range(metadata["num_rows"]):
                # Sample each side from its tile boundary to the central landing.
                lo = np.rint((size * [row, col] - lower) / spacing).astype(int)
                cx, cy = lo + np.rint(size / (2 * spacing)).astype(int)
                lines = [heights[cy, lo[0]:cx + 1], heights[lo[1]:cy + 1, cx]]
                for axis, line in enumerate(lines):
                    with self.subTest(kind=kind, row=row, col=col, axis=axis):
                        risers = np.flatnonzero(np.diff(line))
                        self.assertGreater(len(risers), 2)
                        np.testing.assert_allclose(np.diff(risers) * spacing[axis], .35)
