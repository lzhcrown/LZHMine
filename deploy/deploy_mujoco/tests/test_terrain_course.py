"""Check the generated course's collision geometry with the real MuJoCo engine.

Run from the repository root with::

    .venv-mujoco/bin/python -m unittest discover -s deploy/deploy_mujoco/tests -v

The probe model includes the scene's actual terrain assets and geoms. It omits
the robot and adds a sphere with the robot's collision/friction parameters, so
these tests do not require a policy, Torch, Isaac Gym, or an OpenGL display.
"""

import copy
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


SCENE_PATH = Path(__file__).resolve().parents[1] / "nezha/mjcf/scene_course.xml"
PROBE_RADIUS_M = 0.10


def _terrain_scene_element(path):
    """Expand actual terrain includes while preserving their file locations."""
    root = ET.parse(path).getroot()

    def expand(parent):
        for child in list(parent):
            if child.tag == "include":
                include_path = (path.parent / child.attrib["file"]).resolve()
                position = list(parent).index(child)
                parent.remove(child)
                if include_path.name == "nezha.xml":
                    continue
                included = _terrain_scene_element(include_path)
                for offset, element in enumerate(included):
                    parent.insert(position + offset, element)
            else:
                if "file" in child.attrib:
                    child.set("file", str((path.parent / child.attrib["file"]).resolve()))
                expand(child)

    expand(root)
    return root


class TerrainCourseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = copy.deepcopy(_terrain_scene_element(SCENE_PATH))
        # The omitted robot include defines radians in the deployment model.
        compiler = scene.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(scene, "compiler")
        compiler.set("angle", "radian")
        worldbody = scene.find("worldbody")
        if worldbody is None:
            raise AssertionError("Terrain scene does not contain a worldbody")
        body = ET.SubElement(worldbody, "body", name="contact_probe", pos="-9 -19 1")
        ET.SubElement(body, "freejoint", name="contact_probe_joint")
        ET.SubElement(
            body, "geom", name="contact_probe_geom", type="sphere",
            size=str(PROBE_RADIUS_M), density="1000", group="3",
            contype="1", conaffinity="2", condim="3", priority="0",
            friction="0.6 0.01 0.001", solref="0.005 1",
        )
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
        cls.probe_id = mujoco.mj_name2id(
            cls.model, mujoco.mjtObj.mjOBJ_GEOM, "contact_probe_geom"
        )

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def surface_height(self, x, y):
        group_mask = np.ones(6, dtype=np.uint8)
        group_mask[3] = 0  # Exclude the probe, leaving the actual terrain.
        hit_geom = np.full(1, -1, dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model, self.data, np.array([x, y, 10.0]),
            np.array([0.0, 0.0, -1.0]), group_mask, True, -1, hit_geom,
        )
        self.assertGreaterEqual(distance, 0.0, f"No terrain at {(x, y)}")
        self.assertNotEqual(int(self.model.geom_contype[hit_geom[0]]), 0)
        return 10.0 - distance

    def assert_surface_contact(self, point, normal, expected_friction, geom_name=None):
        normal = np.asarray(normal, dtype=float)
        self.data.qpos[:3] = np.asarray(point) + (PROBE_RADIUS_M - 0.001) * normal
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        contacts = [
            contact for contact in self.data.contact
            if self.probe_id in (contact.geom1, contact.geom2)
        ]
        self.assertTrue(contacts, f"No physical contact at {point}")
        for contact in contacts:
            np.testing.assert_allclose(contact.friction[:2], expected_friction, atol=1e-12)
            self.assertEqual(contact.dim, 3)
            if geom_name is not None:
                other = contact.geom2 if contact.geom1 == self.probe_id else contact.geom1
                actual_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                self.assertEqual(actual_name, geom_name,
                                 "Unexpected overlapping contact surface")

    def test_staging_runout_and_side_margin_are_flat(self):
        for y in (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0):
            for x in (-5.0, 12.0):
                with self.subTest(x=x, y=y):
                    self.assertAlmostEqual(self.surface_height(x, y), 0.0, places=7)
        self.assertAlmostEqual(self.surface_height(2.0, -18.0), 0.0, places=7)
        self.assertAlmostEqual(self.surface_height(2.0, 18.0), 0.0, places=7)

    def test_stairs_have_requested_risers_and_30cm_treads(self):
        # Independent physical sample points on all nine treads/top surfaces.
        centers = (1.15, 1.45, 1.75, 2.05, 2.70, 3.35, 3.65, 3.95, 4.25)
        levels = (1, 2, 3, 4, 5, 4, 3, 2, 1)
        for height, y in ((0.15, -15.0), (0.20, -10.0), (0.25, -5.0)):
            with self.subTest(step_height=height):
                for x, level in zip(centers, levels):
                    self.assertAlmostEqual(self.surface_height(x, y), height * level, places=6)
                # Probe closely on both sides of each riser: heightfields would
                # turn these jumps into sloped 5 cm strips and fail this check.
                for x in (1.0, 1.3, 1.6, 1.9, 2.2):
                    rise = self.surface_height(x + 0.0001, y) - self.surface_height(x - 0.0001, y)
                    self.assertAlmostEqual(rise, height, places=6)
                for x in (3.2, 3.5, 3.8, 4.1, 4.4):
                    fall = self.surface_height(x - 0.0001, y) - self.surface_height(x + 0.0001, y)
                    self.assertAlmostEqual(fall, height, places=6)
                # Preserve the original 5 m cross-track length of each step.
                for side_y in (y - 2.49, y + 2.49):
                    self.assertAlmostEqual(self.surface_height(1.15, side_y), height, places=6)
                self.assertAlmostEqual(self.surface_height(2.2001, y), 5 * height, places=6)
                self.assertAlmostEqual(self.surface_height(3.1999, y), 5 * height, places=6)

    def test_ramps_have_exact_requested_angles_and_one_metre_flat_top(self):
        for angle, y in ((15, 0.0), (25, 5.0), (30, 10.0)):
            with self.subTest(angle=angle):
                tangent = math.tan(math.radians(angle))
                for x in (1.5, 2.5, 3.5, 4.25):
                    self.assertAlmostEqual(self.surface_height(x, y), (x - 1.0) * tangent, places=6)
                for side_y in (y - 2.49, y + 2.49):
                    self.assertAlmostEqual(self.surface_height(2.5, side_y), 1.5 * tangent, places=6)
                measured_angle = math.degrees(math.atan(
                    self.surface_height(3.5, y) - self.surface_height(2.5, y)
                ))
                self.assertAlmostEqual(measured_angle, angle, places=5)
                for x in (4.5001, 5.0, 5.4999):
                    self.assertAlmostEqual(self.surface_height(x, y), 3.5 * tangent, places=6)
                for x in (6.0, 7.0, 8.0, 8.75):
                    self.assertAlmostEqual(self.surface_height(x, y), (9.0 - x) * tangent, places=6)
                measured_descent = math.degrees(math.atan(
                    self.surface_height(7.0, y) - self.surface_height(8.0, y)
                ))
                self.assertAlmostEqual(measured_descent, angle, places=5)
                self.assertAlmostEqual(self.surface_height(0.9, y), 0.0, places=7)
                self.assertAlmostEqual(self.surface_height(9.1, y), 0.0, places=7)

    def test_ramp_friction_is_low_in_real_contacts_without_duplicate_ground(self):
        for angle, y in ((15, 0.0), (25, 5.0), (30, 10.0)):
            radians = math.radians(angle)
            for part, x, normal in (
                ("up", 2.5, (-math.sin(radians), 0.0, math.cos(radians))),
                ("platform", 5.0, (0.0, 0.0, 1.0)),
                ("down", 7.5, (math.sin(radians), 0.0, math.cos(radians))),
            ):
                with self.subTest(angle=angle, part=part):
                    z = self.surface_height(x, y)
                    self.assert_surface_contact(
                        (x, y, z), normal, 0.35, f"ramp_{angle}deg_{part}"
                    )

    def test_other_terrain_retains_normal_contact_friction(self):
        self.assert_surface_contact((-5.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, "training_terrain")
        self.assert_surface_contact((1.45, -15.0, 0.30), (0.0, 0.0, 1.0), 1.0)
        self.assert_surface_contact((5.5, 15.0, 0.20), (0.0, 0.0, 1.0), 1.0, "training_terrain")

    def test_speed_bumps_reach_20cm_and_extend_across_the_lane(self):
        self.assertEqual(self.model.nhfield, 1)
        shape = (int(self.model.hfield_nrow[0]), int(self.model.hfield_ncol[0]))
        samples = self.model.hfield_data.reshape(shape)
        row, col = np.unravel_index(int(samples.argmax()), shape)
        terrain_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "training_terrain")
        center = self.model.geom_pos[terrain_id]
        half_x, half_y, scale_z, _ = self.model.hfield_size[0]
        peak_x = center[0] - half_x + col * 2.0 * half_x / (shape[1] - 1)
        peak_y = center[1] - half_y + row * 2.0 * half_y / (shape[0] - 1)
        self.assertGreaterEqual(peak_y, 12.5)
        self.assertLessEqual(peak_y, 17.5)
        self.assertAlmostEqual(float(samples.max()) * scale_z, 0.20, places=6)
        self.assertAlmostEqual(self.surface_height(peak_x, peak_y), 0.20, places=6)
        # Tall features must cross the lane rather than being isolated rocks.
        x_samples = np.linspace(0.5, 9.5, 181)
        for y in (13.0, 14.0, 15.0, 16.0, 17.0):
            with self.subTest(y=y):
                heights = np.array([self.surface_height(x, y) for x in x_samples])
                self.assertGreater(heights.max(), 0.12)
                self.assertGreater(np.count_nonzero(heights > 0.10), 10)


if __name__ == "__main__":
    unittest.main()
