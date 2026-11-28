"""Cross-check the deployed student's scan against physical terrain and Isaac's sampling convention."""

from pathlib import Path
import unittest

import mujoco
import numpy as np

from deploy.deploy_mujoco.runtime.terrain_scan import TerrainHeightScanner


class TerrainScanTest(unittest.TestCase):
    def test_heightfield_matches_the_isaac_three_vertex_minimum_on_a_tilted_robot(self):
        model = mujoco.MjModel.from_xml_string("""
            <mujoco>
              <asset><hfield name="land" nrow="31" ncol="41" size="2 1.5 1.8 0.1"/></asset>
              <worldbody><geom type="hfield" hfield="land" pos="1 2 -.7"/></worldbody>
            </mujoco>""")
        samples = np.random.default_rng(3).uniform(size=(31, 41)).astype(np.float32)
        model.hfield_data[:] = samples.flatten()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sensor = TerrainHeightScanner(mujoco, model, .5)
        position = np.array([.971, 2.023, .83])
        quat = np.array([.6, -.3, .4, np.sqrt(.39)])
        yaw = 2 * np.arctan2(quat[3], quat[0])
        expected = []
        for y_body in np.arange(-5, 6) / 10:
            for x_body in np.arange(-8, 9) / 10:
                x = position[0] + np.cos(yaw) * x_body - np.sin(yaw) * y_body
                y = position[1] + np.sin(yaw) * x_body + np.cos(yaw) * y_body
                ix, iy = int((x + 1.) / .1), int((y - .5) / .1)
                height = -.7 + 1.8 * min(samples[iy, ix], samples[iy + 1, ix], samples[iy, ix + 1])
                expected.append(position[2] - .5 - height)
        np.testing.assert_allclose(sensor.scan(data, position, quat), expected, atol=1e-6)

    def test_scan_reads_the_actual_course_boxes_and_changes_when_robot_moves(self):
        path = Path(__file__).resolve().parents[1] / "nezha/mjcf/scene_course.xml"
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sensor = TerrainHeightScanner(mujoco, model, .5)
        for x, expected_surface in ((-.5, 0.), (1.16, .15), (1.47, .30), (2.71, .75)):
            scan = sensor.scan(data, [x, -15., .6], [1., 0., 0., 0.])
            self.assertEqual(scan.shape, (187,))
            self.assertAlmostEqual(float(scan[5 * 17 + 8]), .6 - .5 - expected_surface, places=6)
