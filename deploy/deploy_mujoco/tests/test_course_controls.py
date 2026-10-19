"""Lane switching must stop motion and reuse a safe, flat spawn."""

import unittest

import numpy as np

from deploy.deploy_mujoco.deploy_mujoco import (
    KeyboardCommandController,
    _terrain_lane_spawn,
)


class CourseControlsTest(unittest.TestCase):
    def test_lane_switch_stops_commands_and_is_consumed_once(self):
        controller = KeyboardCommandController(
            [0.5, 0.2, 0.3], terrain_names=("stairs_15cm", "ramp_25deg")
        )
        controller.on_key(ord("2"))
        np.testing.assert_array_equal(controller.get(), [0.0, 0.0, 0.0])
        self.assertEqual(controller.consume_reset_request(), (True, 1))
        self.assertEqual(controller.consume_reset_request(), (False, None))

    def test_reset_keeps_current_lane_and_cancels_pending_selection(self):
        controller = KeyboardCommandController(
            [0.5, 0.0, 0.0], terrain_names=("stairs_15cm", "ramp_25deg")
        )
        controller.on_key(ord("2"))
        controller.on_key(ord("R"))
        self.assertEqual(controller.consume_reset_request(), (True, None))
        np.testing.assert_array_equal(controller.get(), [0.0, 0.0, 0.0])

    def test_unavailable_lane_and_non_character_keys_are_ignored(self):
        controller = KeyboardCommandController([0.5, 0.0, 0.0])
        for key in (ord("1"), ord("9"), 300, -1):
            controller.on_key(key)
        self.assertEqual(controller.consume_reset_request(), (False, None))
        np.testing.assert_allclose(controller.get(), [0.5, 0.0, 0.0])

    def test_lane_spawn_stays_on_staging_area_without_mutating_metadata(self):
        description = {
            "spawn": np.array([-5.0, -15.0, 0.60]),
            "lanes": [
                {"name": "stairs_15cm", "y_center_m": -15.0},
                {"name": "ramp_25deg", "y_center_m": 5.0},
            ],
        }
        position = _terrain_lane_spawn(description, "ramp_25deg")
        np.testing.assert_allclose(position, [-5.0, 5.0, 0.60])
        np.testing.assert_allclose(description["spawn"], [-5.0, -15.0, 0.60])
        with self.assertRaisesRegex(ValueError, "Unknown terrain"):
            _terrain_lane_spawn(description, "missing")
        with self.assertRaisesRegex(ValueError, "requires a scene"):
            _terrain_lane_spawn(None, "stairs_15cm")


if __name__ == "__main__":
    unittest.main()
