# Adapted from wheel_gym_CQ/tests/test_evaluation_crossing.py
"""Pure CPU wheel-projection cases, including the gap between both axles."""
import unittest

import torch

from legged_gym.evaluation.crossing import WheelWindowCrossing


def wheel_bounds(front_x, rear_x, *, side=0, half_length=.1, other_rear_x=None):
    centers_x = [front_x, front_x, rear_x, rear_x]
    if other_rear_x is not None:
        centers_x[3 if side == 0 else 2] = other_rear_x
    centers_y = [0., -.6, 0., -.6] if side == 0 else [.6, 0., .6, 0.]
    return torch.tensor([[[x - half_length, x + half_length, y - .03, y + .03]
                          for x, y in zip(centers_x, centers_y)]], dtype=torch.float64)


def tracker(direction=1, side=0):
    state = WheelWindowCrossing(1, "cpu")
    initial = wheel_bounds(-.4, -1.2, side=side) if direction == 1 else wheel_bounds(2., 1.2, side=side)
    state.reset([0], window_bounds=[[0., .4, -.2, .2]], travel_direction=direction,
                selected_side=side, wheel_bounds=initial)
    return state


def wide_wheel_bounds(front_x, rear_x, other_rear_x=None):
    bounds = wheel_bounds(front_x, rear_x, other_rear_x=other_rear_x)
    bounds[:, [0, 2], 2:] = torch.tensor([.22, .28], dtype=torch.float64)
    bounds[:, [1, 3], 2:] = torch.tensor([-.28, -.22], dtype=torch.float64)
    return bounds


class EvaluationCrossingTests(unittest.TestCase):
    def test_three_metre_all_wheel_mode_waits_for_last_wheel_in_each_direction(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                state = WheelWindowCrossing(1, "cpu")
                initial = wide_wheel_bounds(-.4, -1.2) if direction == 1 else wide_wheel_bounds(4.2, 3.4)
                state.reset([0], window_bounds=[[0., 3., -1.5, 1.5]], travel_direction=direction,
                            selected_side=2, wheel_bounds=initial)
                self.assertTrue(state.selected_pair_mask.all())
                entry = wide_wheel_bounds(-.05, -.85) if direction == 1 else wide_wheel_bounds(3.85, 3.05)
                event = state.update(entry, .02)
                self.assertTrue(event["entered"][0])
                self.assertEqual(state.leading_axle[0], 0 if direction == 1 else 1)
                inside = wide_wheel_bounds(1., .2) if direction == 1 else wide_wheel_bounds(2.8, 2.)
                state.update(inside, .04)
                self.assertTrue(state.wheels_entered_mask.all())
                before_clear = (wide_wheel_bounds(3.9, 3.1, other_rear_x=3.05) if direction == 1
                                else wide_wheel_bounds(.05, -.75))
                event = state.update(before_clear, .06)
                self.assertFalse(event["exited"][0] or event["failed"][0])
                final = wide_wheel_bounds(3.9, 3.1001) if direction == 1 else wide_wheel_bounds(-.1001, -.9001)
                event = state.update(final, .08)
                self.assertTrue(event["exited"][0])
                self.assertFalse(event["failed"][0])
                self.assertEqual(state.entry_time_s[0], .02)
                self.assertEqual(state.exit_time_s[0], .08)

    def test_all_wheel_mode_rejects_any_single_wheel_bypassing_the_entrance(self):
        for skipped in range(4):
            with self.subTest(skipped_wheel=skipped):
                def bounds(front, rear):
                    result = wide_wheel_bounds(front, rear)
                    result[0, skipped, 2:] = torch.tensor([2., 2.06], dtype=torch.float64)
                    return result
                state = WheelWindowCrossing(1, "cpu")
                state.reset([0], window_bounds=[[0., 3., -1.5, 1.5]], travel_direction=1,
                            selected_side=2, wheel_bounds=bounds(-.4, -1.2))
                state.update(bounds(-.05, -.85), .02)
                state.update(bounds(1., .2), .04)
                event = state.update(bounds(3.9, 3.1), .06)
                self.assertFalse(event["exited"][0])
                self.assertTrue(event["failed"][0])
                self.assertEqual(event["reason_code"][0], 1)
                self.assertFalse(state.wheels_entered_mask[0, skipped])
                self.assertEqual(int(state.wheels_entered_mask.sum()), 3)

    def test_all_wheel_mode_checks_opposite_side_wheel_at_the_exit_plane(self):
        state = WheelWindowCrossing(1, "cpu")
        state.reset([0], window_bounds=[[0., 3., -1.5, 1.5]], travel_direction=1,
                    selected_side=2, wheel_bounds=wide_wheel_bounds(-.4, -1.2))
        state.update(wide_wheel_bounds(-.05, -.85), .02)
        state.update(wide_wheel_bounds(1., .2), .04)
        state.update(wide_wheel_bounds(2., 1.2), .06)
        final = wide_wheel_bounds(3.9, 3.1)
        final[0, 1, 2:] += 4.
        event = state.update(final, .08)
        self.assertTrue(event["failed"][0])
        self.assertEqual(event["reason_code"][0], 1)
        self.assertFalse(event["exited"][0])

    def test_forward_crossing_keeps_one_window_through_wheel_free_gap(self):
        state = tracker()
        event = state.update(wheel_bounds(-.05, -.85), .02)
        self.assertTrue(event["entered"][0])
        self.assertEqual(state.leading_axle[0], 0)
        self.assertEqual(state.wheels_entered_mask.tolist(), [[True, False, False, False]])
        # Front wheels have left; rear wheels have not reached the entry.
        event = state.update(wheel_bounds(.65, -.15), .04)
        self.assertTrue(state.window_started[0])
        self.assertTrue(state.active[0])
        self.assertFalse(event["entered"][0] or event["exited"][0] or event["failed"][0])
        self.assertEqual(state.entry_time_s[0], .02)
        state.update(wheel_bounds(1., .2), .06)
        event = state.update(wheel_bounds(1.4, .6), .08)
        self.assertTrue(event["exited"][0])
        self.assertFalse(state.active[0])
        self.assertEqual(state.exit_time_s[0], .08)
        self.assertEqual(state.wheels_entered_mask.tolist(), [[True, False, True, False]])
        # Airborne projections are valid: no contact observation was required.
        self.assertFalse(state.wheel_contact_mask.any())
        event = state.update(wheel_bounds(2., 1.2), .10)
        self.assertFalse(event["entered"].any() or event["exited"].any() or event["failed"].any())

    def test_backward_crossing_enters_rear_axle_first(self):
        state = tracker(direction=-1, side=1)
        event = state.update(wheel_bounds(1.25, .45, side=1), .02)
        self.assertTrue(event["entered"][0])
        self.assertEqual(state.leading_axle[0], 1)
        self.assertEqual(state.wheels_entered_mask.tolist(), [[False, False, False, True]])
        event = state.update(wheel_bounds(.55, -.25, side=1), .04)
        self.assertTrue(state.active[0])
        self.assertFalse(event["failed"][0] or event["exited"][0])
        state.update(wheel_bounds(.2, -.6, side=1), .06)
        event = state.update(wheel_bounds(-.2, -1., side=1), .08)
        self.assertTrue(event["exited"][0])
        self.assertEqual(state.wheels_entered_mask.tolist(), [[False, True, False, True]])

    def test_exit_waits_for_opposite_side_last_wheel(self):
        state = tracker()
        state.update(wheel_bounds(-.05, -.85), .02)
        state.update(wheel_bounds(1., .2), .04)
        event = state.update(wheel_bounds(1.4, .6, other_rear_x=.45), .06)
        self.assertFalse(event["exited"][0])
        self.assertTrue(state.active[0])
        event = state.update(wheel_bounds(1.4, .6, other_rear_x=.5), .08)
        self.assertTrue(event["exited"][0])

    def test_selected_wheel_side_escape_and_reverse_retreat_fail(self):
        state = tracker()
        state.update(wheel_bounds(-.05, -.85), .02)
        escaped = wheel_bounds(.1, -.7)
        escaped[0, 0, 2:] += 1.
        event = state.update(escaped, .04)
        self.assertTrue(event["failed"][0])
        self.assertEqual(event["reason_code"][0], 1)
        state = tracker()
        state.update(wheel_bounds(-.05, -.85), .02)
        event = state.update(wheel_bounds(-.3, -1.1), .04)
        self.assertTrue(event["failed"][0])
        self.assertEqual(event["reason_code"][0], 2)

    def test_following_wheel_can_align_laterally_before_entering(self):
        state = tracker()
        approaching = wheel_bounds(-.05, -.85)
        approaching[0, 2, 2:] += 1.
        event = state.update(approaching, .02)
        self.assertTrue(event["entered"][0])
        self.assertFalse(event["failed"][0])
        gap = wheel_bounds(.65, -.15)
        gap[0, 2, 2:] += 1.
        event = state.update(gap, .04)
        self.assertFalse(event["failed"][0])
        # Rear wheel recovers its lateral alignment while still before entry.
        event = state.update(wheel_bounds(.65, -.15), .06)
        self.assertFalse(event["failed"][0])
        state.update(wheel_bounds(1., .2), .08)
        event = state.update(wheel_bounds(1.4, .6), .10)
        self.assertTrue(event["exited"][0])
        self.assertFalse(event["failed"][0])

    def test_side_escape_during_final_exit_interval_is_not_success(self):
        for direction, side in ((1, 0), (-1, 1)):
            with self.subTest(direction=direction):
                state = tracker(direction=direction, side=side)
                if direction == 1:
                    state.update(wheel_bounds(-.05, -.85), .02)
                    state.update(wheel_bounds(1., .2), .04)
                    final = wheel_bounds(1.4, .6)
                    final[0, 2, 2:] += 1.
                else:
                    state.update(wheel_bounds(1.25, .45, side=1), .02)
                    state.update(wheel_bounds(.2, -.6, side=1), .04)
                    final = wheel_bounds(-.2, -1., side=1)
                    final[0, 1, 2:] += 1.
                event = state.update(final, .06)
                self.assertTrue(event["failed"][0])
                self.assertEqual(event["reason_code"][0], 1)
                self.assertFalse(event["exited"][0])

    def test_block_between_wheels_is_unstarted_bypass_not_success(self):
        state = WheelWindowCrossing(1, "cpu")
        initial = wheel_bounds(-.4, -1.2)
        initial[:, [0, 2], 2:] = torch.tensor([.24, .30], dtype=torch.float64)
        state.reset([0], window_bounds=[[0., .4, -.2, .2]], travel_direction=1,
                    selected_side=0, wheel_bounds=initial)
        final = initial.clone()
        final[:, :, :2] += 2.
        event = state.update(final, .02)
        self.assertTrue(event["failed"][0])
        self.assertEqual(event["reason_code"][0], 1)
        self.assertFalse(event["window_started"][0])
        self.assertFalse(event["exited"][0])

    def test_lateral_crossing_must_overlap_at_entrance_not_only_after_exit(self):
        state = WheelWindowCrossing(1, "cpu")
        initial = wheel_bounds(-.4, -1.2)
        initial[:, [0, 2], 2:] += 2.
        state.reset([0], window_bounds=[[0., .4, -.2, .2]], travel_direction=1,
                    selected_side=0, wheel_bounds=initial)
        event = state.update(wheel_bounds(1.6, .8), .02)
        self.assertTrue(event["failed"][0])
        self.assertFalse(event["window_started"][0])

    def test_contacts_are_observations_separate_from_geometric_entry(self):
        state = tracker()
        event = state.update(wheel_bounds(-.05, -.85), .02, [[True, True, True, True]])
        self.assertEqual(event["wheel_contact_mask"].tolist(), [[True, False, False, False]])
        event = state.update(wheel_bounds(1., .2), .04, [[False, False, True, True]])
        self.assertEqual(event["wheel_contact_mask"].tolist(), [[True, False, True, False]])

    def test_reset_clears_previous_episode_and_handles_same_step_crossing(self):
        state = tracker()
        event = state.update(wheel_bounds(1.6, .8), .02)
        self.assertTrue(event["entered"][0] and event["exited"][0])
        self.assertEqual(state.entry_time_s[0], state.exit_time_s[0])
        state.reset([0], window_bounds=[[0., .4, -.2, .2]], travel_direction=1,
                    selected_side=0, wheel_bounds=wheel_bounds(-.4, -1.2))
        self.assertFalse(state.window_started[0] or state.wheels_entered_mask.any())
        self.assertTrue(torch.isnan(state.entry_time_s[0]))
        self.assertEqual(state.leading_axle[0], -1)
        event = state.update(wheel_bounds(-.3, -1.1), .02)
        self.assertFalse(event["entered"][0])

    def test_invalid_geometry_and_initial_state_are_rejected(self):
        state = tracker()
        invalid = wheel_bounds(-.05, -.85)
        invalid[0, 0, 0] = float("nan")
        event = state.update(invalid, .02)
        self.assertTrue(event["failed"][0])
        self.assertEqual(event["reason_code"][0], 3)
        with self.assertRaises(ValueError):
            state.reset([0], window_bounds=[[0., .4, -.2, .2]], travel_direction=1,
                        selected_side=0, wheel_bounds=wheel_bounds(.1, -.7))


class PassageExitTests(unittest.TestCase):
    def tracker(self, direction=1):
        state = WheelWindowCrossing(1, "cpu", exit_rule="opposite_or_sides")
        state.reset([0], window_bounds=[[0., 3., -1.5, 1.5]], travel_direction=direction,
                    selected_side=2, wheel_bounds=wide_wheel_bounds(-.4, -1.2) if direction == 1
                    else wide_wheel_bounds(4.2, 3.4))
        return state

    def test_both_lateral_edges_work_in_both_directions_after_last_wheel_clears(self):
        for direction in (1, -1):
            for side in (-1, 1):
                state = self.tracker(direction)
                inside = wide_wheel_bounds(1.3, .5)
                state.update(inside, .02)
                partial = inside.clone()
                partial[..., 2:] += side * 1.4
                event = state.update(partial, .04)
                self.assertFalse(event["exited"][0])
                final = inside.clone()
                final[..., 2:] += side * 2.
                event = state.update(final, .06)
                self.assertTrue(event["exited"][0])
                self.assertEqual(int(state.exit_edge_code[0]), 3 if side < 0 else 4)

    def test_original_entrance_is_excluded_and_retreat_allows_a_new_attempt_to_exit(self):
        for direction in (1, -1):
            state = self.tracker(direction)
            state.update(wide_wheel_bounds(1.3, .5), .02)
            retreat = wide_wheel_bounds(-.4, -1.2) if direction == 1 else wide_wheel_bounds(4.2, 3.4)
            event = state.update(retreat, .04)
            self.assertFalse(event["failed"][0] or event["exited"][0])
            self.assertTrue(state.active[0])
            state.update(wide_wheel_bounds(1.3, .5), .06)
            final = wide_wheel_bounds(4.2, 3.4) if direction == 1 else wide_wheel_bounds(-.4, -1.2)
            event = state.update(final, .08)
            self.assertTrue(event["exited"][0])
            self.assertEqual(int(state.exit_edge_code[0]), 2 if direction == 1 else 1)

    def test_partial_entry_and_never_entered_bypass_are_not_success_or_failure(self):
        state = self.tracker()
        state.update(wide_wheel_bounds(-.05, -.85), .02)
        side = wide_wheel_bounds(-.05, -.85)
        side[..., 2:] += 2.
        event = state.update(side, .04)
        self.assertFalse(event["failed"][0] or event["exited"][0])
        self.assertFalse(state.wheels_entered_mask.all())
        state = self.tracker()
        before = wide_wheel_bounds(-.4, -1.2)
        before[..., 2:] += 4.
        state.update(before, .02)
        after = wide_wheel_bounds(4.2, 3.4)
        after[..., 2:] += 4.
        event = state.update(after, .04)
        self.assertFalse(event["entered"][0] or event["exited"][0] or event["failed"][0])

    def test_extending_an_exit_line_beyond_its_segment_does_not_count(self):
        state = self.tracker()
        state.update(wide_wheel_bounds(1.3, .5), .02)
        # Leave by the excluded entrance, then move left-to-right above the ROI.
        state.update(wide_wheel_bounds(-.4, -1.2), .04)
        side = wide_wheel_bounds(-.4, -1.2)
        side[..., 2:] += 4.
        state.update(side, .06)
        far = wide_wheel_bounds(4.2, 3.4)
        far[..., 2:] += 4.
        event = state.update(far, .08)
        self.assertFalse(event["exited"][0] or event["failed"][0])
        self.assertTrue(state.active[0])

    def test_entry_from_a_side_counts_as_actual_entry_and_reset_clears_exit_history(self):
        state = self.tracker()
        side = wide_wheel_bounds(-.4, -1.2)
        side[..., 2:] += 4.
        state.update(side, .02)
        side = wide_wheel_bounds(1.3, .5)
        side[..., 2:] += 4.
        state.update(side, .04)
        state.update(wide_wheel_bounds(1.3, .5), .06)
        self.assertTrue(state.wheels_entered_mask.all())
        event = state.update(wide_wheel_bounds(4.2, 3.4), .08)
        self.assertTrue(event["exited"][0])
        state.reset([0], window_bounds=[[0., 3., -1.5, 1.5]], travel_direction=1,
                    selected_side=2, wheel_bounds=wide_wheel_bounds(-.4, -1.2))
        self.assertFalse(state.wheel_exit_edges.any())
        self.assertEqual(int(state.exit_edge_code[0]), 0)


if __name__ == "__main__":
    unittest.main()
