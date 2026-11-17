# Adapted from wheel_gym_CQ/wheel_legged_gym/evaluation/crossing.py
"""Directional wheel-projection crossing state, independent of Isaac Gym.

Wheel ordering is FL, FR, RL, RR. Bounds contain xmin, xmax, ymin, ymax in
world coordinates, including the projected collision geometry of each wheel.
Projection entry is independent of height/contact, so airborne crossings count.
The caller samples at control boundaries: entry is timestamped at that boundary,
and measurement continues through any gap between front and rear axles.
"""

import torch


WHEEL_NAMES = ("FL", "FR", "RL", "RR")
AXLE_NAMES = ("front", "rear")
REASON_NAMES = {0: "", 1: "lateral_escape", 2: "retreated", 3: "invalid_geometry"}


class WheelWindowCrossing:
    """Track one complete traversal per environment until reset.

    ``selected_side`` is 0 for FL/RL, 1 for FR/RR, or 2 for all four wheels.
    Every selected wheel must enter through the entrance; all four wheels must fully clear the exit for
    success. This prevents a narrow block between both wheel tracks from being
    counted solely because the robot base passes it.
    """

    def __init__(self, num_envs, device, tolerance_m=1e-7, exit_rule="opposite"):
        if int(num_envs) != num_envs or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not 0 <= float(tolerance_m) < 1e-3:
            raise ValueError("tolerance_m must be nonnegative and less than 1 mm")
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.tolerance_m = float(tolerance_m)
        if exit_rule not in ("opposite", "opposite_or_sides"):
            raise ValueError("Unknown exit rule")
        self.exit_rule = exit_rule
        self.active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.window_started = self.active.clone()
        self.wheels_entered_mask = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.bool)
        self.wheel_contact_mask = self.wheels_entered_mask.clone()
        self.selected_pair_mask = self.wheels_entered_mask.clone()
        self.selected_side = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.travel_direction = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.leading_axle = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.reason_code = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.window_bounds = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float64)
        self.previous_wheel_bounds = torch.zeros((self.num_envs, 4, 4), device=self.device, dtype=torch.float64)
        self.entry_time_s = torch.full((self.num_envs,), float("nan"), device=self.device, dtype=torch.float64)
        self.exit_time_s = self.entry_time_s.clone()
        self.previous_time_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        # xmin, xmax, ymin, ymax: valid outward crossings for each wheel.
        self.wheel_exit_edges = torch.zeros((self.num_envs, 4, 4), device=self.device, dtype=torch.bool)
        self.exit_edge_code = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _tensor(self, value, shape, name, dtype=torch.float64):
        value = torch.as_tensor(value, dtype=dtype, device=self.device)
        if tuple(value.shape) != tuple(shape):
            raise ValueError("{} must have shape {}".format(name, shape))
        return value

    def _ids(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("environment IDs must be unique and within range")
        return ids

    def _vector(self, value, n, name):
        result = torch.as_tensor(value, device=self.device, dtype=torch.float64)
        if result.ndim == 0:
            result = result.expand(n)
        return self._tensor(result, (n,), name)

    @staticmethod
    def _longitudinal(bounds, direction):
        """Return increasing-travel leading/trailing edges, irrespective of yaw."""
        forward = direction[:, None] > 0
        return (torch.where(forward, bounds[:, :, 1], -bounds[:, :, 0]),
                torch.where(forward, bounds[:, :, 0], -bounds[:, :, 1]))

    @torch.no_grad()
    def reset(self, env_ids, *, window_bounds, travel_direction, selected_side,
              wheel_bounds, wheel_contact_mask=None):
        """Initialize selected slots using fresh physical or FK wheel bounds.

        All four wheels must start wholly before the entrance plane. Reset
        deliberately discards previous crossing/contact state and timestamps.
        ``wheel_contact_mask`` is accepted for symmetry but pre-window contact
        is never included in the measurement's observed contact mask.
        """
        ids = self._ids(env_ids)
        n = ids.numel()
        window = self._tensor(window_bounds, (n, 4), "window_bounds")
        wheels = self._tensor(wheel_bounds, (n, 4, 4), "wheel_bounds")
        direction = self._vector(travel_direction, n, "travel_direction")
        side = self._vector(selected_side, n, "selected_side")
        if not bool(torch.isfinite(window).all() & torch.isfinite(wheels).all()):
            raise ValueError("initial projection bounds must be finite")
        if bool(((window[:, 1] <= window[:, 0]) | (window[:, 3] <= window[:, 2])).any()):
            raise ValueError("window bounds must have positive width and length")
        if bool(((wheels[:, :, 1] < wheels[:, :, 0]) | (wheels[:, :, 3] < wheels[:, :, 2])).any()):
            raise ValueError("wheel bounds must be ordered")
        if bool(((direction != 1) & (direction != -1)).any()) or bool(((side != 0) & (side != 1) & (side != 2)).any()):
            raise ValueError("travel_direction must be +/-1 and selected_side must be 0/1/2 (left/right/all)")
        if wheel_contact_mask is not None:
            self._tensor(wheel_contact_mask, (n, 4), "wheel_contact_mask", dtype=torch.bool)
        leading, _ = self._longitudinal(wheels, direction)
        entrance = torch.where(direction > 0, window[:, 0], -window[:, 1])
        if bool((leading > entrance[:, None] + self.tolerance_m).any()):
            raise ValueError("all wheels must start completely before the entrance")
        self.window_bounds[ids], self.previous_wheel_bounds[ids] = window, wheels
        self.travel_direction[ids], self.selected_side[ids] = direction.long(), side.long()
        self.selected_pair_mask[ids] = ((side[:, None] == 2) |
                                        (torch.arange(4, device=self.device)[None, :] % 2 == side[:, None]))
        self.wheels_entered_mask[ids] = False
        self.wheel_contact_mask[ids] = False
        self.window_started[ids] = False
        self.leading_axle[ids] = -1
        self.reason_code[ids] = 0
        self.entry_time_s[ids] = float("nan")
        self.exit_time_s[ids] = float("nan")
        self.previous_time_s[ids] = 0.
        self.wheel_exit_edges[ids] = False
        self.exit_edge_code[ids] = 0
        self.active[ids] = True

    def _update_passage(self, bounds, elapsed, contact, valid):
        """Protocol 7: enter the rectangle, then clear a non-entry edge.

        Retreat/side motion never ends an attempt. Each wheel must have entered;
        exit crossings must intersect the finite edge segment. A robot returning
        through the original entrance can re-enter and try again until timeout.
        """
        p = self.window_bounds[:, None, :]
        previous = self.previous_wheel_bounds
        # Swept linear AABB overlap catches entry between two control samples.
        # As in the legacy tracker this is an approximation between sampled poses.
        start = torch.stack((previous[..., 1] - p[..., 0], p[..., 1] - previous[..., 0],
                             previous[..., 3] - p[..., 2], p[..., 3] - previous[..., 2]), dim=-1)
        end = torch.stack((bounds[..., 1] - p[..., 0], p[..., 1] - bounds[..., 0],
                           bounds[..., 3] - p[..., 2], p[..., 3] - bounds[..., 2]), dim=-1)
        change = end - start
        moving = change.abs() > 1e-12
        crossing = -start / torch.where(moving, change, torch.ones_like(change))
        lower = torch.where(change > 1e-12, crossing, torch.zeros_like(crossing)).amax(dim=-1).clamp_min(0.)
        upper = torch.where(change < -1e-12, crossing, torch.ones_like(crossing)).amin(dim=-1).clamp_max(1.)
        overlaps = (lower <= upper) & ((moving | (start >= 0.)).all(dim=-1))
        new_entry = self.active[:, None] & valid[:, None] & ~self.wheels_entered_mask & overlaps
        selected_entry = new_entry & self.selected_pair_mask
        entered = self.active & ~self.window_started & selected_entry.any(dim=1)
        first = torch.where(selected_entry, lower, torch.full_like(lower, float("inf"))).argmin(dim=1)
        self.leading_axle = torch.where(entered, first // 2, self.leading_axle)
        self.entry_time_s = torch.where(entered, elapsed, self.entry_time_s)
        self.window_started |= entered
        self.wheels_entered_mask |= new_entry
        overlap_now = (end >= 0.).all(dim=-1)
        self.wheel_contact_mask |= contact & overlap_now & self.active[:, None] & valid[:, None]

        clear_edges = []
        # A wheel clears an edge only when its trailing extremity crosses it.
        for edge, axis, sign in ((0, 0, -1), (1, 0, 1), (2, 1, -1), (3, 1, 1)):
            trailing_index = 2 * axis + (1 if sign < 0 else 0)
            plane = sign * self.window_bounds[:, edge, None]
            before = sign * previous[..., trailing_index]
            after = sign * bounds[..., trailing_index]
            delta = after - before
            fraction = ((plane - before) / delta.clamp_min(1e-12)).clamp(0., 1.)
            orthogonal = 2 * (1 - axis)
            edge_bounds = previous[..., orthogonal:orthogonal + 2] + fraction[..., None] * (
                bounds[..., orthogonal:orthogonal + 2] - previous[..., orthogonal:orthogonal + 2])
            finite_edge = ((edge_bounds[..., 1] >= p[..., orthogonal]) &
                           (edge_bounds[..., 0] <= p[..., orthogonal + 1]))
            clear = after >= plane - self.tolerance_m
            crossed = ((before < plane - self.tolerance_m) & clear & (delta > 0) & finite_edge &
                       self.wheels_entered_mask & self.active[:, None] & valid[:, None])
            self.wheel_exit_edges[:, :, edge] = (self.wheel_exit_edges[:, :, edge] | crossed) & clear
            valid_exits = (self.wheel_exit_edges[:, :, edge] | ~self.selected_pair_mask).all(dim=1)
            allowed = ((self.travel_direction < 0) if edge == 0 else
                       (self.travel_direction > 0) if edge == 1 else torch.ones_like(self.active))
            clear_edges.append(clear.all(dim=1) & valid_exits & allowed)
        cleared = torch.stack(clear_edges, dim=1)
        all_entered = (self.wheels_entered_mask | ~self.selected_pair_mask).all(dim=1)
        failed = self.active & ~valid  # Numerical errors are handled separately by the environment.
        exited = self.active & self.window_started & all_entered & cleared.any(dim=1) & valid
        self.exit_edge_code = torch.where(exited, cleared.long().argmax(dim=1) + 1, self.exit_edge_code)
        self.exit_time_s = torch.where(exited, elapsed, self.exit_time_s)
        self.reason_code = torch.where(failed, 3, self.reason_code)
        self.previous_wheel_bounds = torch.where(self.active[:, None, None], bounds, previous)
        self.previous_time_s = torch.where(self.active, elapsed, self.previous_time_s)
        self.active &= ~(exited | failed)
        return dict(entered=entered, exited=exited, failed=failed, reason_code=self.reason_code,
                    window_started=self.window_started, entry_time_s=self.entry_time_s,
                    exit_time_s=self.exit_time_s, leading_axle=self.leading_axle,
                    wheels_entered_mask=self.wheels_entered_mask, wheel_contact_mask=self.wheel_contact_mask)

    @torch.no_grad()
    def update(self, wheel_bounds, trial_elapsed_s, wheel_contact_mask=None):
        """Return one-step ``entered``, ``exited``, and ``failed`` event masks.

        Contact observations accumulate only while the wheel's XY projection
        overlaps the window. They are diagnostics, never success requirements.
        For a wheel crossing the entrance between samples, lateral overlap is
        checked by interpolating the two observed AABBs at that entrance plane.
        Entry/exit timestamps remain discrete control-boundary timestamps.
        """
        bounds = self._tensor(wheel_bounds, (self.num_envs, 4, 4), "wheel_bounds")
        elapsed = self._vector(trial_elapsed_s, self.num_envs, "trial_elapsed_s")
        contact = (torch.zeros_like(self.wheel_contact_mask) if wheel_contact_mask is None else
                   self._tensor(wheel_contact_mask, (self.num_envs, 4), "wheel_contact_mask", dtype=torch.bool))
        valid = (torch.isfinite(bounds).all(dim=2).all(dim=1) & torch.isfinite(elapsed)
                 & (elapsed >= self.previous_time_s)
                 & (bounds[:, :, 1] >= bounds[:, :, 0]).all(dim=1)
                 & (bounds[:, :, 3] >= bounds[:, :, 2]).all(dim=1))
        if self.exit_rule == "opposite_or_sides":
            return self._update_passage(bounds, elapsed, contact, valid)
        entrance = torch.where(self.travel_direction > 0, self.window_bounds[:, 0], -self.window_bounds[:, 1])
        exit_plane = torch.where(self.travel_direction > 0, self.window_bounds[:, 1], -self.window_bounds[:, 0])
        leading, trailing = self._longitudinal(bounds, self.travel_direction)
        previous_leading, previous_trailing = self._longitudinal(self.previous_wheel_bounds, self.travel_direction)
        movement = leading - previous_leading
        fraction = ((entrance[:, None] - previous_leading) / movement.clamp_min(1e-12)).clamp(0., 1.)
        entrance_y = self.previous_wheel_bounds[:, :, 2:] + fraction[:, :, None] * (bounds[:, :, 2:] - self.previous_wheel_bounds[:, :, 2:])
        lateral_at_entry = ((entrance_y[:, :, 1] >= self.window_bounds[:, None, 2])
                            & (entrance_y[:, :, 0] <= self.window_bounds[:, None, 3]))
        new_wheel_entry = (self.active[:, None] & valid[:, None] & ~self.wheels_entered_mask
                           & (previous_leading <= entrance[:, None] + self.tolerance_m)
                           & (leading >= entrance[:, None] - self.tolerance_m)
                           & (movement > 0) & lateral_at_entry)
        selected_entry = new_wheel_entry & self.selected_pair_mask
        entered = self.active & ~self.window_started & selected_entry.any(dim=1)
        first_wheel = torch.where(selected_entry, fraction, torch.full_like(fraction, float("inf"))).argmin(dim=1)
        self.leading_axle = torch.where(entered, first_wheel // 2, self.leading_axle)
        self.entry_time_s = torch.where(entered, elapsed, self.entry_time_s)
        self.window_started |= entered
        self.wheels_entered_mask |= new_wheel_entry
        lateral_overlap = ((bounds[:, :, 3] >= self.window_bounds[:, None, 2])
                           & (bounds[:, :, 2] <= self.window_bounds[:, None, 3]))
        longitudinal_overlap = ((leading >= entrance[:, None] - self.tolerance_m)
                                & (trailing <= exit_plane[:, None] + self.tolerance_m))
        observed = contact & lateral_overlap & longitudinal_overlap
        self.wheel_contact_mask |= observed & self.active[:, None] & self.window_started[:, None] & valid[:, None]

        # Check the exit plane at its crossing time as well. Otherwise a wheel
        # can leave through a side and move beyond the exit within one sample,
        # appearing longitudinally clear at the final boundary.
        trailing_movement = trailing - previous_trailing
        exit_fraction = ((exit_plane[:, None] - previous_trailing) / trailing_movement.clamp_min(1e-12)).clamp(0., 1.)
        exit_y = self.previous_wheel_bounds[:, :, 2:] + exit_fraction[:, :, None] * (bounds[:, :, 2:] - self.previous_wheel_bounds[:, :, 2:])
        lateral_at_exit = ((exit_y[:, :, 1] >= self.window_bounds[:, None, 2])
                           & (exit_y[:, :, 0] <= self.window_bounds[:, None, 3]))
        crossing_exit = ((previous_trailing < exit_plane[:, None] - self.tolerance_m)
                         & (trailing >= exit_plane[:, None] - self.tolerance_m)
                         & (trailing_movement > 0))

        # A selected wheel may be behind the entrance while the first axle is
        # already past the block. This gap does NOT end or pause the window.
        # A following wheel may align laterally while still approaching. Only
        # a wheel that actually entered can escape through a side boundary.
        side_exit = (~lateral_overlap & (trailing < exit_plane[:, None] - self.tolerance_m)) | (crossing_exit & ~lateral_at_exit)
        lateral_escape = (self.selected_pair_mask & self.wheels_entered_mask & side_exit).any(dim=1)
        retreated = ((self.selected_pair_mask & self.wheels_entered_mask
                      & (leading < entrance[:, None] - self.tolerance_m)).any(dim=1))
        selected_all_entered = (self.wheels_entered_mask | ~self.selected_pair_mask).all(dim=1)
        all_wheels_clear = (trailing >= exit_plane[:, None] - self.tolerance_m).all(dim=1)
        bypassed = all_wheels_clear & ~selected_all_entered
        failed = self.active & (~valid | bypassed | (self.window_started & (lateral_escape | retreated)))
        reason = torch.where(~valid, 3, torch.where(lateral_escape | bypassed, 1, torch.where(retreated, 2, 0)))
        self.reason_code = torch.where(failed, reason, self.reason_code)
        exited = self.active & self.window_started & selected_all_entered & all_wheels_clear & ~failed
        self.exit_time_s = torch.where(exited, elapsed, self.exit_time_s)
        self.previous_wheel_bounds = torch.where(self.active[:, None, None], bounds, self.previous_wheel_bounds)
        self.previous_time_s = torch.where(self.active, elapsed, self.previous_time_s)
        self.active &= ~(exited | failed)
        return {
            "entered": entered, "exited": exited, "failed": failed,
            "reason_code": self.reason_code, "window_started": self.window_started,
            "entry_time_s": self.entry_time_s, "exit_time_s": self.exit_time_s,
            "leading_axle": self.leading_axle, "wheels_entered_mask": self.wheels_entered_mask,
            "wheel_contact_mask": self.wheel_contact_mask,
        }
