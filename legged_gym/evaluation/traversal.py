"""Physical obstacle traversal at control-step resolution, shared by Gym replays."""
import math

import torch


def wheel_centers_and_extents(rigid_states, collision_offsets, collision_axes, radii, widths):
    """World center and axis-aligned half-extents of each oriented wheel cylinder."""
    quat = rigid_states[..., 3:7]

    def rotate(vectors):
        vectors = vectors.expand_as(rigid_states[..., :3])
        twice_cross = 2 * torch.cross(quat[..., :3], vectors, dim=-1)
        return vectors + quat[..., 3:] * twice_cross + torch.cross(quat[..., :3], twice_cross, dim=-1)

    center = rigid_states[..., :3] + rotate(collision_offsets)
    axis = torch.nn.functional.normalize(rotate(collision_axes), dim=-1)
    extent = radii[:, None] * (1. - axis.square()).clamp_min(0.).sqrt() + widths[:, None] * .5 * axis.abs()
    return center, extent


class ObstacleTraversal:
    """Remember a supported visit to the middle, then require sustained far-side ground contact."""

    HEIGHT_TOLERANCE_M = .08
    SUPPORT_FORCE_N = 1.
    SETTLE_TIME_S = .20
    MAX_GRAVITY_Z = -.8
    MAX_VERTICAL_SPEED_M_S = .30
    MAX_TILT_RATE_RAD_S = 1.

    def __init__(self, num_envs, dt, device):
        # The simulator stores dt as float32 (e.g. .01999999955 for 50 Hz).
        steps = self.SETTLE_TIME_S / dt
        self.required_steps = max(1, round(steps) if math.isclose(steps, round(steps), rel_tol=1e-7)
                                  else math.ceil(steps))
        self.reached_middle = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.landing_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids):
        self.reached_middle[env_ids] = False
        self.landing_steps[env_ids] = 0

    def update(self, on_middle, on_far_ground, in_lane, support_z, gravity_z, world_vz, body_omega_xy):
        """Surface masks are per wheel and include footprint and wheel-bottom height checks."""
        supported = (support_z > self.SUPPORT_FORCE_N).sum(dim=1) >= 3
        self.reached_middle |= on_middle.all(dim=1) & supported
        self.reached_middle &= in_lane
        stable = (self.reached_middle & on_far_ground.all(dim=1) & supported
                  & (gravity_z <= self.MAX_GRAVITY_Z) & (world_vz.abs() <= self.MAX_VERTICAL_SPEED_M_S)
                  & (torch.linalg.vector_norm(body_omega_xy, dim=1) <= self.MAX_TILT_RATE_RAD_S))
        self.landing_steps.copy_(torch.where(stable, self.landing_steps + 1, 0))
        return self.landing_steps >= self.required_steps
