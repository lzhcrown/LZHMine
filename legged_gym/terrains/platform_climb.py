"""Training measurements relative to a single intended face of a square WF platform."""

from typing import NamedTuple

import torch


class PlatformClimbState(NamedTuple):
    forward: torch.Tensor
    lateral: torch.Tensor
    in_lane: torch.Tensor
    wheel_fraction: torch.Tensor
    settled: torch.Tensor
    bypass: torch.Tensor


def route_coordinates(points_xy, center_xy, forward_xy):
    """Signed distances along and to the left of a cardinal unit direction.

    Centers and directions have shape [N, 2]. Points are either base positions
    [N, 2] or a set of wheel centers per environment [N, K, 2].
    """
    if points_xy.ndim == 3:
        center_xy = center_xy.unsqueeze(1)
        forward_xy = forward_xy.unsqueeze(1)
    offset = points_xy - center_xy
    along = (offset * forward_xy).sum(dim=-1)
    lateral = offset[..., 1] * forward_xy[..., 0] - offset[..., 0] * forward_xy[..., 1]
    return along, lateral


def platform_climb_state(base_xy, wheel_xy, support_z, gravity_z, center_xy,
                         forward_xy, is_block, half_width, tile_half_length=4.5):
    """Measure a control step from simulator state before reset.

    All world positions are in meters. `half_width` is one half-side of each
    raised square / pit opening, not the half-side of the whole terrain tile.
    `support_z` is the four wheel contact forces along world +Z in newtons.
    `gravity_z` is the vertical component of unit gravity in the body frame.
    A caller can retain peak wheel_fraction and reward only new highs; it must
    check several consecutive settled steps to verify stable completion.
    """
    forward, lateral = route_coordinates(base_xy, center_xy, forward_xy)
    wheel_forward, wheel_lateral = route_coordinates(wheel_xy, center_xy, forward_xy)
    in_lane = lateral.abs() <= (half_width - 0.25)

    # Keep wheel centers clear of the vertical face and the edge ambiguity of
    # the WF mesh (0.10 m horizontal samples, 0.11 m nominal wheel radius).
    edge_clearance = 0.15
    inset = half_width[:, None] - edge_clearance
    block_top = wheel_forward.abs() <= inset
    pit_exit = ((wheel_forward >= half_width[:, None] + edge_clearance)
                & (wheel_forward <= tile_half_length - edge_clearance))
    upper_surface = torch.where(is_block[:, None], block_top, pit_exit)
    upper_surface &= (wheel_lateral.abs() <= inset) & in_lane[:, None]

    supported = support_z > 1.0
    wheel_fraction = (upper_surface & supported).to(base_xy.dtype).mean(dim=1)
    settled = upper_surface.all(dim=1) & (supported.sum(dim=1) >= 3) & (gravity_z < -0.8)
    return PlatformClimbState(forward, lateral, in_lane, wheel_fraction, settled, ~in_lane)
