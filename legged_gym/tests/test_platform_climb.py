import pytest
import torch

from legged_gym.terrains.platform_climb import platform_climb_state, route_coordinates


def measure(wheels, *, block=True, forces=(20., 20., 20., 20.), gravity_z=-1.,
            direction=(1., 0.)):
    # Specify wheel positions in route coordinates and place the tile away
    # from the world origin, as happens on most rows and columns.
    center = torch.tensor([[49.5, 22.5]])
    forward = torch.tensor([direction])
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    along, lateral = torch.tensor(wheels).unbind(dim=1)
    world = center + along[:, None] * forward + lateral[:, None] * left
    return platform_climb_state(world.mean(dim=0, keepdim=True), world[None],
                                torch.tensor([forces]), torch.tensor([gravity_z]),
                                center, forward, torch.tensor([block]), torch.tensor([1.6]))


def wheel_rectangle(x, y=0.):
    return [(x + dx, y + dy) for dx, dy in ((0.3, 0.2), (0.3, -0.2),
                                             (-0.3, 0.2), (-0.3, -0.2))]


@pytest.mark.parametrize("direction", [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)])
def test_pit_requires_assigned_exit_face(direction):
    result = measure(wheel_rectangle(2.5, -0.2), block=False, direction=direction)
    assert result.settled and result.wheel_fraction == 1.
    assert result.forward.item() == pytest.approx(2.5, abs=1e-5)
    assert result.lateral.item() == pytest.approx(-0.2, abs=1e-5)
    for x, y in ((-2.5, 0.), (0., 2.5), (0., -2.5), (5., 0.)):
        wrong_surface = measure(wheel_rectangle(x, y), block=False, direction=direction)
        assert not wrong_surface.settled and wrong_surface.wheel_fraction == 0.


def test_block_discriminates_approach_front_axle_and_full_ascent():
    for x, fraction in ((-2.2, 0.), (-1.6, 0.5), (-1.1, 1.), (2.2, 0.)):
        result = measure(wheel_rectangle(x))
        assert result.wheel_fraction == fraction
        assert result.settled.item() == (fraction == 1.)
    # A wheel on the vertical face has not cleared it, even if exerting load.
    on_lip = measure([(-1.55, 0.)] * 4)
    assert on_lip.wheel_fraction == 0.
    # Edges outside the approach lane earn nothing even when on top.
    bypass = measure(wheel_rectangle(0., 1.4))
    assert bypass.bypass and not bypass.in_lane and not bypass.settled
    assert bypass.wheel_fraction == 0.


def test_three_supports_suffice_only_with_all_wheels_over_the_upper_surface():
    all_on_top = wheel_rectangle(0.)
    assert measure(all_on_top, forces=(20., 20., 20., 0.)).settled
    two_supports = measure(all_on_top, forces=(0., 20., -2., 20.))
    assert two_supports.wheel_fraction == 0.5 and not two_supports.settled
    assert not measure(all_on_top, gravity_z=-0.4).settled
    assert not measure([*all_on_top[:3], (-1.8, -0.3)], forces=(20., 20., 20., 0.)).settled


def test_batched_mixed_shapes_use_distinct_tile_origins_and_headings():
    centers = torch.tensor([[22.5, 4.5], [4.5, 13.5]])
    directions = torch.tensor([[1., 0.], [0., -1.]])
    base = centers + torch.tensor([[-0.2, -0.5], [0.3, -2.5]])
    forward, left = route_coordinates(base, centers, directions)
    torch.testing.assert_close(forward, torch.tensor([-0.2, 2.5]))
    torch.testing.assert_close(left, torch.tensor([-0.5, 0.3]))
    wheels = base[:, None, :] + torch.tensor(wheel_rectangle(0.))[None, :, :]
    state = platform_climb_state(base, wheels, torch.full((2, 4), 10.),
                                 torch.full((2,), -1.), centers, directions,
                                 torch.tensor([True, False]), torch.tensor([1.6, 1.6]))
    assert state.settled.tolist() == [True, True]
