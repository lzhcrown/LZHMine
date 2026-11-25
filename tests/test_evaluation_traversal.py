"""A complete obstacle run includes contact in the middle and a stable far-side landing."""
import pytest
import torch

from legged_gym.evaluation.traversal import ObstacleTraversal, wheel_centers_and_extents


def test_settling_is_consecutive_and_scoped_to_each_robot_and_attempt():
    traversal = ObstacleTraversal(2, .01999999955, "cpu")
    middle = torch.ones(2, 4, dtype=torch.bool)
    ground = torch.zeros_like(middle)
    lane = torch.ones(2, dtype=torch.bool)
    forces = torch.full((2, 4), 80.)
    forces[0, 3] = 0.
    forces[1, 2:] = 0.
    gravity = torch.full((2,), -1.)
    vz, omega = torch.zeros(2), torch.zeros(2, 2)

    def step():
        return traversal.update(middle, ground, lane, forces, gravity, vz, omega).tolist()

    assert step() == [False, False]
    assert traversal.reached_middle.tolist() == [True, False]
    middle[:] = False
    ground[:] = True
    forces[:] = 80.
    for _ in range(9):
        assert step() == [False, False]
    vz[0] = .31
    assert step() == [False, False]
    assert traversal.landing_steps.tolist() == [0, 0]
    vz[0] = 0.
    for _ in range(9):
        assert step() == [False, False]
    assert step() == [True, False]
    traversal.reset(torch.tensor([0]))
    assert step() == [False, False]


@pytest.mark.parametrize("disturbance", ["tilt", "rotation", "airborne", "one_wheel_high", "outside_lane"])
def test_unstable_or_partial_landing_never_completes(disturbance):
    traversal = ObstacleTraversal(1, .02, "cpu")
    inside = torch.ones((1, 4), dtype=torch.bool)
    lane = torch.tensor([True])
    force = torch.ones(1, 4) * 50
    gravity, vz, omega = torch.tensor([-1.]), torch.zeros(1), torch.zeros(1, 2)
    traversal.update(inside, ~inside, lane, force, gravity, vz, omega)
    if disturbance == "tilt":
        gravity[0] = -.7
    elif disturbance == "rotation":
        omega[0, 1] = 1.1
    elif disturbance == "airborne":
        force[:] = 0.
    elif disturbance == "one_wheel_high":
        inside[0, 3] = False
    else:
        lane[:] = False
    for _ in range(15):
        assert not traversal.update(torch.zeros_like(inside), inside, lane, force, gravity, vz, omega).item()


def test_oriented_wheel_bottom_uses_cylinder_projection_and_collision_offset():
    states = torch.zeros(1, 1, 13)
    states[0, 0, :3] = torch.tensor([0., 0., .6])
    states[0, 0, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    center, extent = wheel_centers_and_extents(
        states, torch.tensor([[0., .08, 0.]]), torch.tensor([[0., 1., 0.]]),
        torch.tensor([.11]), torch.tensor([.04]))
    torch.testing.assert_close(center[..., 2] - extent[..., 2], torch.tensor([[.66]]))
