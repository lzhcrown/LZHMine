"""Observation contract tests that need neither Isaac Gym nor a GPU."""

import json

import pytest
import torch

from rsl_rl.utils.cts_observations import (
    CTSObservationHistory,
    build_privileged_observations,
    build_proprioception,
    contact_history_features,
    default_observation_config,
    extract_current_observation,
    observation_dimensions,
    validate_observation_config,
)


def raw_observations(batch=2):
    config = default_observation_config()
    return {term["name"]: torch.zeros(batch, term["dim"])
            for term in config["privileged_terms"]}


def test_wf_dimensions_and_schema_round_trip():
    config = default_observation_config()
    assert observation_dimensions(config) == (57, 228, 295)
    assert json.loads(json.dumps(config)) == config
    config["history_length"] = 10
    assert observation_dimensions(config) == (57, 570, 295)
    assert default_observation_config()["history_length"] == 4


def test_proprioception_preserves_wheel_rates_and_masks_only_wheel_positions():
    raw = raw_observations()
    raw["base_ang_vel"].fill_(4)
    raw["projected_gravity"][:, 2] = -1
    raw["velocity_commands"][:] = torch.tensor([1.0, 2.0, 3.0])
    raw["joint_pos"].fill_(0.5)
    raw["joint_vel"].fill_(20)
    raw["actions"].fill_(0.8)
    before = {key: value.clone() for key, value in raw.items()}
    obs = build_proprioception(raw)
    assert obs.shape == (2, 57)
    assert torch.equal(obs[:, :3], torch.ones(2, 3))
    assert torch.equal(obs[:, 6:9], torch.tensor([[2.0, 4.0, 0.75]]).expand(2, -1))
    assert torch.equal(obs[:, 9:25][:, [3, 7, 11, 15]], torch.zeros(2, 4))
    assert torch.equal(obs[:, 25:41], torch.ones(2, 16))
    assert torch.equal(obs[:, 41:], raw["actions"])
    assert all(torch.equal(raw[key], value) for key, value in before.items())


def test_privileged_order_independent_command_scale_and_clip_before_scale():
    raw = raw_observations(1)
    raw["base_lin_vel"][:] = torch.tensor([[200.0, -200.0, 3.0]])
    raw["base_ang_vel"].fill_(200.0)
    raw["velocity_commands"][:] = torch.tensor([[1.0, 2.0, 3.0]])
    raw["joint_acc"].fill_(2000.0)
    raw["joint_torque"].fill_(-200.0)
    raw["contact_force"].fill_(1000.0)
    raw["height_scan"].fill_(2.0)
    obs = build_privileged_observations(raw)
    assert obs.shape == (1, 295)
    assert torch.equal(obs[0, :3], torch.tensor([200.0, -200.0, 6.0]))
    assert torch.equal(obs[0, 3:6], torch.full((3,), 25.0))
    assert torch.equal(obs[0, 9:12], raw["velocity_commands"][0])
    assert torch.allclose(obs[:, 60:76], torch.full((1, 16), 0.01))
    assert torch.equal(obs[:, 76:92], torch.full((1, 16), -1.0))
    assert torch.allclose(obs[:, 92:108], torch.full((1, 16), 0.1))
    assert torch.equal(obs[:, 108:], torch.full((1, 187), 2.5))


def test_uniform_noise_bounds_and_wheel_position_zero_after_corruption():
    torch.manual_seed(17)
    raw = raw_observations(1024)
    obs = build_proprioception(raw, add_noise=True)
    assert obs[:, :3].abs().max() <= 0.2 * 0.25
    assert obs[:, 3:6].abs().max() <= 0.05
    assert obs[:, 9:25].abs().max() <= 0.03
    assert obs[:, 25:41].abs().max() <= 2.0 * 0.05
    assert obs[:, 25:41].std() > 0.04
    assert torch.count_nonzero(obs[:, 6:9]) == 0
    assert torch.count_nonzero(obs[:, 41:]) == 0
    assert torch.count_nonzero(obs[:, 9:25][:, [3, 7, 11, 15]]) == 0
    assert torch.count_nonzero(build_privileged_observations(raw)) == 0


def test_term_major_history_is_chronological_and_recovers_identical_current_frame():
    config = default_observation_config()
    history = CTSObservationHistory(2, config)
    dims = [term["dim"] for term in config["proprioceptive_terms"]]
    frames = [torch.arange(114, dtype=torch.float32).reshape(2, 57) + i * 1000 for i in range(4)]
    for frame in frames:
        obs = history.update(frame)
    expected_angular = torch.stack([frame[:, :3] for frame in frames], dim=1).reshape(2, 12)
    expected_gravity = torch.stack([frame[:, 3:6] for frame in frames], dim=1).reshape(2, 12)
    assert torch.equal(obs[:, :12], expected_angular)
    assert torch.equal(obs[:, 12:24], expected_gravity)
    assert torch.equal(extract_current_observation(obs, 4, dims), frames[-1])


def test_reset_repeats_first_frame_only_for_reset_environments():
    history = CTSObservationHistory(2)
    first = torch.ones(2, 57)
    history.update(first)
    assert torch.equal(history.buffer, first.unsqueeze(1).expand(-1, 4, -1))
    history.update(first * 2)
    history.reset(torch.tensor([0]))
    history.update(first * 3)
    assert torch.equal(history.buffer[0], torch.full((4, 57), 3.0))
    assert torch.equal(history.buffer[1, :, 0], torch.tensor([1.0, 1.0, 2.0, 3.0]))


def test_contact_feature_layout_tracks_physics_samples_and_maximum():
    history = torch.zeros(1, 3, 4, 3)
    history[0, 0, :, 2] = torch.tensor([1.0, 8.0, 3.0, 4.0])
    history[0, 1, :, 2] = torch.tensor([5.0, 2.0, 7.0, 4.0])
    history[0, 2, :, 2] = torch.tensor([4.0, 6.0, 1.0, 9.0])
    assert torch.equal(contact_history_features(history), torch.tensor([
        [5.0, 8.0, 7.0, 9.0, 1.0, 8.0, 3.0, 4.0,
         5.0, 2.0, 7.0, 4.0, 4.0, 6.0, 1.0, 9.0]
    ]))


def test_invalid_contract_is_rejected_before_training():
    config = default_observation_config()
    config["history_layout"] = "frame_major"
    with pytest.raises(ValueError, match="term_major"):
        validate_observation_config(config)
    raw = raw_observations()
    raw["joint_vel"] = torch.zeros(2, 12)
    with pytest.raises(ValueError, match="joint_vel"):
        build_proprioception(raw)
