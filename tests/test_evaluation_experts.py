"""Actual-action routing, pre-step clocks, terrain separation and selection."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from legged_gym.evaluation.experts import act_with_expert_weights, record_expert_step
from legged_gym.evaluation.metrics import VectorEpisodeMetrics
from legged_gym.evaluation.selective_writer import SelectiveCsvEvaluationWriter
from legged_gym.scripts.validate_evaluation_capture import validate


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_actions_and_gates_match_inference_and_never_call_critic():
    policy = ActorCriticMoECTS(num_actor_obs=12, num_critic_obs=5, num_one_step_obs=3,
        num_actions=2, observation_dims=(1, 1, 1), actor_type='moe', critic_type='moe',
        actor_hidden_dims=[16], critic_hidden_dims=[16], student_encoder_hidden_dims=[16],
        teacher_encoder_hidden_dims=[16]).eval()
    obs = torch.randn(4, 12)
    with torch.inference_mode(), patch.object(policy.critic, 'forward', side_effect=AssertionError('critic called')), \
            patch.object(policy.critic.moe, 'forward', side_effect=AssertionError('critic called')):
        expected = policy.act_inference(obs)
        actions, gates = act_with_expert_weights(policy, obs)
        assert torch.equal(actions, expected)
        torch.testing.assert_close(gates['student'], policy.get_expert_probabilities(obs))
        torch.testing.assert_close(gates['actor'], policy.get_actor_expert_probabilities(obs))
        for value in gates.values():
            assert value.shape == (4, 8)
            torch.testing.assert_close(value.sum(dim=1), torch.ones(4))


def capture(directory):
    counts = {'student': 8, 'actor': 8}
    with SelectiveCsvEvaluationWriter(directory, num_joints=2, expert_counts=counts,
                                      terrains=['flat', 'slope'], successes_per_terrain=1) as writer:
        metrics = VectorEpisodeMetrics(2, 'cpu', num_joints=2)
        env = NS(metrics=metrics, num_envs=2, dt=.02)
        all_rows = []
        uniform = torch.ones(2, 8) / 8
        record_expert_step(writer, env, {'student': uniform, 'actor': uniform})
        assert not writer.pending  # Approach is excluded.
        for attempt in range(2):
            metrics.start([0, 1], position_world=[[0., 0., 0.]] * 2, velocity_world=[[1., 0., 0.]] * 2,
                goal_direction_xy=[[1., 0.]] * 2, mass_kg=[20., 20.], gravity_m_s2=9.81,
                metadata=[dict(task='g7_all_moe', checkpoint='model.pt', terrain=terrain, terrain_id=i,
                               episode_index=attempt, eval_seed=10, entry_time_s=.1, exit_time_s=.14,
                               trial_elapsed_s=.14, travel_direction=1, selected_side='all')
                          for i, terrain in enumerate(['flat', 'slope'])])
            for step in (1, 2):
                student = torch.softmax(torch.arange(8, dtype=torch.float) * step, dim=0).repeat(2, 1)
                record_expert_step(writer, env, {'student': student, 'actor': student.flip(-1)})
                for _ in range(4):
                    metrics.physics_step(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), .005)
                velocity = torch.tensor([[1., 0., 0.]] * 2, dtype=torch.float64)
                metrics.control_step(velocity * step * .02, velocity, velocity, torch.zeros(2, 3), velocity, .02)
            rows = metrics.finish([0, 1], ['success', 'failure'] if attempt == 0 else 'success')
            writer.write_episodes(rows)
            all_rows.extend(rows)
            assert not writer.pending
        writer.write_summary(all_rows)
        metadata = dict(status='complete', group='g7_all_moe', checkpoint='model.pt', recorded_attempts=4,
            expert_counts=counts, run_id=writer.run_id, trial_annotation_schema=1,
            protocol=dict(data_mode='expert_weights_only', metric_successes_per_terrain=1,
                          terrain_types=['flat', 'slope'], envs_per_terrain=1, episodes_per_env=2,
                          episodes_per_terrain=2, total_attempts=4, seed=10, command_body_vx_vy_wz=[1., 0., 0.]),
            environment_config={'control': {'decimation': 4}}, joint_names=['a', 'b'],
            physics_dt_s=.005, control_dt_s=.02)
        (Path(directory) / 'metadata.json').write_text(json.dumps(metadata))


def test_selected_gates_by_terrain_align_with_exact_action_intervals():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        capture(root)
        report = validate(root)
        assert report['expert_samples'] == {'student': 4, 'actor': 4}
        assert report['physics_samples'] == 0 and report['metric_trials_selected'] == 2
        assert not (root / 'power.csv').exists() and not (root / 'tracking.csv').exists()
        for terrain, attempt in [('flat', '0'), ('slope', '1')]:
            student = read(root / 'experts' / terrain / 'student_weights.csv')
            actor = read(root / 'experts' / terrain / 'actor_weights.csv')
            assert len(student) == len(actor) == 2
            for step, row in enumerate(student, 1):
                assert row['terrain'] == terrain and row['episode_index'] == attempt
                assert int(row['control_step']) == step
                assert float(row['time_start_s']) == pytest.approx((step - 1) * .02)
                expected = torch.softmax(torch.arange(8, dtype=torch.float) * step, dim=0)
                assert float(row['expert_00_weight']) == pytest.approx(expected[0].item())
                assert float(actor[step - 1]['expert_07_weight']) == float(row['expert_00_weight'])


@pytest.mark.parametrize('field,value', [('expert_00_weight', '-0.1'), ('control_step', '99'),
                                         ('terrain', 'slope')])
def test_invalid_gate_files_are_rejected(field, value):
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        capture(root)
        path = root / 'experts/flat/student_weights.csv'
        rows = read(path)
        rows[0][field] = value
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(ValueError, match='expert probabilities|clock/step|provenance'):
            validate(root)
