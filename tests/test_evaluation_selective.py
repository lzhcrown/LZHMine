"""Selective capture keeps full successful windows and every trial outcome."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from legged_gym.evaluation.metrics import VectorEpisodeMetrics
from legged_gym.evaluation.selective_writer import SelectiveCsvEvaluationWriter
from legged_gym.scripts.validate_evaluation_capture import validate


def read(directory, name):
    with (Path(directory) / (name + '.csv')).open() as handle:
        return list(csv.DictReader(handle))


def emit(metrics, writer, env_id, episode_index, terrain, status, torque=None):
    metrics.start([env_id], position_world=[[0., 0., 0.]], velocity_world=[[1., 0., 0.]],
                  goal_direction_xy=[[1., 0.]], mass_kg=[20.], gravity_m_s2=9.81,
                  metadata=[dict(task='g0_mlp', checkpoint='model.pt', terrain=terrain, terrain_id=env_id,
                                 episode_index=episode_index, eval_seed=10, entry_time_s=.02, exit_time_s=.04,
                                 travel_direction=1, selected_side='all', trial_elapsed_s=.04)])
    for _ in range(4):
        metrics.physics_step(torch.ones(2, 2) if torque is None else torque,
                             torch.ones(2, 2), torch.ones(2, 2), .005)
    velocity = torch.tensor([[1., 0., 0.]] * 2, dtype=torch.float64)
    metrics.control_step(velocity * .02, velocity, velocity, torch.zeros(2, 3), velocity, .02,
                         orientation_world=torch.tensor([[0., 0., 0., 1.]] * 2),
                         angular_velocity_world=torch.zeros(2, 3))
    # Until the result is known, candidate samples must never touch CSV files.
    before = len(read(writer.output_dir, 'power'))
    rows = metrics.finish([env_id], status, reason='test')
    writer.write_episodes(rows)
    return rows, before


def capture(directory, limit=1):
    rows = []
    with SelectiveCsvEvaluationWriter(directory, num_joints=2, successes_per_terrain=limit) as writer:
        metrics = VectorEpisodeMetrics(2, 'cpu', num_joints=2,
                                       physics_sink=writer.write_physics, trajectory_sink=writer.write_trajectories)
        attempts = [(0, 0, 'flat', 'failure'), (1, 0, 'slope', 'success'),
                    (0, 1, 'flat', 'success'), (1, 1, 'slope', 'timeout'),
                    (0, 2, 'flat', 'success'), (1, 2, 'slope', 'success')]
        for attempt in attempts:
            result, before = emit(metrics, writer, *attempt)
            rows.extend(result)
            if len(rows) == 1:
                assert before == 0 and read(directory, 'power') == []
            assert not writer.pending
        writer.write_summary(rows)
        metadata = dict(status='complete', group='g0_mlp', checkpoint='model.pt', recorded_attempts=6,
                        run_id=writer.run_id, trial_annotation_schema=1,
                        protocol=dict(data_mode='first_successes_split', metric_successes_per_terrain=limit,
                                      terrain_types=['flat', 'slope'], envs_per_terrain=1, episodes_per_env=3,
                                      episodes_per_terrain=3, total_attempts=6, seed=10,
                                      command_body_vx_vy_wz=[1., 0., 0.]),
                        environment_config={'control': {'decimation': 4}}, joint_names=['a', 'b'],
                        physics_dt_s=.005, control_dt_s=.02)
        (Path(directory) / 'metadata.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('limit,selected', [(0, 0), (1, 2), (10, 4)])
def test_per_terrain_selection_keeps_outcomes_and_splits_raw_inputs(limit, selected):
    with TemporaryDirectory() as directory:
        capture(directory, limit)
        report = validate(directory)
        assert report['episodes'] == 6 and report['metric_trials_selected'] == selected
        assert report['physics_samples'] == selected * 4
        assert report['control_samples'] == selected
        assert not (Path(directory) / 'physics.csv').exists()
        assert not (Path(directory) / 'trajectory.csv').exists()
        episodes = read(directory, 'episodes')
        timing = read(directory, 'traversal_time')
        assert len(timing) == 6
        for row in timing:
            if row['status'] == 'success':
                assert float(row['traversal_time_s']) == pytest.approx(.02)
                assert float(row['total_time_to_success_s']) == pytest.approx(.04)
            else:
                assert row['traversal_time_s'] == row['total_time_to_success_s'] == ''
        ids = {row['trial_id'] for row in episodes if row['metrics_selected'] == 'True'}
        for name in ('power', 'cot', 'tracking', 'stability'):
            assert {row['trial_id'] for row in read(directory, name)} == ids
            assert all('metadata_json' not in row for row in read(directory, name))
        rates = read(directory, 'success_rate')
        assert float(rates[-1]['success_rate']) == pytest.approx(4 / 6)
        # Exact physical inputs survive buffering; no downsampling/integration.
        if selected:
            assert float(read(directory, 'power')[0]['torque_j00_nm']) == 1.
            assert float(read(directory, 'cot')[0]['mass_kg']) == 20.
            for name in ('energy_net_trap_j', 'energy_abs_trap_j', 'energy_positive_trap_j'):
                assert float(read(directory, 'cot')[0][name]) == pytest.approx(.04)
            assert float(read(directory, 'tracking')[0]['command_vx_mps']) == 1.


def test_full_quota_discards_other_inflight_candidates_and_interrupts():
    with TemporaryDirectory() as directory:
        with SelectiveCsvEvaluationWriter(directory, num_joints=2, successes_per_terrain=1) as writer:
            metrics = VectorEpisodeMetrics(2, 'cpu', num_joints=2,
                                           physics_sink=writer.write_physics, trajectory_sink=writer.write_trajectories)
            metrics.start([0, 1], position_world=[[0., 0., 0.]] * 2, velocity_world=[[0., 0., 0.]] * 2,
                          goal_direction_xy=[[1., 0.]] * 2, mass_kg=[20., 20.], gravity_m_s2=9.81,
                          metadata=[dict(task='g0_mlp', terrain='flat', entry_time_s=0., travel_direction=1)] * 2)
            metrics.physics_step(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), .005)
            assert len(writer.pending) == 2
            assert read(directory, 'power') == []
            writer.write_episodes(metrics.finish([1], 'success'))
            assert not writer.pending
            writer.write_episodes(metrics.finish([0], 'interrupted'))
            assert len(read(directory, 'power')) == 1
            assert [row['status'] for row in read(directory, 'episodes')] == ['success', 'interrupted']


def test_wrong_success_selection_or_misaligned_metric_rows_are_rejected():
    for corruption in ('selection', 'alignment', 'energy', 'timing'):
        with TemporaryDirectory() as directory:
            capture(directory)
            name = {'selection': 'episodes', 'alignment': 'tracking', 'energy': 'cot',
                    'timing': 'traversal_time'}[corruption]
            rows = read(directory, name)
            field, value = {'selection': ('metrics_selected', 'True'), 'alignment': ('trial_id', 'wrong'),
                            'energy': ('energy_net_trap_j', '999'),
                            'timing': ('traversal_time_s', '999')}[corruption]
            rows[1 if corruption == 'timing' else 0][field] = value
            # Keep index consistent so selection validation itself is tested.
            if corruption == 'selection':
                for companion in ('trial_index', 'traversal_time'):
                    index = read(directory, companion)
                    index[0]['metrics_selected'] = 'True'
                    with (Path(directory) / (companion + '.csv')).open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
                        writer.writeheader()
                        writer.writerows(index)
            with (Path(directory) / (name + '.csv')).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with pytest.raises(ValueError, match='Incorrect first-success selection|alignment differs|COT energy differs|Traversal time differs'):
                validate(directory)


def test_cot_keeps_net_absolute_and_positive_work_distinct():
    with TemporaryDirectory() as directory:
        with SelectiveCsvEvaluationWriter(directory, num_joints=2) as writer:
            metrics = VectorEpisodeMetrics(2, 'cpu', num_joints=2,
                                           physics_sink=writer.write_physics, trajectory_sink=writer.write_trajectories)
            emit(metrics, writer, 0, 0, 'flat', 'success', torque=torch.tensor([[2., -3.]] * 2))
            row = read(directory, 'cot')[0]
            assert float(row['energy_net_trap_j']) == pytest.approx(-.02)
            assert float(row['energy_abs_trap_j']) == pytest.approx(.1)
            assert float(row['energy_positive_trap_j']) == pytest.approx(.04)
