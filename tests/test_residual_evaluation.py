"""Residual factorial configuration and the actions actually executed in evaluation."""
import csv
import json
from types import SimpleNamespace as NS

import pytest
import torch

from experiments.registry import (EXPERIMENT_KEYS, LEGACY_EXPERIMENT_KEYS, RESIDUAL_EXPERIMENT_KEYS,
                                  configure_experiment, get_experiment)
from legged_gym.evaluation.config import parse_args, protocol_dict, selected_groups
from legged_gym.evaluation.experts import (act_with_expert_weights, apply_residual_scale,
                                          record_expert_step, record_residual_step)
from legged_gym.evaluation.metrics import VectorEpisodeMetrics
from legged_gym.evaluation.selective_writer import SelectiveCsvEvaluationWriter
from legged_gym.scripts.evaluate_nezha_moe_cts import child_command
from legged_gym.scripts.validate_evaluation_capture import validate
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_four_new_arms_change_only_residual_temporal_and_metadata():
    assert len(EXPERIMENT_KEYS) == 12
    assert selected_groups('all') == LEGACY_EXPERIMENT_KEYS
    assert selected_groups('residual') == RESIDUAL_EXPERIMENT_KEYS
    base = NS(policy=NS(actor_type='moe'), algorithm=NS(learning_rate=.001), runner=NS())
    for key, residual, temporal in zip(RESIDUAL_EXPERIMENT_KEYS,
                                     (False, True, False, True), (False, False, True, True)):
        experiment = get_experiment(key)
        cfg = configure_experiment(base, key)
        assert cfg.policy.residual_enabled == residual
        assert cfg.algorithm.temporal_loss_coef == (.01 if temporal else 0.)
        assert cfg.algorithm.learning_rate == .001
        assert all(getattr(cfg.policy, name) == 'moe'
                   for name in ('student_encoder_type', 'actor_type', 'critic_type'))
        assert all(getattr(cfg.policy, name) == 8
                   for name in ('expert_num', 'actor_expert_num', 'critic_expert_num'))
        assert cfg.runner.experiment_key == key
        assert experiment.TASK_NAME == 'nezha_cts_' + key
        if residual:
            assert cfg.policy.residual_action_scales == [.1, .1, .1, .05] * 4
        else:
            assert cfg.policy.residual_action_scales is None
    assert not hasattr(base.policy, 'residual_enabled')
    assert not hasattr(base.algorithm, 'temporal_loss_coef')


def test_new_capture_options_round_trip_to_child_process():
    args = parse_args(['--group', 'residual', '--expert-weights-only', '--record-residual',
                       '--residual-scale', '0', '--detail-selection', 'trial_ids',
                       '--detail-trial-indices', '2,0'])
    child = parse_args(child_command(args, RESIDUAL_EXPERIMENT_KEYS[1], '/tmp/model.pt', '/tmp/out')[2:])
    assert protocol_dict(args) == protocol_dict(child)
    assert args.detail_trial_indices == (0, 2)


@pytest.mark.parametrize('options', [
    ['--residual-scale', 'nan'], ['--residual-scale', '-.1'], ['--residual-scale', '1.1'],
    ['--detail-trial-indices', '0'],
    ['--detail-selection', 'trial_ids', '--detail-trial-indices', '0,0'],
    ['--detail-selection', 'trial_ids', '--detail-trial-indices', '-1'],
    ['--detail-selection', 'trial_ids', '--detail-trial-indices', '25'],
    ['--detail-selection', 'trial_ids', '--detail-trial-indices', 'a'],
    ['--group', 'residual', '--checkpoint-path', '/tmp/model.pt'],
    ['--record-residual', '--no-record'],
])
def test_invalid_residual_capture_options_fail_before_simulator(options):
    with pytest.raises(SystemExit):
        parse_args(options)


def test_expert_mode_executes_nonzero_compensation_from_same_forward_pass():
    policy = ActorCriticMoECTS(num_actor_obs=12, num_critic_obs=5, num_one_step_obs=3,
        num_actions=2, observation_dims=(1, 1, 1), actor_type='moe', critic_type='moe',
        actor_hidden_dims=[16], critic_hidden_dims=[16], student_encoder_hidden_dims=[16],
        teacher_encoder_hidden_dims=[16], residual_enabled=True, residual_hidden_dims=[8],
        residual_action_scales=[.1, .05]).eval()
    with torch.no_grad():
        list(policy.actor.residual.network.children())[-1].bias.fill_(1.)
    obs = torch.randn(3, 12)
    expected, student, actor, pre, correction = policy.act_inference_with_diagnostics(obs)
    actual, gates = act_with_expert_weights(policy, obs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, pre + correction)
    assert correction.abs().sum() > 0
    torch.testing.assert_close(gates['student'], student)
    torch.testing.assert_close(gates['actor'], actor)
    apply_residual_scale(policy, 0.)
    torch.testing.assert_close(policy.act_inference(obs), pre)


def capture_fixed_trials(directory, experts_only):
    counts = {'student': 8, 'actor': 8} if experts_only else {}
    with SelectiveCsvEvaluationWriter(directory, num_joints=2, expert_counts=counts,
            terrains=['flat', 'slope'], record_residual=True,
            detail_selection='trial_ids', detail_trial_indices=[0]) as writer:
        metrics = VectorEpisodeMetrics(2, 'cpu', num_joints=2,
            physics_sink=None if experts_only else writer.write_physics,
            trajectory_sink=None if experts_only else writer.write_trajectories)
        env = NS(metrics=metrics, num_envs=2, dt=.02, wheel_indices=torch.tensor([1]),
                 cfg=NS(normalization=NS(clip_actions=1.)))
        rows = []
        gates = {'student': torch.full((2, 8), .125), 'actor': torch.full((2, 8), .125)}
        for attempt in range(2):
            metrics.start([0, 1], position_world=[[0., 0., 0.]] * 2,
                velocity_world=[[1., 0., 0.]] * 2, goal_direction_xy=[[1., 0.]] * 2,
                mass_kg=[20., 20.], gravity_m_s2=9.81,
                metadata=[dict(task='g9_all_moe_residual', checkpoint='model.pt', terrain=terrain,
                    terrain_id=i, episode_index=attempt, eval_seed=10, entry_time_s=.1,
                    exit_time_s=.12, trial_elapsed_s=.12, travel_direction=1, selected_side='all')
                    for i, terrain in enumerate(['flat', 'slope'])])
            pre = torch.tensor([[.2, 1.1]]).repeat(2, 1)
            delta = torch.tensor([[.1, -.025]]).repeat(2, 1)
            record_residual_step(writer, env, pre, delta, pre + delta, [.1, .05])
            if experts_only:
                record_expert_step(writer, env, gates)
            for _ in range(4):
                metrics.physics_step(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), .005)
            velocity = torch.tensor([[1., 0., 0.]] * 2, dtype=torch.float64)
            metrics.control_step(velocity * .02, velocity, velocity, torch.zeros(2, 3), velocity, .02,
                orientation_world=torch.tensor([[0., 0., 0., 1.]] * 2), angular_velocity_world=torch.zeros(2, 3))
            finished = metrics.finish([0, 1], ['failure', 'success'] if attempt == 0 else 'success')
            writer.write_episodes(finished)
            rows.extend(finished)
            assert not writer.pending
        writer.write_summary(rows)
        metadata = dict(status='complete', group='g9_all_moe_residual', checkpoint='model.pt',
            recorded_attempts=4, expert_counts=counts, run_id=writer.run_id, trial_annotation_schema=1,
            protocol=dict(data_mode='expert_weights_only' if experts_only else 'first_successes_split',
                metric_successes_per_terrain=10, detail_selection='trial_ids', detail_trial_indices=[0],
                record_residual=True, terrain_types=['flat', 'slope'], envs_per_terrain=1,
                episodes_per_env=2, episodes_per_terrain=2, total_attempts=4, seed=10,
                command_body_vx_vy_wz=[1., 0., 0.]),
            environment_config={'control': {'decimation': 4}}, joint_names=['leg', 'wheel'],
            effective_residual_action_scales=[.1, .05],
            residual_diagnostics=dict(wheel_indices=[1], saturation_ratio=.95, clip_actions=1.),
            physics_dt_s=.005, control_dt_s=.02)
        (directory / 'metadata.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('experts_only', [False, True])
def test_fixed_trials_include_failure_and_record_executed_residual_means(tmp_path, experts_only):
    capture_fixed_trials(tmp_path, experts_only)
    report = validate(tmp_path)
    assert report['episodes'] == 4 and report['metric_trials_selected'] == 2
    assert report['residual_samples'] == 2
    rows = read(tmp_path / 'residual.csv')
    assert {row['episode_index'] for row in rows} == {'0'}
    assert {row['terrain'] for row in rows} == {'flat', 'slope'}
    assert float(rows[0]['residual_leg_rms']) == pytest.approx(.1)
    assert float(rows[0]['residual_wheel_rms']) == pytest.approx(.025)
    assert float(rows[0]['residual_saturation_fraction']) == .5
    assert float(rows[0]['action_clip_fraction']) == .5
    episodes = read(tmp_path / 'episodes.csv')
    assert episodes[0]['status'] == 'failure' and episodes[0]['metrics_selected'] == 'True'
    assert episodes[-1]['status'] == 'success' and episodes[-1]['metrics_selected'] == 'False'


def test_validator_detects_residual_action_mismatch(tmp_path):
    capture_fixed_trials(tmp_path, True)
    path = tmp_path / 'residual.csv'
    rows = read(path)
    rows[0]['final_mean_j00'] = '99'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='Residual final mean mismatch'):
        validate(tmp_path)
