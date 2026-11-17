"""Checkpoint selection and fixed protocol isolation for all CTS ablations."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile

import pytest

from experiments.registry import EXPERIMENT_KEYS, LEGACY_EXPERIMENT_KEYS
from legged_gym.evaluation.config import parse_args, resolve_checkpoint, configure_evaluation, protocol_dict


def test_upload_and_training_run_paths_are_selected_without_crossing_groups():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for index, group in enumerate(EXPERIMENT_KEYS):
            folder = root / group if index % 2 == 0 else root / 'nezha_moe_cts' / group / 'run_a'
            folder.mkdir(parents=True)
            for iteration in (9000, 20000):
                (folder / ('model_%d.pt' % iteration)).touch()
        args = parse_args(['--log-root', str(root), '--checkpoint', '-1'])
        for group in EXPERIMENT_KEYS:
            path = resolve_checkpoint(args, group)
            assert path.name == 'model_20000.pt'
            assert group in path.parts
        extra = root / EXPERIMENT_KEYS[0] / 'run_b'
        extra.mkdir()
        (extra / 'model_20000.pt').touch()
        with pytest.raises(ValueError, match='Ambiguous'):
            resolve_checkpoint(args, EXPERIMENT_KEYS[0])


def test_evaluation_overrides_do_not_mutate_training_snapshot():
    cfg = NS(env=NS(), noise=NS(add_noise=True), init_state=NS(rand_init_dof=True),
        domain_rand=NS(randomize_motor_strength=True, delay=True),
        commands=NS(curriculum=True, heading_command=True, curriculum_profile='legbot_wf', ranges=NS()),
        terrain=NS(generator_profile='legbot_wf', curriculum=True), sim=NS(physx=NS()))
    args = parse_args(['--envs-per-terrain', '2', '--speed', '-1'])
    result = configure_evaluation(cfg, args, 'g0_mlp', Path('/tmp/model_20000.pt'))
    assert result.env.num_envs == 20
    assert result.evaluation.episodes_per_env == 250
    assert result.evaluation.command_vx == -1.
    assert result.commands.curriculum_profile == 'evaluation'
    assert result.terrain.generator_profile == 'evaluation'
    assert result.terrain.slope_angle_deg == 20.
    assert result.terrain.slope_length_m == 6.5
    assert result.terrain.slope_width_m == 5.6
    assert not result.domain_rand.randomize_torque_scale
    assert not result.domain_rand.randomize_motor_strength
    assert not result.domain_rand.delay
    assert cfg.domain_rand.randomize_motor_strength and cfg.domain_rand.delay
    assert cfg.terrain.curriculum and cfg.commands.curriculum
    assert not hasattr(cfg.domain_rand, 'randomize_torque_scale')


@pytest.mark.parametrize('arguments', [
    ['--speed', 'nan'], ['--speed', '0'], ['--max-seconds', 'inf'],
    ['--envs-per-terrain', '0'], ['--spawn-xy-jitter', '.06'],
    ['--terrain-types', 'flat,flat'], ['--checkpoint-path', '/tmp/model.pt'],
    ['--episodes-per-terrain', '0'], ['--episodes-per-terrain', '-500'],
    ['--episodes-per-env', '0'], ['--episodes-per-env', '-1'],
    ['--envs-per-terrain', '16', '--episodes-per-terrain', '500'],
    ['--episodes-per-terrain', '500', '--episodes-per-env', '25'],
    ['--metric-successes-per-terrain', '-1'],
    ['--expert-weights-only'],
    ['--group', 'g0_mlp', '--expert-weights-only'],
    ['--group', 'g7_all_moe', '--expert-weights-only', '--no-record'],
])
def test_invalid_protocol_rejected(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_preview_defaults_and_rejects_output_destination():
    args = parse_args(['--no-record', '--viewer'])
    assert args.envs_per_terrain == 1
    assert args.episodes_per_env == args.episodes_per_terrain == 1
    assert len(args.terrain_types) == 10
    assert args.output_dir is None
    assert protocol_dict(args)['write_files'] is False
    assert protocol_dict(args)['terrain_roi_size_xy_m']['slope'] == [6.5, 5.6]
    with pytest.raises(SystemExit):
        parse_args(['--no-record', '--output-dir', '/tmp/not-a-capture'])


def test_recording_defaults_to_exactly_500_attempts_per_terrain_and_model():
    args = parse_args([])
    protocol = protocol_dict(args)
    assert args.envs_per_terrain == 20
    assert args.episodes_per_env == 25
    assert protocol['episodes_per_terrain'] == 500
    assert protocol['total_attempts'] == 5000
    assert protocol['metric_successes_per_terrain'] == 10
    assert protocol['data_mode'] == 'first_successes_split'
    assert len(LEGACY_EXPERIMENT_KEYS) * protocol['total_attempts'] == 40000


def test_g7_expert_only_command_round_trip():
    from legged_gym.scripts.evaluate_nezha_moe_cts import child_command
    args = parse_args(['--group', 'g7_all_moe', '--expert-weights-only'])
    child = parse_args(child_command(args, 'g7_all_moe', Path('/tmp/model_20000.pt'), Path('/tmp/gates'))[2:])
    assert protocol_dict(args) == protocol_dict(child)
    assert protocol_dict(args)['data_mode'] == 'expert_weights_only'


@pytest.mark.parametrize('slots', [1, 10, 20, 25])
def test_terrain_total_and_batch_child_command_preserve_exact_quota(slots):
    from legged_gym.scripts.evaluate_nezha_moe_cts import child_command
    args = parse_args(['--terrain-types', 'flat,slope', '--envs-per-terrain', str(slots),
                       '--episodes-per-terrain', '500', '--metric-successes-per-terrain', '7'])
    child = parse_args(child_command(args, 'g7_all_moe', Path('/tmp/model_20000.pt'),
                                    Path('/tmp/formal-capture'))[2:])
    for resolved in (args, child):
        assert resolved.envs_per_terrain * resolved.episodes_per_env == 500
        assert protocol_dict(resolved)['total_attempts'] == 1000
        assert resolved.metric_successes_per_terrain == 7
    assert protocol_dict(child) == protocol_dict(args)


@pytest.mark.parametrize('mode', [[], ['--no-record', '--viewer']])
def test_explicit_per_slot_quota_remains_available_and_round_trips(mode):
    from legged_gym.scripts.evaluate_nezha_moe_cts import child_command
    args = parse_args(mode + ['--envs-per-terrain', '16', '--episodes-per-env', '3'])
    child = parse_args(child_command(args, 'g7_all_moe', Path('/tmp/model_20000.pt'),
                                    Path('/tmp/legacy-capture'))[2:])
    assert child.episodes_per_env == 3
    assert child.episodes_per_terrain == 48
    assert protocol_dict(child) == protocol_dict(args)


def test_preview_single_and_batch_never_create_capture_files(tmp_path, monkeypatch):
    from legged_gym.scripts import evaluate_nezha_moe_cts as entry
    from unittest.mock import Mock
    monkeypatch.setattr(entry, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(entry, 'resolve_checkpoint', lambda args, group: tmp_path / group / 'model_20000.pt')
    single = Mock(return_value=0)
    monkeypatch.setattr(entry, 'evaluate_one', single)
    assert entry.main(['--group', 'g0_mlp', '--no-record', '--viewer']) == 0
    assert single.call_args.args[-1] is None
    child = Mock(return_value=NS(returncode=0))
    monkeypatch.setattr(entry.subprocess, 'run', child)
    assert entry.main(['--group', 'all', '--no-record', '--viewer']) == 0
    assert child.call_count == 8
    for call in child.call_args_list:
        command = call.args[0]
        assert '--no-record' in command and '--viewer' in command
        assert '--output-dir' not in command
    assert list(tmp_path.iterdir()) == []
