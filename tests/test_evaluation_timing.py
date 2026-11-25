"""Event-clock timing for all successes, including trials without trajectories."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from legged_gym.evaluation.timing import attempt_step_limit, traversal_record
from legged_gym.scripts.backfill_traversal_times import backfill, source_hash


def episode(**overrides):
    row = dict(run_id='run', trial_id='run/g0/flat/env0000/ep0000', task='g0', terrain='flat',
               env_id=0, episode_index=0, status='success', entry_time_s=.5, exit_time_s=7.5,
               trial_elapsed_s=7.5, elapsed_s=7., metrics_selected=False,
               metadata_json=json.dumps({'exit_edge': 'xmax', 'measurement_boundary_resolution_s': .02}))
    row.update(overrides)
    return row


def test_success_timing_includes_unselected_trials_and_separates_approach():
    row = traversal_record(episode())
    assert row['traversal_time_s'] == 7.
    assert row['total_time_to_success_s'] == 7.5
    assert row['metrics_selected'] is False
    assert row['exit_edge'] == 'xmax' and row['time_resolution_s'] == .02
    assert traversal_record(episode(entry_time_s=7.5))['traversal_time_s'] == 0.


@pytest.mark.parametrize('status', ['failure', 'timeout', 'interrupted'])
def test_unsuccessful_trials_never_get_a_passage_time(status):
    row = traversal_record(episode(status=status))
    assert row['traversal_time_s'] is row['total_time_to_success_s'] is None
    assert row['trial_elapsed_s'] == 7.5
    assert row['timing_status'] == 'not_successful'


@pytest.mark.parametrize('value,expected', [('', 'missing_timestamps'), (None, 'missing_timestamps'),
                                         ('nan', 'invalid_timestamps'), (.4, 'invalid_timestamps')])
def test_missing_or_invalid_success_clock_is_flagged(value, expected):
    row = traversal_record(episode(exit_time_s=value))
    assert row['traversal_time_s'] is None and row['timing_status'] == expected


def create_capture(directory, group, bad=False):
    folder = directory / group
    folder.mkdir()
    rows = [episode(task=group, trial_id='run/' + group + '/success', exit_time_s='' if bad else 7.5),
            episode(task=group, trial_id='run/' + group + '/timeout', status='timeout', episode_index=1)]
    source = folder / 'episodes.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'metadata.json').write_text(json.dumps(dict(status='complete', recorded_attempts=2, control_dt_s=.02)))
    return source


def test_batch_backfill_preserves_sources_and_refuses_overwrite():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        sources = [create_capture(root, group) for group in ['g0', 'g1']]
        before = {path: source_hash(path) for path in sources}
        report = backfill(root)
        assert report['attempts'] == 4 and report['successful_times'] == 2
        assert {path: source_hash(path) for path in sources} == before
        with (root / 'traversal_times_all.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len({row['trial_id'] for row in rows}) == 4
        assert rows[0]['traversal_time_s'] == '7.0'
        assert rows[1]['traversal_time_s'] == ''
        with pytest.raises(FileExistsError):
            backfill(root)


def test_bad_timestamps_do_not_produce_partial_batch_outputs():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        create_capture(root, 'g0')
        create_capture(root, 'g1', bad=True)
        with pytest.raises(ValueError, match='missing/invalid timing'):
            backfill(root)
        assert not list(root.rglob('traversal_time*.csv'))


def test_simulation_time_cap_counts_whole_control_steps():
    assert attempt_step_limit(30, .02) == 1500
    assert attempt_step_limit(30, .004999999888 * 4) == 1500
    assert attempt_step_limit(30, .07) == 428
    assert attempt_step_limit(.041, .02) == 2
    with pytest.raises(ValueError):
        attempt_step_limit(.01, .02)
    with pytest.raises(ValueError):
        attempt_step_limit(31, .02)
