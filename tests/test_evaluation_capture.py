"""Validate per-terrain quotas using temporary synthetic CSVs, without simulation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from legged_gym.evaluation.metrics import CsvEvaluationWriter
from legged_gym.scripts.validate_evaluation_capture import validate


@pytest.fixture
def capture_data():
    terrains = ["flat", "slope"]
    metadata = dict(
        status="complete", group="g0_mlp", checkpoint="/synthetic/model_20000.pt",
        recorded_attempts=1000,
        protocol=dict(terrain_types=terrains, envs_per_terrain=20,
                      episodes_per_env=25, episodes_per_terrain=500,
                      total_attempts=1000, seed=10, command_body_vx_vy_wz=[1., 0., 0.]),
        environment_config={"control": {"decimation": 4}},
        joint_names=["joint_%02d" % j for j in range(16)],
        physics_dt_s=.005, control_dt_s=.02,
    )
    rows = []
    for env_id in range(40):
        terrain_id = env_id % len(terrains)
        for episode_index in range(25):
            failed = (env_id + episode_index) % 2 == 0
            rows.append(dict(
                task=metadata["group"], checkpoint=metadata["checkpoint"], eval_seed=10,
                env_id=env_id, episode_index=episode_index,
                terrain=terrains[terrain_id], terrain_id=terrain_id,
                status="failure" if failed else "timeout",
                reason="fall" if failed else "time_limit",
                window_started=False, has_physics_samples=False,
                entry_time_s=None, exit_time_s=None,
                elapsed_s=None, physics_elapsed_s=None,
                trial_elapsed_s=1. if failed else 30.,
                control_sample_count=0, physics_sample_count=0,
                mass_kg=20., gravity_m_s2=9.81, travel_direction=1,
            ))
    with TemporaryDirectory(prefix="lzhmine_capture_test_") as temporary:
        yield Path(temporary), metadata, rows


def write_capture(directory, metadata, rows):
    with CsvEvaluationWriter(directory, num_joints=len(metadata["joint_names"])) as writer:
        metadata.update(run_id=writer.run_id, trial_annotation_schema=1)
        writer.write_episodes(rows)
        writer.write_summary(rows)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize("legacy", [False, True], ids=["explicit-terrain-quota", "legacy-slot-quota"])
def test_500_completed_attempts_per_terrain_validate(capture_data, legacy):
    directory, metadata, rows = capture_data
    if legacy:
        del metadata["protocol"]["episodes_per_terrain"]
    write_capture(directory, metadata, rows)
    if legacy:
        del metadata["trial_annotation_schema"]
        del metadata["run_id"]
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = validate(directory)

    assert result["status"] == "passed"
    assert result["episodes"] == result["unstarted_attempts"] == 1000
    assert result["terrains"] == {"flat": 500, "slope": 500}
    assert result["statuses"] == {"failure": 500, "timeout": 500}
    assert result["physics_samples"] == result["control_samples"] == 0


def test_terrain_quota_must_equal_parallel_slot_quotas(capture_data):
    directory, metadata, rows = capture_data
    metadata["protocol"]["episodes_per_terrain"] = 501
    write_capture(directory, metadata, rows)

    with pytest.raises(ValueError, match="Terrain quota differs from parallel-slot quotas"):
        validate(directory)


def test_unequal_terrain_counts_rejected_even_when_total_is_correct(capture_data):
    directory, metadata, rows = capture_data
    rows[0].update(terrain="slope", terrain_id=1)
    write_capture(directory, metadata, rows)

    with pytest.raises(ValueError, match="Unequal attempt counts per terrain"):
        validate(directory)


def test_missing_attempt_rejected(capture_data):
    directory, metadata, rows = capture_data
    rows.pop()
    write_capture(directory, metadata, rows)

    with pytest.raises(ValueError, match="Duplicate or missing episodes"):
        validate(directory)


@pytest.mark.parametrize('field,value', [('trial_id', 'wrong-trial'), ('status', 'success')])
def test_mislabeled_trial_index_is_rejected(capture_data, field, value):
    import csv
    directory, metadata, rows = capture_data
    write_capture(directory, metadata, rows)
    path = directory / 'trial_index.csv'
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        index = list(reader)
    assert len({row['trial_id'] for row in index}) == 1000
    index[0][field] = value
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index)
    with pytest.raises(ValueError, match='Trial index differs'):
        validate(directory)


def test_repeated_capture_gets_distinct_trial_ids():
    with TemporaryDirectory() as first, TemporaryDirectory() as second:
        row = dict(task='g7_all_moe', terrain='slope', env_id=0, episode_index=0)
        from legged_gym.evaluation.metrics import trial_id_for
        with CsvEvaluationWriter(first) as one, CsvEvaluationWriter(second) as two:
            assert trial_id_for(one.run_id, row) != trial_id_for(two.run_id, row)
