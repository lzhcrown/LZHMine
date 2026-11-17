# Adapted from wheel_gym_CQ/scripts/validate_evaluation_capture.py
"""Stream-check raw CSV completeness and timing without computing performance metrics."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from itertools import zip_longest
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-8)


def key(row):
    return int(row["env_id"]), int(row["episode_index"])


def sample_rows(directory, kind, split):
    names = (['power'] if kind == 'physics' else ['cot', 'tracking', 'stability']) if split else [kind]
    with ExitStack() as stack:
        readers = [csv.DictReader(stack.enter_context((directory / (name + '.csv')).open(newline='')))
                   for name in names]
        for parts in zip_longest(*readers):
            require(all(part is not None for part in parts), 'Metric CSV row counts differ')
            merged = {}
            for part in parts:
                require(all(merged[name] == part[name] for name in merged.keys() & part.keys()),
                        'Metric CSV trial/clock alignment differs')
                merged.update(part)
            yield merged


def validate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata["status"] == "complete", "Collection is not complete")
    protocol = metadata["protocol"]
    experts_only = protocol.get('data_mode') == 'expert_weights_only'
    split = protocol.get('data_mode') in ('first_successes_split', 'expert_weights_only')
    with (directory / "episodes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    episodes = {key(row): row for row in rows}
    require(len(episodes) == len(rows) == protocol["total_attempts"] == metadata["recorded_attempts"],
            "Duplicate or missing episodes")
    annotated = metadata.get("trial_annotation_schema") == 1
    if annotated:
        require(bool(metadata.get("run_id")), "Missing capture run ID")
        require(all(row.get("run_id") == metadata["run_id"] and row.get("trial_id") for row in rows),
                "Missing or inconsistent trial labels")
        require(len({row["trial_id"] for row in rows}) == len(rows), "Duplicate trial ID")
        with (directory / "trial_index.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            require({"run_id", "trial_id", "task", "terrain", "env_id", "episode_index", "status", "reason",
                     "control_sample_count", "physics_sample_count"}.issubset(reader.fieldnames or []),
                    "Missing trial index columns")
            index_rows = list(reader)
        require(len(index_rows) == len(rows) and {key(row) for row in index_rows} == set(episodes),
                "Duplicate or missing trial index rows")
        for row in index_rows:
            episode = episodes[key(row)]
            require(all(row[field] == episode[field] for field in row.keys() & episode.keys()),
                    "Trial index differs from episode record")
    counts = Counter(row["terrain"] for row in rows)
    require(set(counts) == set(protocol["terrain_types"]), "Terrain set mismatch")
    slot_total = protocol["envs_per_terrain"] * protocol["episodes_per_env"]
    terrain_total = protocol.get("episodes_per_terrain", slot_total)
    require(terrain_total == slot_total, "Terrain quota differs from parallel-slot quotas")
    require(set(counts.values()) == {terrain_total},
            "Unequal attempt counts per terrain")
    require(set(episodes) == {(i, j) for i in range(len(counts) * protocol["envs_per_terrain"])
                             for j in range(protocol["episodes_per_env"])}, "Environment/episode ID mismatch")
    for row in rows:
        require(row["status"] in ("success", "failure", "timeout"), "Invalid terminal status")
        require(float(row["mass_kg"]) > 0 and float(row["gravity_m_s2"]) > 0, "Invalid mass/gravity")
        if row["window_started"] == "True":
            require(close(row["elapsed_s"], row["physics_elapsed_s"]), "Physics/control duration mismatch")
        else:
            require(row["window_started"] == "False" and row["elapsed_s"] == row["physics_elapsed_s"] == "" and
                    int(row["control_sample_count"]) == int(row["physics_sample_count"]) == 0,
                    "Unstarted attempt must have no measured duration or samples")
        require(row["status"] != "success" or row["window_started"] == "True", "Success without entry")
    timing_path = directory / 'traversal_time.csv'
    if timing_path.exists():
        with timing_path.open(newline='') as handle:
            timing_rows = list(csv.DictReader(handle))
        require(len(timing_rows) == len(rows) and {key(row) for row in timing_rows} == set(episodes),
                'Duplicate or missing traversal-time rows')
        for timing in timing_rows:
            episode = episodes[key(timing)]
            require(all(timing[field] == episode[field] for field in timing.keys() & episode.keys()),
                    'Traversal-time provenance differs from episode')
            if episode['status'] == 'success':
                require(timing['timing_status'] == 'complete' and
                        close(timing['traversal_time_s'], float(episode['exit_time_s']) - float(episode['entry_time_s'])) and
                        close(timing['total_time_to_success_s'], episode['trial_elapsed_s']),
                        'Traversal time differs from event clocks')
            else:
                require(timing['timing_status'] == 'not_successful' and
                        timing['traversal_time_s'] == timing['total_time_to_success_s'] == '',
                        'Unsuccessful attempt must not have a passage time')
    selected = set(episodes)
    if split:
        selected = set()
        successes = Counter()
        limit = protocol['metric_successes_per_terrain']
        selection_mode = protocol.get('detail_selection', 'first_successes')
        trial_indices = protocol.get('detail_trial_indices', [])
        require(selection_mode in ('first_successes', 'trial_ids'), 'Unknown detailed trial selection')
        if selection_mode == 'trial_ids':
            require(bool(trial_indices) and len(set(trial_indices)) == len(trial_indices) and
                    all(isinstance(index, int) and not isinstance(index, bool) and
                        0 <= index < protocol['episodes_per_env'] for index in trial_indices),
                    'Invalid preselected trial indices')
        for row in rows:
            terrain = row['terrain']
            if row['status'] == 'success':
                successes[terrain] += 1
                expected_rank = str(successes[terrain])
            else:
                expected_rank = ''
            keep = (int(row['episode_index']) in trial_indices if selection_mode == 'trial_ids' else
                    bool(expected_rank) and successes[terrain] <= limit)
            require(row.get('success_rank') == expected_rank and row.get('metrics_selected') == str(keep),
                    'Incorrect first-success selection' if selection_mode == 'first_successes' else
                    'Incorrect preselected trial selection')
            for kind in ('control', 'physics'):
                require(int(row['recorded_' + kind + '_samples']) ==
                        (int(row[kind + '_sample_count']) if keep and not (experts_only and kind == 'physics') else 0),
                        'Incorrect recorded sample count')
            if keep:
                selected.add(key(row))
        with (directory / 'success_rate.csv').open(newline='') as handle:
            rate_rows = list(csv.DictReader(handle))
        require(len(rate_rows) == len(counts) + 1 and
                {(row['scope'], row['terrain']) for row in rate_rows} ==
                {('terrain', terrain) for terrain in counts} | {('overall', 'all')},
                'Missing/duplicate success-rate rows')
        for rate in rate_rows:
            group = rows if rate['scope'] == 'overall' else [row for row in rows if row['terrain'] == rate['terrain']]
            totals = Counter(row['status'] for row in group)
            require(all(int(rate[status]) == totals[status] for status in ('success', 'failure', 'timeout', 'interrupted'))
                    and int(rate['completed_episodes']) == len(group)
                    and close(rate['success_rate'], totals['success'] / len(group))
                    and int(rate['metric_trials_selected']) == sum(key(row) in selected for row in group),
                    'Incorrect success-rate summary')
    decimation = metadata["environment_config"]["control"]["decimation"]
    joint_count = len(metadata["joint_names"])
    report = dict(directory=str(directory.resolve()), status="passed", episodes=len(rows),
                  statuses=dict(Counter(row["status"] for row in rows)), terrains=dict(counts),
                  unstarted_attempts=sum(row["window_started"] == "False" for row in rows),
                  control_samples=0, physics_samples=0, files_bytes={})
    energy = {}
    for kind in (() if experts_only else ("physics", "trajectory")):
        names = (['power'] if kind == 'physics' else ['cot', 'tracking', 'stability']) if split else [kind]
        for name in names:
            path = directory / (name + '.csv')
            report['files_bytes'][path.name] = path.stat().st_size
        previous = {}
        seen = Counter()
        if kind == "physics":
            numeric = ["torque_j%02d_nm" % j for j in range(joint_count)] + [
                "joint_vel_%s_j%02d_radps" % (end, j) for end in ("before", "after") for j in range(joint_count)]
            step_name, dt_name, dt = "physics_step", "dt_s", metadata["physics_dt_s"]
        else:
            numeric = [prefix + "world_" + axis + "_m" for prefix in ("", "previous_") for axis in "xyz"] + [
                prefix + "world_v" + axis + "_mps" for prefix in ("", "previous_") for axis in "xyz"] + [
                "body_v" + axis + "_mps" for axis in "xyz"] + [
                "body_w" + axis + "_radps" for axis in "xyz"] + [
                "world_a" + axis + "_mps2" for axis in "xyz"] + [
                "world_q" + axis for axis in "xyzw"] + ["world_w" + axis + "_radps" for axis in "xyz"] + [
                "command_vx_mps", "command_vy_mps", "command_wz_radps",
                "acceleration_xy_mps2", "acceleration_xyz_mps2"]
            step_name, dt_name, dt = "control_step", "control_dt_s", metadata["control_dt_s"]
        # Validate headers even when no success produced any samples.
        header = set()
        for name in names:
            with (directory / (name + '.csv')).open(newline='') as handle:
                header.update(csv.DictReader(handle).fieldnames or [])
        require(set(numeric).issubset(header), 'Missing raw columns: ' + kind)
        for line, row in enumerate(sample_rows(directory, kind, split), 2):
            identity = key(row)
            context = "{}:{} episode={}".format(path.name, line, identity)
            require(identity in episodes, "Unknown episode at " + context)
            require(identity in selected, 'Samples for unselected trial at ' + context)
            episode = episodes[identity]
            if split and kind == 'trajectory':
                require(close(row['mass_kg'], episode['mass_kg']) and
                        close(row['gravity_m_s2'], episode['gravity_m_s2']), 'COT mass/gravity mismatch')
                expected_energy = energy.pop((identity, int(row['control_step'])), None)
                require(expected_energy is not None and all(close(row[field], value) for field, value in zip(
                    ('energy_net_trap_j', 'energy_abs_trap_j', 'energy_positive_trap_j'), expected_energy)),
                    'COT energy differs from recorded mechanical power inputs')
            if annotated:
                require(row.get("run_id") == episode["run_id"] and row.get("trial_id") == episode["trial_id"],
                        "Trial label mismatch at " + context)
            require(row["task"] == episode["task"] == metadata["group"] and
                    row["checkpoint"] == episode["checkpoint"] == metadata["checkpoint"] and
                    row["terrain"] == episode["terrain"] and int(row["eval_seed"]) == protocol["seed"],
                    "Provenance mismatch at " + context)
            require(episode["window_started"] == "True", "Samples before entry at " + context)
            values = {name: float(row[name]) for name in numeric}
            require(all(math.isfinite(x) for x in values.values()), "Nonfinite sample at " + context)
            seen[identity] += 1
            step = seen[identity]
            require(int(row[step_name]) == step and close(row[dt_name], dt) and
                    close(row["time_start_s"], (step - 1) * dt) and close(row["time_end_s"], step * dt),
                    "Missing/duplicate step or bad interval at " + context)
            entry = float(episode["entry_time_s"])
            require(close(row["trial_time_start_s"], entry + (step - 1) * dt) and
                    close(row["trial_time_end_s"], entry + step * dt), "Trial clock mismatch at " + context)
            last = previous.get(identity)
            if kind == "physics":
                if split:
                    totals = energy.setdefault((identity, int(row['control_step'])), [0., 0., 0.])
                    for joint in range(joint_count):
                        torque = values['torque_j%02d_nm' % joint]
                        for endpoint in ('before', 'after'):
                            power = torque * values['joint_vel_%s_j%02d_radps' % (endpoint, joint)]
                            totals[0] += .5 * dt * power
                            totals[1] += .5 * dt * abs(power)
                            totals[2] += .5 * dt * max(power, 0.)
                require(int(row["control_step"]) == (step - 1) // decimation + 1 and
                        int(row["substep_index"]) == (step - 1) % decimation + 1,
                        "Substep mapping mismatch at " + context)
                before = [values["joint_vel_before_j%02d_radps" % j] for j in range(joint_count)]
                if last is not None:
                    require(all(close(x, y) for x, y in zip(before, last)), "Joint-velocity gap at " + context)
                previous[identity] = [values["joint_vel_after_j%02d_radps" % j] for j in range(joint_count)]
            else:
                acceleration = [(values["world_v" + a + "_mps"] - values["previous_world_v" + a + "_mps"]) / dt
                                for a in "xyz"]
                require(all(close(values["world_a" + a + "_mps2"], x) for a, x in zip("xyz", acceleration)) and
                        close(values["acceleration_xy_mps2"], math.hypot(*acceleration[:2])) and
                        close(values["acceleration_xyz_mps2"], math.sqrt(sum(x*x for x in acceleration))),
                        "Acceleration cannot be reconstructed at " + context)
                command = [values[name] for name in ("command_vx_mps", "command_vy_mps", "command_wz_radps")]
                require(all(close(x, y) for x, y in zip(command, protocol["command_body_vx_vy_wz"])),
                        "Unexpected command at " + context)
                old_state = [values["previous_world_" + a + "_m"] for a in "xyz"] + [
                    values["previous_world_v" + a + "_mps"] for a in "xyz"]
                if last is None:
                    last = [float(episode["start_world_" + a + "_m"]) for a in "xyz"] + [
                        float(episode["start_world_v" + a + "_mps"]) for a in "xyz"]
                require(all(close(x, y) for x, y in zip(old_state, last)), "State gap/reset jump at " + context)
                previous[identity] = [values["world_" + a + "_m"] for a in "xyz"] + [
                    values["world_v" + a + "_mps"] for a in "xyz"]
        for identity, episode in episodes.items():
            name = "physics" if kind == "physics" else "control"
            expected_samples = int(episode[name + "_sample_count"]) if identity in selected else 0
            require(seen[identity] == expected_samples, "Sample count mismatch: " + str(identity))
            if episode["window_started"] == "True" and identity in selected:
                require(close(seen[identity] * dt, episode["elapsed_s"]), "Window duration mismatch: " + str(identity))
            if kind == "trajectory" and seen[identity]:
                terminal = [float(episode["end_world_" + a + "_m"]) for a in "xyz"] + [
                    float(episode["end_world_v" + a + "_mps"]) for a in "xyz"]
                require(all(close(x, y) for x, y in zip(previous[identity], terminal)), "Terminal sample mismatch")
        report[("physics" if kind == "physics" else "control") + "_samples"] = sum(seen.values())
        print("Validated {} / {}: {} samples".format(directory.name, kind, sum(seen.values())), flush=True)
    report['metric_trials_selected'] = len(selected)
    if experts_only:
        require(set(metadata['expert_counts']) == {'student', 'actor'} and
                all(isinstance(count, int) and count > 0 for count in metadata['expert_counts'].values()),
                'Expert capture must contain Student and Actor expert counts')
        gate_counts = {}
        for module, count in metadata['expert_counts'].items():
            columns = ['expert_%02d_weight' % i for i in range(count)]
            seen = Counter()
            for terrain in protocol['terrain_types']:
                path = directory / 'experts' / terrain / (module + '_weights.csv')
                report['files_bytes'][str(path.relative_to(directory))] = path.stat().st_size
                with path.open(newline='') as handle:
                    reader = csv.DictReader(handle)
                    require(set(columns).issubset(reader.fieldnames or []), 'Missing expert columns')
                    for row in reader:
                        identity = key(row)
                        require(identity in selected, 'Expert samples for unselected trial')
                        episode = episodes[identity]
                        require(row['trial_id'] == episode['trial_id'] and row['run_id'] == episode['run_id'] and
                                row['terrain'] == episode['terrain'] == terrain and
                                row['task'] == metadata['group'] == episode['task'] and
                                row['checkpoint'] == episode['checkpoint'] == metadata['checkpoint'] and
                                int(row['eval_seed']) == protocol['seed'], 'Expert provenance mismatch')
                        probabilities = [float(row[name]) for name in columns]
                        require(all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities) and
                                math.isclose(sum(probabilities), 1., rel_tol=1e-5, abs_tol=1e-6),
                                'Invalid expert probabilities')
                        seen[identity] += 1
                        step, dt, entry = seen[identity], metadata['control_dt_s'], float(episode['entry_time_s'])
                        require(int(row['control_step']) == step and close(row['control_dt_s'], dt) and
                                close(row['time_start_s'], (step - 1) * dt) and close(row['time_end_s'], step * dt) and
                                close(row['trial_time_start_s'], entry + (step - 1) * dt) and
                                close(row['trial_time_end_s'], entry + step * dt), 'Expert clock/step mismatch')
            for identity, episode in episodes.items():
                require(seen[identity] == (int(episode['control_sample_count']) if identity in selected else 0),
                        'Expert stream sample count mismatch')
            gate_counts[module] = sum(seen.values())
        report['expert_samples'] = gate_counts
        report['control_samples'] = gate_counts['student']
    if protocol.get('record_residual', False):
        path = directory / 'residual.csv'
        report['files_bytes'][path.name] = path.stat().st_size
        columns = [prefix + '_j%02d' % i for prefix in ('pre_mean', 'residual', 'final_mean')
                   for i in range(joint_count)]
        diagnostics = ['residual_leg_rms', 'residual_wheel_rms', 'residual_saturation_fraction',
                       'action_clip_fraction']
        diagnostic_config = metadata.get('residual_diagnostics')
        if diagnostic_config:
            wheels = diagnostic_config['wheel_indices']
            require(len(set(wheels)) == len(wheels) and
                    all(isinstance(i, int) and 0 <= i < joint_count for i in wheels),
                    'Invalid residual wheel indices')
            legs = [i for i in range(joint_count) if i not in wheels]
            scales = metadata.get('effective_residual_action_scales') or [0.] * joint_count
            require(len(scales) == joint_count and all(math.isfinite(x) and x >= 0 for x in scales),
                    'Invalid effective residual scales')
            thresholds = diagnostic_config.get('saturation_action_thresholds') or [
                diagnostic_config['saturation_ratio'] * scale for scale in scales]
            require(len(thresholds) == joint_count and all(math.isfinite(x) and x >= 0 for x in thresholds),
                    'Invalid residual saturation thresholds')
        seen = Counter()
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            require(set(columns + diagnostics).issubset(reader.fieldnames or []), 'Missing residual columns')
            for row in reader:
                identity = key(row)
                require(identity in selected, 'Residual samples for unselected trial')
                episode = episodes[identity]
                require(row['trial_id'] == episode['trial_id'] and row['run_id'] == episode['run_id'] and
                        row['terrain'] == episode['terrain'] and row['task'] == metadata['group'] == episode['task'] and
                        row['checkpoint'] == episode['checkpoint'] == metadata['checkpoint'] and
                        int(row['eval_seed']) == protocol['seed'], 'Residual provenance mismatch')
                require(all(math.isfinite(float(row[name])) for name in columns + diagnostics),
                        'Nonfinite residual diagnostic')
                require(all(math.isclose(float(row['final_mean_j%02d' % i]),
                                         float(row['pre_mean_j%02d' % i]) + float(row['residual_j%02d' % i]),
                                         rel_tol=1e-5, abs_tol=1e-6) for i in range(joint_count)),
                        'Residual final mean mismatch')
                require(all(float(row[name]) >= 0 for name in diagnostics[:2]) and
                        all(0 <= float(row[name]) <= 1 for name in diagnostics[2:]),
                        'Invalid residual diagnostic range')
                if diagnostic_config:
                    delta = [float(row['residual_j%02d' % i]) for i in range(joint_count)]
                    require(all(abs(x) <= scale + 1e-6 for x, scale in zip(delta, scales)),
                            'Residual exceeds recorded action scale')
                    rms = lambda indices: math.sqrt(sum(delta[i] ** 2 for i in indices) / len(indices)) if indices else 0.
                    saturated = [abs(x) >= threshold for x, scale, threshold in zip(delta, scales, thresholds)
                                 if scale > 0]
                    expected = [rms(legs), rms(wheels),
                        sum(saturated) / len(saturated) if saturated else 0.,
                        sum(abs(float(row['final_mean_j%02d' % i])) > diagnostic_config['clip_actions']
                            for i in range(joint_count)) / joint_count]
                    require(all(math.isclose(float(row[name]), value, rel_tol=1e-5, abs_tol=1e-6)
                                for name, value in zip(diagnostics, expected)),
                            'Residual summaries differ from recorded action means')
                seen[identity] += 1
                step, dt, entry = seen[identity], metadata['control_dt_s'], float(episode['entry_time_s'])
                require(int(row['control_step']) == step and close(row['control_dt_s'], dt) and
                        close(row['time_start_s'], (step - 1) * dt) and close(row['time_end_s'], step * dt) and
                        close(row['trial_time_start_s'], entry + (step - 1) * dt) and
                        close(row['trial_time_end_s'], entry + step * dt), 'Residual clock/step mismatch')
        for identity, episode in episodes.items():
            require(seen[identity] == (int(episode['control_sample_count']) if identity in selected else 0),
                    'Residual stream sample count mismatch')
        report['residual_samples'] = sum(seen.values())
    require(not energy, 'COT intervals missing for recorded power inputs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.directory)
    except (ValueError, KeyError) as exc:
        result = dict(directory=str(args.directory.resolve()), status="failed", error=str(exc))
    (args.directory / "data_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
