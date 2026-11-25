"""Passage times from per-attempt event clocks; no trajectory replay required."""
import json
import math


MAX_ATTEMPT_SECONDS = 30.0


def attempt_step_limit(seconds, dt):
    """End at the last whole control step within the requested simulation time."""
    if not 0 < seconds <= MAX_ATTEMPT_SECONDS or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Attempt duration must be in (0, 30] seconds and control dt must be positive")
    steps = seconds / dt
    nearest = round(steps)
    # Gym stores its timestep in float32, including nominal 5 ms physics steps.
    limit = nearest if math.isclose(steps, nearest, rel_tol=1e-7) else math.floor(steps)
    if limit < 1:
        raise ValueError("Attempt duration must cover at least one control step")
    return limit


TIMING_FIELDS = [
    'run_id', 'trial_id', 'task', 'terrain', 'env_id', 'episode_index', 'eval_seed', 'checkpoint',
    'status', 'reason', 'metrics_selected', 'success_rank', 'travel_direction', 'exit_edge',
    'entry_time_s', 'exit_time_s', 'trial_elapsed_s', 'traversal_time_s', 'total_time_to_success_s',
    'timing_status', 'time_resolution_s',
]


def traversal_record(episode, resolution_s=None):
    """Only successful outcomes have a passage time, including zero-duration crossings.

    Clocks are simulation seconds relative to attempt start. The interval starts
    at first wheel entry and ends when the terrain's success condition is met.
    """
    extra = episode.get('metadata_json')
    extra = json.loads(extra) if isinstance(extra, str) and extra else {}
    row = {name: episode.get(name, extra.get(name)) for name in TIMING_FIELDS}
    row.update(traversal_time_s=None, total_time_to_success_s=None,
               time_resolution_s=resolution_s if resolution_s is not None else
               episode.get('measurement_boundary_resolution_s', extra.get('measurement_boundary_resolution_s')))
    if episode.get('status') != 'success':
        row['timing_status'] = 'not_successful'
        return row
    clocks = [episode.get(name) for name in ('entry_time_s', 'exit_time_s', 'trial_elapsed_s')]
    if any(value is None or value == '' for value in clocks):
        row['timing_status'] = 'missing_timestamps'
        return row
    try:
        entry, exit_time, total = map(float, clocks)
    except (TypeError, ValueError):
        row['timing_status'] = 'invalid_timestamps'
        return row
    if (not all(math.isfinite(value) for value in (entry, exit_time, total)) or
            not 0 <= entry <= exit_time or total < exit_time - 1e-8):
        row['timing_status'] = 'invalid_timestamps'
        return row
    row.update(traversal_time_s=exit_time - entry, total_time_to_success_s=total, timing_status='complete')
    return row
