"""Bounded candidates selected by first successes or predeclared trial indices.

Selection follows episode-sink completion order (environment ID breaks ties
within one control step). All attempts still enter the success-rate denominator.
"""
from array import array
from collections import Counter
import csv

from .metrics import (CsvEvaluationWriter, ID_FIELDS, EPISODE_FIELDS, TRIAL_INDEX_FIELDS,
                      TRAJECTORY_VALUE_FIELDS, PHYSICS_BASE_FIELDS, physics_fields)
from .timing import TIMING_FIELDS
from .experts import (GATE_CLOCK, expert_fields, weight_columns,
                      residual_fields, residual_value_fields)
from .presets import PATCH_SUPPORTED_TERRAIN_TYPES

SELECTION_FIELDS = ['metrics_selected', 'success_rank', 'recorded_control_samples', 'recorded_physics_samples']
CONTROL_CLOCK = ['control_step', 'time_start_s', 'time_end_s', 'elapsed_s', 'control_dt_s',
                 'trial_time_start_s', 'trial_time_end_s']
COT_VALUES = [prefix + 'world_' + axis + '_m' for prefix in ('', 'previous_') for axis in 'xyz']
ENERGY_FIELDS = ['energy_net_trap_j', 'energy_abs_trap_j', 'energy_positive_trap_j']
TRACKING_VALUES = ['body_v' + axis + '_mps' for axis in 'xyz'] + [
    'body_w' + axis + '_radps' for axis in 'xyz'] + ['command_vx_mps', 'command_vy_mps', 'command_wz_radps']
STABILITY_VALUES = [name for name in TRAJECTORY_VALUE_FIELDS
                    if name not in CONTROL_CLOCK + COT_VALUES + TRACKING_VALUES]


class SelectiveCsvEvaluationWriter(CsvEvaluationWriter):
    def __init__(self, output_dir, num_joints=16, run_id=None, successes_per_terrain=10,
                 expert_counts=None, terrains=(), record_residual=False,
                 detail_selection='first_successes', detail_trial_indices=()):
        if int(successes_per_terrain) != successes_per_terrain or successes_per_terrain < 0:
            raise ValueError('successes_per_terrain must be a nonnegative integer')
        self.limit = int(successes_per_terrain)
        self.success_counts = Counter()
        self.selected_counts = Counter()
        self.pending = {}
        self.record_residual = bool(record_residual)
        self.detail_selection = detail_selection
        self.detail_trial_indices = frozenset(detail_trial_indices)
        if (detail_selection not in ('first_successes', 'trial_ids') or
                (detail_selection == 'trial_ids' and not self.detail_trial_indices) or
                any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in self.detail_trial_indices)):
            raise ValueError('Invalid detailed trial selection')
        self.expert_counts = dict(expert_counts or {})
        self.terrains = tuple(terrains)
        if self.expert_counts and (set(self.expert_counts) != {'student', 'actor'} or
                any(not isinstance(n, int) or n < 1 for n in self.expert_counts.values()) or
                not self.terrains or len(set(self.terrains)) != len(self.terrains) or
                set(self.terrains) - set(PATCH_SUPPORTED_TERRAIN_TYPES)):
            raise ValueError('Expert capture requires Student/Actor counts and unique known terrains')
        super().__init__(output_dir, num_joints=num_joints, run_id=run_id)
        self.physics_values = [name for name in physics_fields(num_joints)
                               if name not in ID_FIELDS + ['selected_side']]

    def file_specs(self, save_trajectories):
        shared = [('episodes', EPISODE_FIELDS + SELECTION_FIELDS),
                ('trial_index', TRIAL_INDEX_FIELDS + SELECTION_FIELDS),
                ('traversal_time', TIMING_FIELDS)]
        if self.record_residual:
            shared.append(('residual', residual_fields(self.num_joints)))
        if self.expert_counts:
            return shared + [('experts/{}/{}_weights'.format(terrain, module), expert_fields(count))
                             for terrain in self.terrains for module, count in self.expert_counts.items()]
        return shared + [('power', physics_fields(self.num_joints)),
                ('cot', ID_FIELDS + CONTROL_CLOCK + COT_VALUES + ENERGY_FIELDS +
                 ['mass_kg', 'gravity_m_s2', 'goal_direction_x', 'goal_direction_y']),
                ('tracking', ID_FIELDS + CONTROL_CLOCK + TRACKING_VALUES),
                ('stability', ID_FIELDS + CONTROL_CLOCK + STABILITY_VALUES)]

    @staticmethod
    def identity(row):
        return (row['task'], row['terrain'], int(row['env_id']), int(row['episode_index']))

    def _buffer(self, kind, rows, fields):
        if self._closed:
            raise ValueError('evaluation writer is closed')
        for row in rows:
            identity = self.identity(row)
            if not self._keep_candidate(identity):
                continue
            streams = self.pending.setdefault(identity, {})
            # Packed doubles avoid a Python dictionary/string copy per sample.
            streams.setdefault(kind, array('d')).extend(float(row[name]) for name in fields)

    def _keep_candidate(self, identity):
        if self.detail_selection == 'trial_ids':
            return identity[3] in self.detail_trial_indices
        return self.success_counts[identity[:2]] < self.limit

    def write_physics(self, rows):
        self._buffer('physics', rows, self.physics_values)

    def write_trajectories(self, rows):
        self._buffer('trajectory', rows, TRAJECTORY_VALUE_FIELDS)

    def write_expert_weights(self, module, rows):
        fields = GATE_CLOCK + weight_columns(self.expert_counts[module])
        self._buffer('gate_' + module, rows, fields)

    def write_residual_diagnostics(self, rows):
        if not self.record_residual:
            raise ValueError('Residual capture was not enabled')
        self._buffer('residual', rows, GATE_CLOCK + residual_value_fields(self.num_joints))

    @staticmethod
    def _unpack(values, fields, episode):
        base = {name: episode.get(name) for name in ID_FIELDS + PHYSICS_BASE_FIELDS}
        for offset in range(0, len(values), len(fields)):
            row = dict(base, **dict(zip(fields, values[offset:offset + len(fields)])))
            for name in ('control_step', 'physics_step', 'substep_index'):
                if name in row and row[name] is not None:
                    row[name] = int(row[name])
            yield row

    def write_episodes(self, rows):
        for episode in rows:
            identity = self.identity(episode)
            group = identity[:2]
            successful = episode['status'] == 'success'
            if successful:
                self.success_counts[group] += 1
            rank = self.success_counts[group] if successful else None
            selected = (identity[3] in self.detail_trial_indices if self.detail_selection == 'trial_ids'
                        else successful and rank <= self.limit)
            if selected:
                self.selected_counts[group] += 1
            streams = self.pending.pop(identity, {})
            physics = streams.get('physics', array('d'))
            trajectory = streams.get('trajectory', array('d'))
            physics_count = len(physics) // len(self.physics_values) if selected else 0
            control_count = len(trajectory) // len(TRAJECTORY_VALUE_FIELDS) if selected else 0
            if selected and self.expert_counts:
                control_count = episode['control_sample_count']
                for module, count in self.expert_counts.items():
                    fields = GATE_CLOCK + weight_columns(count)
                    values = streams.get('gate_' + module, array('d'))
                    if len(values) != control_count * len(fields):
                        raise ValueError('Incomplete expert gate stream for selected trial')
                    name = 'experts/{}/{}_weights'.format(episode['terrain'], module)
                    self._write(name, self._unpack(values, fields, episode))
            if selected and self.record_residual:
                fields = GATE_CLOCK + residual_value_fields(self.num_joints)
                values = streams.get('residual', array('d'))
                if len(values) != episode['control_sample_count'] * len(fields):
                    raise ValueError('Incomplete residual stream for selected trial')
                self._write('residual', self._unpack(values, fields, episode))
            if selected and not self.expert_counts:
                if (physics_count != episode['physics_sample_count'] or
                        control_count != episode['control_sample_count']):
                    raise ValueError('Incomplete candidate buffers for selected trial')
                self._write('power', self._unpack(physics, self.physics_values, episode))
                energy = {}
                for sample in self._unpack(physics, self.physics_values, episode):
                    totals = energy.setdefault(sample['control_step'], [0., 0., 0.])
                    for joint in range(self.num_joints):
                        torque = sample['torque_j%02d_nm' % joint]
                        for endpoint in ('before', 'after'):
                            power = torque * sample['joint_vel_%s_j%02d_radps' % (endpoint, joint)]
                            factor = .5 * sample['dt_s']
                            totals[0] += power * factor
                            totals[1] += abs(power) * factor
                            totals[2] += max(power, 0.) * factor
                # Each selected control stream feeds separate metric-input files.
                for name in ('cot', 'tracking', 'stability'):
                    def samples():
                        for row in self._unpack(trajectory, TRAJECTORY_VALUE_FIELDS, episode):
                            row.update({key: episode.get(key) for key in
                                        ('mass_kg', 'gravity_m_s2', 'goal_direction_x', 'goal_direction_y')})
                            row.update(zip(ENERGY_FIELDS, energy[row['control_step']]))
                            yield row
                    self._write(name, samples())
            labeled = dict(episode, metrics_selected=selected, success_rank=rank,
                           recorded_control_samples=control_count, recorded_physics_samples=physics_count)
            super().write_episodes([labeled])
            if self.detail_selection == 'first_successes' and self.success_counts[group] >= self.limit:
                # Other candidates of this terrain can no longer be selected.
                for key in [key for key in self.pending if key[:2] == group]:
                    del self.pending[key]

    def write_summary(self, episodes):
        rows = super().write_summary(episodes)
        with (self.output_dir / 'success_rate.csv').open('w', newline='', encoding='utf-8') as handle:
            fields = list(rows[0]) + ['success_rate', 'metric_trials_selected']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                selected = (sum(self.selected_counts.values())
                            if row['scope'] == 'overall' else
                            sum(count for (_, terrain), count in self.selected_counts.items()
                                if terrain == row['terrain']))
                writer.writerow(dict(row, success_rate=(row['success'] / row['completed_episodes']
                                                        if row['completed_episodes'] else None),
                                     metric_trials_selected=selected))
        return rows

    def close(self):
        self.pending.clear()
        super().close()
