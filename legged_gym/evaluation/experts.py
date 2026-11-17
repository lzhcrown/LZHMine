"""Capture routing and compensation from the same forward pass as the action."""
import math
import torch

from .metrics import ID_FIELDS

GATE_CLOCK = ['control_step', 'time_start_s', 'time_end_s', 'control_dt_s',
              'trial_time_start_s', 'trial_time_end_s']


def weight_columns(count):
    return ['expert_%02d_weight' % index for index in range(count)]


def expert_fields(count):
    return ID_FIELDS + GATE_CLOCK + weight_columns(count)


def residual_value_fields(count):
    return ([prefix + '_j%02d' % index for prefix in ('pre_mean', 'residual', 'final_mean')
             for index in range(count)] +
            ['residual_leg_rms', 'residual_wheel_rms', 'residual_saturation_fraction',
             'action_clip_fraction'])


def residual_fields(count):
    return ID_FIELDS + GATE_CLOCK + residual_value_fields(count)


@torch.no_grad()
def apply_residual_scale(policy, multiplier):
    """Apply once to a newly loaded policy; this is an inference diagnostic."""
    if not math.isfinite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError('Residual multiplier must be finite and in [0, 1]')
    if getattr(policy, 'residual_enabled', False):
        policy.actor.residual_action_scales.mul_(multiplier)


@torch.inference_mode()
def act_with_expert_weights(policy, observations):
    if policy.student_encoder_type != 'moe' or policy.actor_type != 'moe':
        raise ValueError('Expert capture requires MoE Student and Actor')
    actions, student, actor, _, _ = policy.act_inference_with_diagnostics(observations)
    return actions, {'student': student, 'actor': actor}


@torch.inference_mode()
def record_residual_step(writer, env, pre_mean, residual, final_mean, action_scales=None):
    """Record deterministic policy means before the environment clips actions."""
    ids = env.metrics.active.nonzero(as_tuple=False).flatten()
    if not len(ids):
        return
    count = writer.num_joints
    expected = (env.num_envs, count)
    if any(value.shape != expected or not bool(torch.isfinite(value).all())
           for value in (pre_mean, residual, final_mean)):
        raise ValueError('Invalid residual diagnostic shape or values')
    if not bool(torch.allclose(final_mean, pre_mean + residual, atol=1e-6, rtol=1e-5)):
        raise ValueError('Final action mean differs from coarse mean plus residual')
    wheel_indices = torch.as_tensor(env.wheel_indices, device=residual.device, dtype=torch.long)
    leg_mask = torch.ones(count, device=residual.device, dtype=torch.bool)
    leg_mask[wheel_indices] = False
    leg_rms = residual[:, leg_mask].square().mean(-1).sqrt() if bool(leg_mask.any()) else residual.new_zeros(env.num_envs)
    wheel_rms = residual[:, wheel_indices].square().mean(-1).sqrt() if wheel_indices.numel() else residual.new_zeros(env.num_envs)
    saturation = residual.new_zeros(env.num_envs)
    if action_scales is not None:
        scales = torch.as_tensor(action_scales, device=residual.device, dtype=residual.dtype)
        if scales.shape != (count,) or not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
            raise ValueError('Invalid residual action scales')
        if bool((residual.abs() > scales + 1e-6).any()):
            raise ValueError('Residual exceeds its configured action scale')
        enabled = scales > 0
        if bool(enabled.any()):
            saturation = (residual[:, enabled].abs() >= .95 * scales[enabled]).float().mean(-1)
    clip = env.cfg.normalization.clip_actions
    clip_fraction = (final_mean.abs() > clip).float().mean(-1)
    packed = torch.cat((pre_mean, residual, final_mean, leg_rms[:, None], wheel_rms[:, None],
                        saturation[:, None], clip_fraction[:, None]), dim=-1)
    clocks = torch.stack((env.metrics.control_steps.to(torch.float64) + 1,
                          env.metrics.elapsed_s), dim=1)[ids].cpu().tolist()
    rows = []
    for env_id, (step, start), values in zip(ids.cpu().tolist(), clocks, packed[ids].cpu().tolist()):
        row = dict(env.metrics.metadata[env_id])
        entry = float(row['entry_time_s'])
        row.update(control_step=int(step), time_start_s=start, time_end_s=start + env.dt,
                   control_dt_s=env.dt, trial_time_start_s=entry + start,
                   trial_time_end_s=entry + start + env.dt)
        row.update(zip(residual_value_fields(count), values))
        rows.append(row)
    writer.write_residual_diagnostics(rows)


def record_expert_step(writer, env, weights):
    """Match the subsequent measured control interval, including its terminal step.

Call before env.step: automatic resets inside that method retire the trial and
flush successful buffers. The entry-detecting interval is excluded as elsewhere.
"""
    ids = env.metrics.active.nonzero(as_tuple=False).flatten()
    if not len(ids):
        return
    clocks = torch.stack((env.metrics.control_steps.to(torch.float64) + 1,
                          env.metrics.elapsed_s), dim=1)[ids].cpu().tolist()
    env_ids = ids.cpu().tolist()
    for module, count in writer.expert_counts.items():
        values = weights[module]
        if values.shape != (env.num_envs, count):
            raise ValueError('Expert gate shape differs from checkpoint')
        selected = values[ids]
        if (not bool(torch.isfinite(selected).all()) or bool((selected < 0).any()) or
                not bool(torch.allclose(selected.sum(dim=1), selected.new_ones(len(ids)),
                                        rtol=1e-5, atol=1e-6))):
            raise ValueError('Invalid expert probabilities')
        rows = []
        for env_id, (step, start), probabilities in zip(env_ids, clocks, selected.cpu().tolist()):
            row = dict(env.metrics.metadata[env_id])
            entry = float(row['entry_time_s'])
            row.update(control_step=int(step), time_start_s=start, time_end_s=start + env.dt,
                       control_dt_s=env.dt, trial_time_start_s=entry + start,
                       trial_time_end_s=entry + start + env.dt)
            row.update(zip(weight_columns(count), probabilities))
            rows.append(row)
        writer.write_expert_weights(module, rows)
