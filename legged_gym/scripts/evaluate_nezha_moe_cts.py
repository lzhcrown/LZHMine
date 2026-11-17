#!/usr/bin/env python
"""Record legacy CTS students or the G7 residual factorial with one protocol."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

if __package__:
    from ._bootstrap import configure_runtime
else:
    from _bootstrap import configure_runtime
configure_runtime(isaac_gym=True)

from experiments.registry import REPO_ROOT, get_experiment
from legged_gym.evaluation.config import (parse_args, resolve_checkpoint, protocol_dict,
                                         configure_evaluation, selected_groups)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    def serializable(item):
        if isinstance(item, dict):
            return {key: serializable(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [serializable(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        return item
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate_one(args, group, path, output):
    import isaacgym  # Must precede torch.
    from isaacgym import gymapi
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils.helpers import class_to_dict, update_class_from_dict, set_seed, parse_sim_params
    from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint, validate_checkpoint_policy, policy_from_checkpoint
    from legged_gym.evaluation.patch_env import PatchEvaluationEnv
    from legged_gym.evaluation.selective_writer import SelectiveCsvEvaluationWriter
    from legged_gym.evaluation.experts import (apply_residual_scale, record_expert_step,
                                              record_residual_step)

    experiment = get_experiment(group)
    checkpoint = load_cts_checkpoint(str(path), map_location='cpu')
    validate_checkpoint_policy(checkpoint, experiment.POLICY)
    if not checkpoint.get('environment_config'):
        raise ValueError('Evaluation requires a checkpoint containing environment_config')
    cfg, _ = task_registry.get_cfgs(experiment.TASK_NAME)
    update_class_from_dict(cfg, checkpoint['environment_config'])
    cfg.env.observation_config = checkpoint['observation_config']
    # Snapshots from another computer may contain its absolute repository path.
    saved_asset = cfg.asset.file
    marker = '/resources/'
    if marker in saved_asset:
        cfg.asset.file = str(REPO_ROOT / 'resources' / saved_asset.split(marker, 1)[1])
    if not Path(cfg.asset.file).is_file():
        raise FileNotFoundError('Robot asset: ' + cfg.asset.file)
    cfg = configure_evaluation(cfg, args, group, path)
    if args.sim_device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; evaluation has not started')
    set_seed(args.seed)
    policy = policy_from_checkpoint(checkpoint, device=args.sim_device).eval()
    apply_residual_scale(policy, args.residual_scale)
    expert_counts = ({'student': policy.expert_num, 'actor': policy.actor_expert_num}
                     if args.expert_weights_only else {})
    sim_args = SimpleNamespace(physics_engine=gymapi.SIM_PHYSX, device=args.sim_device,
        use_gpu=args.sim_device.startswith('cuda'), subscenes=0,
        use_gpu_pipeline=args.sim_device.startswith('cuda'), num_threads=args.num_threads)
    sim_params = parse_sim_params(sim_args, {'sim': class_to_dict(cfg.sim)})
    if args.no_record:
        from legged_gym.evaluation.preview import run_preview
        return run_preview(args, cfg, sim_params, gymapi.SIM_PHYSX, policy, PatchEvaluationEnv)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='initializing', group=group, started_utc=datetime.now(timezone.utc).isoformat(),
        run_id=uuid4().hex, trial_annotation_schema=1,
        expert_counts=expert_counts,
        residual_enabled=policy.residual_enabled,
        effective_residual_action_scales=(policy.actor.residual_action_scales.cpu().tolist()
                                         if policy.residual_enabled else None),
        checkpoint=str(path), checkpoint_sha256=sha256(path), protocol=protocol_dict(args),
        environment_config=class_to_dict(cfg), policy_config=checkpoint['policy_config'],
        original_asset_path=saved_asset, robot_asset=cfg.asset.file, robot_asset_sha256=sha256(cfg.asset.file),
        physics_dt_s=sim_params.dt, sim_device=args.sim_device, torch_version=str(torch.__version__),
        source_sha256={str(p.relative_to(REPO_ROOT)): sha256(p) for p in
            list((REPO_ROOT / 'legged_gym/evaluation').glob('*.py')) + [Path(__file__).resolve(),
            REPO_ROOT / 'legged_gym/envs/nezha/nezha_moe_cts_env.py', REPO_ROOT / 'legged_gym/envs/nezha/nezha_robot.py',
            REPO_ROOT / 'rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py',
            REPO_ROOT / 'rsl_rl/rsl_rl/modules/moe.py']})
    write_json(output / 'metadata.json', metadata)
    env = None
    status = 'error'
    with SelectiveCsvEvaluationWriter(output, num_joints=cfg.env.num_actions, run_id=metadata['run_id'],
                                      successes_per_terrain=args.metric_successes_per_terrain,
                                      expert_counts=expert_counts, terrains=args.terrain_types,
                                      record_residual=args.record_residual,
                                      detail_selection=args.detail_selection,
                                      detail_trial_indices=args.detail_trial_indices) as writer:
        try:
            env = PatchEvaluationEnv(cfg, sim_params, gymapi.SIM_PHYSX, args.sim_device, not args.viewer,
                episode_sink=writer.write_episodes,
                trajectory_sink=None if args.expert_weights_only else writer.write_trajectories,
                physics_sink=None if args.expert_weights_only else writer.write_physics)
            if policy.num_actor_obs != env.num_obs:
                raise ValueError('Checkpoint observation size differs from environment')
            metadata.update(status='running', joint_names=env.measurement_joint_names,
                total_mass_kg=env.total_mass_kg.cpu().tolist(), terrain=env.terrain.metadata,
                control_dt_s=env.dt, wheel_geometry=env.wheel_geometry,
                units=dict(position='m', linear_velocity='m/s', angular_velocity='rad/s', torque='N*m', mass='kg'),
                residual_diagnostics=dict(action_space='normalized policy mean before environment action clipping',
                    wheel_indices=env.wheel_indices.cpu().tolist(), saturation_ratio=.95,
                    saturation_action_thresholds=((.95 * policy.actor.residual_action_scales).cpu().tolist()
                                                  if policy.residual_enabled else None),
                    clip_actions=cfg.normalization.clip_actions),
                joint_column_mapping='j00..j15 follow joint_names; torque and velocity use the same simulator DOF order')
            write_json(output / 'metadata.json', metadata)
            env.reset()
            if args.viewer:
                center = env.spawn_positions[0].cpu().tolist()
                env.set_camera([center[0]-3, center[1]-3, center[2]+2], center)
            print('Evaluating {}: {} -> {}'.format(group, path, output), flush=True)
            last_report = time.monotonic()
            with torch.inference_mode():
                while not env.evaluation_complete:
                    if args.expert_weights_only or args.record_residual:
                        actions, student_gate, actor_gate, pre, correction = policy.act_inference_with_diagnostics(
                            env.get_observations())
                        if args.expert_weights_only:
                            record_expert_step(writer, env, {'student': student_gate, 'actor': actor_gate})
                        if args.record_residual:
                            record_residual_step(writer, env, pre, correction, actions,
                                policy.actor.residual_action_scales if policy.residual_enabled else None)
                    else:
                        actions = policy.act_inference(env.get_observations())
                    env.step(actions)
                    if time.monotonic() - last_report >= 10:
                        print('{}: {}/{} attempts'.format(group, len(env.completed_episodes),
                              metadata['protocol']['total_attempts']), flush=True)
                        last_report = time.monotonic()
            status = 'complete'
        except (KeyboardInterrupt, SystemExit):
            status = 'interrupted'
        except Exception as exc:
            metadata['error'] = repr(exc)
            raise
        finally:
            try:
                if env is not None and status != 'complete':
                    env.interrupt(reason=status)
                rows = env.completed_episodes if env is not None else []
                writer.write_summary(rows)
                metadata.update(status=status, recorded_attempts=len(rows), finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(output / 'metadata.json', metadata)
            finally:
                if env is not None:
                    if env.viewer is not None:
                        env.gym.destroy_viewer(env.viewer)
                    env.gym.destroy_sim(env.sim)
    print('{}: {}'.format(group, status), flush=True)
    if status == 'complete':
        from legged_gym.scripts.validate_evaluation_capture import validate
        report = validate(output)
        write_json(output / 'data_validation.json', report)
    return 0 if status == 'complete' else 130


def main(argv=None):
    args = parse_args(argv)
    groups = selected_groups(args.group)
    paths = {group: resolve_checkpoint(args, group) for group in groups}
    if args.dry_run:
        print(json.dumps(dict(protocol=protocol_dict(args), checkpoints={k: str(v) for k, v in paths.items()}), indent=2))
        return 0
    if args.no_record:
        if len(groups) == 1:
            return evaluate_one(args, args.group, paths[args.group], None)
        for group in groups:
            result = subprocess.run(child_command(args, group, paths[group]))
            if result.returncode:
                return result.returncode
        return 0
    output = (args.output_dir or REPO_ROOT / 'evaluations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory: ' + str(output))
    if len(groups) == 1:
        return evaluate_one(args, args.group, paths[args.group], output)
    output.mkdir(parents=True)
    manifest = dict(status='running', protocol=protocol_dict(args), groups={})
    write_json(output / 'batch.json', manifest)
    try:
        for group in groups:
            # Separate processes prevent Gym/CUDA state from leaking between groups.
            command = child_command(args, group, paths[group], output / group)
            result = subprocess.run(command)
            manifest['groups'][group] = dict(checkpoint=str(paths[group]), returncode=result.returncode)
            write_json(output / 'batch.json', manifest)
            if result.returncode:
                raise RuntimeError('Evaluation stopped for {}; completed captures retained at {}'.format(group, output))
        rows = []
        for group in groups:
            with (output / group / 'summary.csv').open() as handle:
                rows.extend(dict(group=group, **row) for row in csv.DictReader(handle))
        with (output / 'event_counts.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest['status'] = 'complete'
    finally:
        if manifest['status'] != 'complete':
            manifest['status'] = 'incomplete'
        write_json(output / 'batch.json', manifest)
    print('All groups captured: ' + str(output), flush=True)
    return 0


def child_command(args, group, checkpoint, output=None):
    command = [sys.executable, str(Path(__file__).resolve()), '--group', group,
        '--checkpoint-path', str(checkpoint),
        '--terrain-types', ','.join(args.terrain_types), '--envs-per-terrain', str(args.envs_per_terrain),
        '--episodes-per-terrain', str(args.episodes_per_terrain),
        '--metric-successes-per-terrain', str(args.metric_successes_per_terrain),
        '--detail-selection', args.detail_selection,
        '--residual-scale', str(args.residual_scale),
        '--seed', str(args.seed), '--speed', str(args.speed),
        '--max-seconds', str(args.max_seconds), '--max-tilt-deg', str(args.max_tilt_deg),
        '--spawn-xy-jitter', str(args.spawn_xy_jitter), '--sim-device', args.sim_device,
        '--num-threads', str(args.num_threads)]
    if args.no_record:
        command.append('--no-record')
    else:
        command.extend(['--output-dir', str(output)])
    if args.viewer:
        command.append('--viewer')
    if args.expert_weights_only:
        command.append('--expert-weights-only')
    if args.record_residual:
        command.append('--record-residual')
    if args.detail_trial_indices:
        command.extend(['--detail-trial-indices', ','.join(str(i) for i in args.detail_trial_indices)])
    return command


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print('Evaluation error: ' + str(exc), file=sys.stderr)
        sys.exit(1)
