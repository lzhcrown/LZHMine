"""Fixed wheel_gym_CQ patch protocol and CTS checkpoint selection."""
import argparse
import copy
import math
from pathlib import Path
import re
from types import SimpleNamespace

from experiments.registry import (EXPERIMENT_KEYS, LEGACY_EXPERIMENT_KEYS,
                                  RESIDUAL_EXPERIMENT_KEYS, ALL_MOE_EXPERIMENT_KEYS, REPO_ROOT)
from .presets import (PATCH_TERRAIN_TYPES, PATCH_SUPPORTED_TERRAIN_TYPES, PATCH_SIZE_M, PATCH_PLATFORM_LENGTH_M,
                      PATCH_CORRIDOR_WIDTH_M, PATCH_SLOPE_ANGLE_DEG, PATCH_SLOPE_LENGTH_M, PATCH_SLOPE_WIDTH_M)
from .timing import MAX_ATTEMPT_SECONDS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Capture CTS student evaluation formula inputs to CSV in Isaac Gym.')
    parser.add_argument('--group', choices=('all', 'residual') + EXPERIMENT_KEYS, default='all',
                        help='all preserves the original g0-g7 comparison; residual runs g8-g11.')
    parser.add_argument('--log-root', type=Path, default=REPO_ROOT / 'logs')
    parser.add_argument('--checkpoint-path', type=Path, help='Explicit checkpoint; requires one --group.')
    parser.add_argument('--checkpoint', type=int, default=20000, help='Iteration for every group; -1 selects latest numeric checkpoint.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--metric-successes-per-terrain', type=int, default=10,
                        help='Keep detailed metric inputs for the first N successes per terrain and model (default: 10).')
    parser.add_argument('--detail-selection', choices=('first_successes', 'trial_ids'), default='first_successes',
                        help='trial_ids records fixed episode indices in every environment, including failures.')
    parser.add_argument('--detail-trial-indices',
                        help='Comma-separated zero-based episode indices per environment; trial_ids defaults to 0.')
    parser.add_argument('--expert-weights-only', action='store_true',
                        help='All-MoE groups: capture Student/Actor gates instead of other metric streams.')
    parser.add_argument('--record-residual', action='store_true',
                        help='Also record pre-compensation means, corrections and final means for selected trials.')
    parser.add_argument('--residual-scale', type=float, default=1.,
                        help='Inference-only multiplier in [0,1]; 0 disables a trained residual for diagnosis.')
    parser.add_argument('--terrain-types', default=','.join(PATCH_TERRAIN_TYPES))
    parser.add_argument('--envs-per-terrain', type=int, help='Parallel slots per terrain. Default: 1 for preview, 20 for recording.')
    repetitions = parser.add_mutually_exclusive_group()
    repetitions.add_argument('--episodes-per-terrain', type=int,
        help='Total completed attempts per terrain and model; must divide evenly across slots. Default: 500 for recording.')
    repetitions.add_argument('--episodes-per-env', type=int,
        help='Attempts per slot. Interactive previews repeat until closed; headless previews default to one.')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--max-seconds', type=float, default=MAX_ATTEMPT_SECONDS,
                        help='Maximum simulation seconds per attempt, including the approach (up to 30).')
    parser.add_argument('--spawn-xy-jitter', type=float, default=0.)
    parser.add_argument('--max-tilt-deg', type=float, default=60.)
    parser.add_argument('--sim-device', default='cuda:0')
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--no-record', action='store_true', help='Preview only: no CSV, metadata or output directories.')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths and print protocol without simulation.')
    args = parser.parse_args(argv)
    if args.no_record and args.output_dir is not None:
        parser.error('--no-record cannot be combined with --output-dir')
    if args.metric_successes_per_terrain < 0:
        parser.error('metric-successes-per-terrain must be nonnegative')
    groups = selected_groups(args.group)
    if args.expert_weights_only and (not set(groups).issubset(ALL_MOE_EXPERIMENT_KEYS) or args.no_record):
        parser.error('--expert-weights-only requires all-MoE groups and recording enabled')
    if args.record_residual and args.no_record:
        parser.error('--record-residual requires recording enabled')
    if not math.isfinite(args.residual_scale) or not 0 <= args.residual_scale <= 1:
        parser.error('--residual-scale must be finite and in [0, 1]')
    if args.envs_per_terrain is None:
        args.envs_per_terrain = 1 if args.no_record else 20
    if min(args.envs_per_terrain, args.num_threads) < 1 or args.seed < 0:
        parser.error('counts must be positive and seed nonnegative')
    if args.episodes_per_terrain is None and args.episodes_per_env is None:
        if not args.no_record:
            args.episodes_per_terrain = 500
        elif not args.viewer:
            args.episodes_per_env = 1
    if args.episodes_per_terrain is not None:
        if args.episodes_per_terrain < 1:
            parser.error('episodes-per-terrain must be positive')
        if args.episodes_per_terrain % args.envs_per_terrain:
            parser.error('episodes-per-terrain must be divisible by envs-per-terrain; '
                         'for 500 attempts use e.g. 20 slots x 25 episodes or 10 slots x 50 episodes')
        args.episodes_per_env = args.episodes_per_terrain // args.envs_per_terrain
    elif args.episodes_per_env is not None:
        if args.episodes_per_env < 1:
            parser.error('episodes-per-env must be positive')
        args.episodes_per_terrain = args.envs_per_terrain * args.episodes_per_env
    if args.detail_selection == 'first_successes':
        if args.detail_trial_indices is not None:
            parser.error('--detail-trial-indices requires --detail-selection trial_ids')
        args.detail_trial_indices = ()
    else:
        try:
            indices = tuple(int(value) for value in (args.detail_trial_indices or '0').split(','))
        except ValueError:
            parser.error('--detail-trial-indices must be comma-separated integers')
        if (not indices or len(set(indices)) != len(indices) or
                any(index < 0 or (args.episodes_per_env is not None and index >= args.episodes_per_env)
                    for index in indices)):
            parser.error('--detail-trial-indices must be unique indices in [0, episodes-per-env)')
        args.detail_trial_indices = tuple(sorted(indices))
    args.terrain_types = tuple(args.terrain_types.split(','))
    if (len(set(args.terrain_types)) != len(args.terrain_types) or
            not set(args.terrain_types).issubset(PATCH_SUPPORTED_TERRAIN_TYPES)):
        parser.error('terrain-types must be unique names from ' + ','.join(PATCH_SUPPORTED_TERRAIN_TYPES))
    if not math.isfinite(args.speed) or args.speed == 0:
        parser.error('speed must be finite and nonzero')
    if not 0 < args.max_seconds <= MAX_ATTEMPT_SECONDS:
        parser.error('max-seconds must be in (0, 30]')
    if not math.isfinite(args.max_tilt_deg) or not 0 < args.max_tilt_deg < 90:
        parser.error('max-tilt-deg must be between 0 and 90')
    if not math.isfinite(args.spawn_xy_jitter) or not 0 <= args.spawn_xy_jitter <= .05:
        parser.error('spawn-xy-jitter must be between 0 and .05 metres')
    if not re.fullmatch(r'cpu|cuda:\d+', args.sim_device):
        parser.error('sim-device must be cpu or cuda:N')
    if args.checkpoint < -1:
        parser.error('checkpoint must be -1 or a nonnegative iteration')
    if args.checkpoint_path and len(groups) != 1:
        parser.error('checkpoint-path requires a single group')
    return args


def selected_groups(group):
    if group == 'all':
        return LEGACY_EXPERIMENT_KEYS
    if group == 'residual':
        return RESIDUAL_EXPERIMENT_KEYS
    return (group,)


def resolve_checkpoint(args, group):
    if args.checkpoint_path:
        path = args.checkpoint_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path
    # Uploaded flat directories and the repository's training-run layout.
    candidates = []
    for folder in (args.log_root / group, args.log_root / 'nezha_moe_cts' / group):
        for path in folder.rglob('model_*.pt'):
            match = re.fullmatch(r'model_(\d+)\.pt', path.name)
            if match and path.is_file() and (args.checkpoint == -1 or int(match[1]) == args.checkpoint):
                candidates.append((int(match[1]), path.resolve()))
    if not candidates:
        raise FileNotFoundError('No model_{}.pt for {} under {}'.format(args.checkpoint, group, args.log_root))
    iteration = max(item[0] for item in candidates)
    paths = sorted(set(path for number, path in candidates if number == iteration))
    if len(paths) != 1:
        raise ValueError('Ambiguous runs for {}; select --checkpoint-path: {}'.format(group, paths))
    return paths[0]


def protocol_dict(args):
    return dict(
        version=14, reference='LZHMine obstacle traversal and wheel_gym_CQ patch crossing; terrain revision 12',
        data_mode=('preview_no_record' if args.no_record else
                   'expert_weights_only' if args.expert_weights_only else 'first_successes_split'),
        expert_weights_only=args.expert_weights_only,
        record_residual=args.record_residual, residual_scale=args.residual_scale,
        detail_selection=args.detail_selection, detail_trial_indices=list(args.detail_trial_indices),
        expert_sampling='action-input gates at each measured control interval start; Student and Actor only',
        metric_successes_per_terrain=args.metric_successes_per_terrain,
        metric_selection=('first successes in completion order per model and terrain; env_id breaks same-step ties'
                          if args.detail_selection == 'first_successes' else
                          'preselected episode_index in every environment; all outcomes included'),
        write_files=not args.no_record,
        terrain_types=list(args.terrain_types), patch_size_m=PATCH_SIZE_M,
        terrain_roi_size_xy_m={name: ([PATCH_SLOPE_LENGTH_M, PATCH_SLOPE_WIDTH_M]
                                     if name == 'slope' else [PATCH_SIZE_M, PATCH_SIZE_M])
                               for name in args.terrain_types},
        envs_per_terrain=args.envs_per_terrain, episodes_per_env=args.episodes_per_env,
        episodes_per_terrain=args.episodes_per_terrain,
        total_attempts=(None if args.episodes_per_terrain is None else len(args.terrain_types) * args.episodes_per_terrain),
        seed=args.seed, command_body_vx_vy_wz=[args.speed, 0., 0.], max_seconds=args.max_seconds,
        max_tilt_deg=args.max_tilt_deg, spawn_xy_jitter_m=args.spawn_xy_jitter,
        measurement_window='first wheel entry until successful completion, fall or timeout; entry-detecting interval excluded',
        success_rule='stairs and platforms require supported traversal of the central surface and a stable landing on the far side; other terrain uses four-wheel opposite or lateral exit; fall takes priority',
        failure_rule='base tilt exceeds max_tilt_deg; contact/retreat alone are not failures',
        event_resolution='one control interval', randomization='disabled, including torque scale',
        sampling='every control interval and physics substep in the measurement window; no warmup filtering',
        cot_inputs='whole robot mass, gravity, torque, before/after angular velocity, dt, actual displacement',
        power_scope='joint mechanical power; electrical consumption is not measured',
        outputs=('console only; no capture files' if args.no_record else
                 'all trial outcomes and passage times; selected trials with per-terrain Student/Actor gate CSVs'
                 if args.expert_weights_only else
                 'all trial outcomes and success rates; selected trials in power/cot/tracking/stability CSVs'),
    )


def configure_evaluation(env_cfg, args, group, checkpoint):
    cfg = copy.deepcopy(env_cfg)
    cfg.seed = args.seed
    cfg.env.num_envs = len(args.terrain_types) * args.envs_per_terrain
    cfg.env.episode_length_s = args.max_seconds
    cfg.noise.add_noise = False
    cfg.init_state.rand_init_dof = False
    for name in dir(cfg.domain_rand):
        if not name.startswith('_') and isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.randomize_torque_scale = False
    cfg.commands.curriculum = cfg.commands.heading_command = False
    cfg.commands.curriculum_profile = 'evaluation'
    cfg.commands.resampling_time = args.max_seconds + 1
    for name, value in (('lin_vel_x', args.speed), ('lin_vel_y', 0.), ('ang_vel_yaw', 0.)):
        setattr(cfg.commands.ranges, name, [value, value])
    terrain = dict(mesh_type='trimesh', generator_profile='evaluation', layout='patch', curriculum=False,
                   teleport_robots=False, measure_heights=True, type_names=list(args.terrain_types), seed=args.seed,
                   terrain_length=PATCH_SIZE_M, terrain_width=PATCH_SIZE_M, num_rows=1, num_cols=len(args.terrain_types),
                   max_init_terrain_level=0, horizontal_scale=.02, vertical_scale=.005, border_size=2., gap_size=2.,
                   platform_length=PATCH_PLATFORM_LENGTH_M, corridor_width=PATCH_CORRIDOR_WIDTH_M,
                   spawn_xy_jitter=args.spawn_xy_jitter, slope_treshold=.75, slope_angle_deg=PATCH_SLOPE_ANGLE_DEG,
                   slope_length_m=PATCH_SLOPE_LENGTH_M, slope_width_m=PATCH_SLOPE_WIDTH_M,
                   rough_height=.04, rough_cell_size=.08, step_width=.3)
    for name, value in terrain.items():
        setattr(cfg.terrain, name, value)
    cfg.sim.use_gpu_pipeline = args.sim_device.startswith('cuda')
    cfg.sim.physx.use_gpu = cfg.sim.use_gpu_pipeline
    cfg.sim.physx.max_gpu_contact_pairs = max(2 ** 18, cfg.env.num_envs * 4096)
    cfg.evaluation = SimpleNamespace(episodes_per_env=args.episodes_per_env, seed=args.seed,
        task=group, checkpoint=str(checkpoint), command_vx=args.speed, max_tilt_deg=args.max_tilt_deg,
        trajectory_stride=1, wheel_side='all', spawn_xy_jitter=args.spawn_xy_jitter,
        contact_force_threshold=1., no_record=args.no_record)
    return cfg
