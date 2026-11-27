#!/usr/bin/env python
"""Evaluate the deployed CTS student in Isaac Gym, including one environment."""

from itertools import count
import os
from pathlib import Path

if __package__:
    from ._bootstrap import configure_runtime
else:
    from _bootstrap import configure_runtime

configure_runtime(isaac_gym=True)

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts._joint_angles import open_joint_angles
from legged_gym.scripts._platform_play import configure_fixed_platform
from legged_gym.scripts._platform_trace import open_platform_trace
from legged_gym.scripts._play_env import StudentPlayEnv
from legged_gym.evaluation.timing import MAX_ATTEMPT_SECONDS
from legged_gym.scripts.export_nezha_moe_cts import DEFAULT_EXPORT_PATH, resolve_checkpoint
from rsl_rl.utils.cts_checkpoint import (
    load_cts_checkpoint, policy_from_checkpoint, validate_checkpoint_actor, validate_checkpoint_policy,
)
from rsl_rl.utils.cts_export import CTSStudentExporter


TASK_NAME = "nezha_cts_g1_student_moe"
PLAY_ARGUMENTS = [
    {"name": "--checkpoint_path", "type": str, "help": "Explicit CTS model_<iteration>.pt path."},
    {"name": "--skill-id", "type": int, "choices": [0, 1],
     "help": "Force the student skill (0 ordinary, 1 platform). By default the terrain supplies it."},
    {"name": "--command_x", "type": float, "default": 1.0},
    {"name": "--command_y", "type": float, "default": 0.0},
    {"name": "--command_yaw", "type": float, "default": 0.0},
    {"name": "--play_steps", "type": int,
     "help": "Total playback control steps. Defaults to continuous in the viewer, 5000 headless."},
    {"name": "--play_num_envs", "type": int, "default": 1},
    {"name": "--terrain", "type": str, "default": "trimesh"},
    {"name": "--platform_shape", "type": str, "choices": ["block", "pit"],
     "help": "Use a fixed WF platform for student play, together with --platform_height."},
    {"name": "--platform_height", "type": float,
     "help": "Block height or pit depth in metres, rounded to the terrain grid."},
    {"name": "--platform_width", "type": float,
     "help": "Optional full side of the square platform in metres, rounded to the WF horizontal grid."},
    {"name": "--platform_trace", "type": str,
     "help": "Write env 0 pose, joint targets and wheel contacts into a new CSV during fixed-platform play."},
    {"name": "--joint-angles-csv", "type": str,
     "help": "Write joint angles, leg targets, raw actor actions, velocity commands and measurements, wheel forces and expert weights."},
    {"name": "--joint_angles_csv", "type": str, "help": "Alias for --joint-angles-csv."},
    {"name": "--enable_domain_rand", "action": "store_true", "default": False},
    {"name": "--enable_noise", "action": "store_true", "default": False},
    {"name": "--no_export", "action": "store_true", "default": False},
    {"name": "--gate_log_interval", "type": int, "default": 100},
]


def _fixed_commands(env, observations, command):
    """Keep fixed commands consistent even when episode resets resample them."""
    value = torch.as_tensor(command, dtype=observations.dtype, device=env.device)
    env.commands[:, :3] = value
    offset = 0
    frame_offset = 0
    length = int(env.observation_config["history_length"])
    for term in env.observation_config["proprioceptive_terms"]:
        width = term["dim"]
        if term["name"] == "velocity_commands":
            scaled = value.clamp(*term["clip"]) * value.new_tensor(term["scale"])
            observations[:, offset:offset + width * length] = scaled.repeat(length)
            env.observation_history.buffer[:, :, frame_offset:frame_offset + width] = scaled
            env.current_obs[:, frame_offset:frame_offset + width] = scaled
        offset += width * length
        frame_offset += width
    return observations


def play(args, task_name=TASK_NAME, export_path=None,
         expected_actor_type=None, expected_actor_expert_num=None, expected_policy=None):
    if args.play_num_envs < 1 or (args.play_steps is not None and args.play_steps < 1):
        raise ValueError("play_num_envs and play_steps must be positive")
    if args.terrain not in ("plane", "heightfield", "trimesh"):
        raise ValueError("terrain must be plane, heightfield or trimesh")
    joint_csv = getattr(args, "joint_angles_csv", None)
    if joint_csv and args.platform_trace:
        if Path(joint_csv).expanduser().resolve() == Path(args.platform_trace).expanduser().resolve():
            raise ValueError("--joint-angles-csv and --platform_trace require separate output files")
    args.task = task_name
    # This entry point loads the student directly, without constructing a trainer.
    args.resume = False
    args.num_envs = args.play_num_envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    experiment = args.experiment_name or train_cfg.runner.experiment_name
    path = resolve_checkpoint(args.checkpoint_path, args.load_run, experiment)
    if not args.checkpoint_path and args.checkpoint not in (None, "-1"):
        filename = "best_policy.pt" if args.checkpoint == "best_policy" else "model_{}.pt".format(args.checkpoint)
        path = os.path.join(os.path.dirname(path), filename)
    checkpoint = load_cts_checkpoint(path, map_location="cpu")
    if expected_policy is None:
        from experiments.registry import get_experiment
        expected_policy = get_experiment(train_cfg.runner.experiment_key).POLICY
    validate_checkpoint_policy(checkpoint, expected_policy)
    validate_checkpoint_actor(checkpoint, expected_actor_type, expected_actor_expert_num)
    if checkpoint.get("environment_config"):
        task_registry.restore_environment_config(env_cfg, checkpoint["environment_config"])
        task_registry._restore_asset_path(env_cfg)
    env_cfg.env.observation_config = checkpoint["observation_config"]
    env_cfg.env.fixed_actor_skill_id = args.skill_id
    env_cfg.env.num_envs = args.play_num_envs
    env_cfg.env.episode_length_s = MAX_ATTEMPT_SECONDS
    env_cfg.sim.physx.max_gpu_contact_pairs = max(2 ** 10, args.play_num_envs * 256)
    env_cfg.noise.add_noise = args.enable_noise
    env_cfg.commands.heading_command = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s * 1000.0
    for name, value in (("lin_vel_x", args.command_x), ("lin_vel_y", args.command_y), ("ang_vel_yaw", args.command_yaw)):
        setattr(env_cfg.commands.ranges, name, [value, value])
    env_cfg.terrain.mesh_type = args.terrain
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = True
    height = configure_fixed_platform(env_cfg.terrain, args.platform_shape, args.platform_height,
                                      args.play_num_envs, args.platform_width)
    if height is not None:
        width = env_cfg.terrain.wf_sub_terrains["platform_" + args.platform_shape]["platform_width"]
        print("Fixed {} | height {:.3f} m | width {:.3f} m | initial yaw 0 | command ({}, {}, {})".format(
            args.platform_shape, height, width, args.command_x, args.command_y, args.command_yaw))
    if not args.enable_domain_rand:
        for name in dir(env_cfg.domain_rand):
            if name.startswith("randomize_") or name in ("disturbance", "push_robots", "delay"):
                if isinstance(getattr(env_cfg.domain_rand, name), bool):
                    setattr(env_cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg, env_class=StudentPlayEnv)
    env.reset()
    policy = policy_from_checkpoint(checkpoint, device=env.device).eval()
    if policy.num_actor_obs != env.num_obs:
        raise ValueError("Evaluation environment history does not match the checkpoint")
    if not args.no_export:
        export_path = export_path or DEFAULT_EXPORT_PATH
        CTSStudentExporter(policy, checkpoint["observation_config"], checkpoint.get("control_config")).export(export_path)
        print("Student exported to {}".format(export_path))
    command = (args.command_x, args.command_y, args.command_yaw)
    observations = _fixed_commands(env, env.get_observations(), command)
    rewards = 0.0
    resets = 0
    total_steps = args.play_steps if args.play_steps is not None else (5000 if args.headless else None)
    terrains = env.terrain_names() if joint_csv else ()
    with torch.inference_mode(), open_platform_trace(args.platform_trace, env, command,
                                                    args.platform_shape, height) as trace, \
            open_joint_angles(joint_csv, env, train_cfg.runner.experiment_key, terrains,
                              student_experts=policy.expert_num, actor_experts=policy.actor_expert_num) as joint_angles:
        env.joint_angles = joint_angles
        if joint_angles is not None:
            joint_angles.start()
        for step in (count() if total_steps is None else range(total_steps)):
            actions, student_weights, actor_weights, _, _ = policy.act_inference_with_diagnostics(observations)
            if joint_angles is not None:
                joint_angles.begin_step(student_weights, actor_weights, actions, observations)
            if args.gate_log_interval > 0 and step % args.gate_log_interval == 0:
                weights = student_weights.mean(dim=0).cpu().tolist()
                print("step {} | {}".format(step, " ".join(
                    "expert_{}={:.3f}".format(i, weight) for i, weight in enumerate(weights)
                )))
            result = env.step(actions)
            observations, reward, done = result[0], result[2], result[3]
            observations = _fixed_commands(env, observations, command)
            terminal_reasons = env.terminal_reasons
            for env_id, reason in terminal_reasons.items():
                print(f"step {step + 1} env {env_id} | {reason} | restarting", flush=True)
            if joint_angles is not None and terminal_reasons:
                joint_angles.start(list(terminal_reasons))
            if trace is not None:
                trace.record(step + 1, done, terminal_reasons.get(0, ""))
            rewards += reward.mean().item()
            resets += len(terminal_reasons)
    print("Student evaluation: {} steps, mean reward/step {:.4f}, {} episode resets".format(
        total_steps, rewards / total_steps, resets
    ))


if __name__ == "__main__":
    play(get_args(extra_custom_parameters=PLAY_ARGUMENTS))
