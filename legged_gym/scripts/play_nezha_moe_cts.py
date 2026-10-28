#!/usr/bin/env python
"""Evaluate the deployed CTS student in Isaac Gym, including one environment."""

import os

if __package__:
    from ._bootstrap import configure_runtime
else:
    from _bootstrap import configure_runtime

configure_runtime(isaac_gym=True)

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry, update_class_from_dict
from legged_gym.scripts.export_nezha_moe_cts import DEFAULT_EXPORT_PATH, resolve_checkpoint
from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint, policy_from_checkpoint, validate_checkpoint_actor
from rsl_rl.utils.cts_export import CTSStudentExporter


TASK_NAME = "nezha_moe_cts"
PLAY_ARGUMENTS = [
    {"name": "--checkpoint_path", "type": str, "help": "Explicit CTS model_<iteration>.pt path."},
    {"name": "--command_x", "type": float, "default": 1.0},
    {"name": "--command_y", "type": float, "default": 0.0},
    {"name": "--command_yaw", "type": float, "default": 0.0},
    {"name": "--play_steps", "type": int, "default": 5000},
    {"name": "--play_num_envs", "type": int, "default": 1},
    {"name": "--terrain", "type": str, "default": "trimesh"},
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
         expected_actor_type=None, expected_actor_expert_num=None):
    if args.play_num_envs < 1 or args.play_steps < 1:
        raise ValueError("play_num_envs and play_steps must be positive")
    if args.terrain not in ("plane", "heightfield", "trimesh"):
        raise ValueError("terrain must be plane, heightfield or trimesh")
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
    validate_checkpoint_actor(checkpoint, expected_actor_type, expected_actor_expert_num)
    if checkpoint.get("environment_config"):
        update_class_from_dict(env_cfg, checkpoint["environment_config"])
    env_cfg.env.observation_config = checkpoint["observation_config"]
    env_cfg.env.num_envs = args.play_num_envs
    env_cfg.env.episode_length_s = max(env_cfg.env.episode_length_s, 30)
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
    if not args.enable_domain_rand:
        for name in dir(env_cfg.domain_rand):
            if name.startswith("randomize_") or name in ("disturbance", "push_robots", "delay"):
                if isinstance(getattr(env_cfg.domain_rand, name), bool):
                    setattr(env_cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
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
    with torch.inference_mode():
        for step in range(args.play_steps):
            actions = policy.act_inference(observations)
            if args.gate_log_interval > 0 and step % args.gate_log_interval == 0:
                weights = policy.get_expert_probabilities(observations).mean(dim=0).cpu().tolist()
                print("step {} | {}".format(step, " ".join(
                    "expert_{}={:.3f}".format(i, weight) for i, weight in enumerate(weights)
                )))
            result = env.step(actions)
            observations, reward, done = result[0], result[2], result[3]
            observations = _fixed_commands(env, observations, command)
            rewards += reward.mean().item()
            resets += int(done.sum().item())
    print("Student evaluation: {} steps, mean reward/step {:.4f}, {} episode resets".format(
        args.play_steps, rewards / args.play_steps, resets
    ))


if __name__ == "__main__":
    play(get_args(extra_custom_parameters=PLAY_ARGUMENTS))
