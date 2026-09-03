#!/usr/bin/env python
"""Dedicated Isaac Gym play/export entry point for the Nezha MINE policy.

The script carries the algorithm-specific behavior from LZHMine's play flow:
fixed commands, deterministic-domain evaluation, full 16-DOF logging, online
estimator/gate diagnostics and export of the integrated estimator + actor.
"""

import csv
import os
from datetime import datetime

import isaacgym  # noqa: F401 - must precede torch for Isaac Gym
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers nezha_mine
from legged_gym.utils import Logger, export_policy_as_jit, get_args, task_registry


TASK_NAME = "nezha_mine"
MODE_NAMES = ("wheel", "leg", "hybrid")
DOMAIN_RANDOMIZATION_FLAGS = (
    "randomize_payload_mass",
    "randomize_com_displacement",
    "randomize_link_mass",
    "randomize_friction",
    "randomize_restitution",
    "randomize_motor_strength",
    "randomize_kp",
    "randomize_kd",
    "randomize_initial_joint_pos",
    "disturbance",
    "push_robots",
    "delay",
)

PLAY_ARGUMENTS = [
    {"name": "--command_x", "type": float, "default": 1.0, "help": "Fixed forward velocity command [m/s]."},
    {"name": "--command_y", "type": float, "default": 0.0, "help": "Fixed lateral velocity command [m/s]."},
    {"name": "--command_yaw", "type": float, "default": 0.0, "help": "Fixed yaw-rate command [rad/s]."},
    {"name": "--play_steps", "type": int, "default": 5000, "help": "Number of policy steps to simulate."},
    {"name": "--play_num_envs", "type": int, "default": 1, "help": "Number of evaluation environments."},
    {"name": "--terrain", "type": str, "default": "trimesh", "help": "Evaluation terrain: plane, heightfield or trimesh."},
    {"name": "--enable_domain_rand", "action": "store_true", "default": False, "help": "Keep training domain randomization enabled."},
    {"name": "--no_noise", "action": "store_true", "default": False, "help": "Disable observation noise (LZHMine play keeps it enabled)."},
    {"name": "--no_export", "action": "store_true", "default": False, "help": "Do not export the integrated TorchScript policy."},
    {"name": "--checkpoint_history_order", "type": str, "default": "oldest_first", "help": "History layout used when this checkpoint was trained: oldest_first (Apr10-compatible) or newest_first (legacy LZHMine)."},
    {"name": "--gate_log_interval", "type": int, "default": 100, "help": "Print gate/estimator diagnostics every N steps; 0 disables it."},
    {"name": "--record_frames", "action": "store_true", "default": False, "help": "Write viewer frames under the experiment log."},
    {"name": "--move_camera", "action": "store_true", "default": False, "help": "Make the viewer camera follow Nezha."},
    {"name": "--plot_states", "action": "store_true", "default": False, "help": "Plot representative joint/base states after play."},
    {"name": "--log_joint", "type": int, "default": 1, "help": "DOF index used by the compact plot logger."},
    {"name": "--no_csv", "action": "store_true", "default": False, "help": "Disable the full Nezha play CSV."},
]


def _disable_domain_randomization(env_cfg):
    for name in DOMAIN_RANDOMIZATION_FLAGS:
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def _set_fixed_commands(env, command):
    """Set commands and keep every actor-history frame internally consistent."""
    command_tensor = torch.as_tensor(command, device=env.device, dtype=env.commands.dtype)
    env.commands[:, :3] = command_tensor
    scaled_command = command_tensor * env.commands_scale
    for frame_start in range(0, env.num_obs, env.num_one_step_obs):
        env.obs_buf[:, frame_start + 6 : frame_start + 9] = scaled_command
    return env.obs_buf


class NezhaPlayRecorder:
    """One-file replacement for LZHMine's many per-joint JSON/CSV lists."""

    def __init__(self, dt, dof_names, wheel_indices):
        self.dt = dt
        self.dof_names = list(dof_names)
        self.wheel_indices = set(int(index) for index in wheel_indices)
        self.rows = []

    def record(self, step, env, actions, gate_probs, estimate):
        robot = 0
        row = {
            "step": step,
            "time_s": step * self.dt,
            "command_x": env.commands[robot, 0].item(),
            "command_y": env.commands[robot, 1].item(),
            "command_yaw": env.commands[robot, 2].item(),
            "base_pos_x": env.root_states[robot, 0].item(),
            "base_pos_y": env.root_states[robot, 1].item(),
            "base_pos_z": env.root_states[robot, 2].item(),
            "base_vel_x": env.base_lin_vel[robot, 0].item(),
            "base_vel_y": env.base_lin_vel[robot, 1].item(),
            "base_vel_z": env.base_lin_vel[robot, 2].item(),
            "base_ang_vel_x": env.base_ang_vel[robot, 0].item(),
            "base_ang_vel_y": env.base_ang_vel[robot, 1].item(),
            "base_ang_vel_z": env.base_ang_vel[robot, 2].item(),
        }
        if gate_probs is not None:
            for mode_name, probability in zip(MODE_NAMES, gate_probs[robot]):
                row[f"gate_{mode_name}_probability"] = probability.item()
            row["selected_mode"] = MODE_NAMES[
                int(torch.argmax(gate_probs[robot]).item())
            ]
        if estimate is not None:
            row["estimated_base_vel_x"] = estimate[robot, 0].item()
            row["estimated_base_vel_y"] = estimate[robot, 1].item()
            row["estimated_base_vel_z"] = estimate[robot, 2].item()
            row["estimated_base_height"] = estimate[robot, 3].item()

        for dof_index, dof_name in enumerate(self.dof_names):
            prefix = dof_name[:-6] if dof_name.endswith("_joint") else dof_name
            row[f"{prefix}_pos"] = env.dof_pos[robot, dof_index].item()
            row[f"{prefix}_vel"] = env.dof_vel[robot, dof_index].item()
            row[f"{prefix}_torque"] = env.torques[robot, dof_index].item()
            row[f"{prefix}_action"] = actions[robot, dof_index].item()
            if dof_index in self.wheel_indices:
                target = actions[robot, dof_index] * env.cfg.control.vel_scale
            else:
                target = (
                    actions[robot, dof_index] * env.cfg.control.action_scale
                    + env.default_dof_pos[robot, dof_index]
                )
            row[f"{prefix}_target"] = target.item()
        self.rows.append(row)

    def save(self, path):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def _diagnostics(actor_critic, obs, env):
    with torch.inference_mode():
        estimate_scaled, _ = actor_critic.estimator.get_latent(obs)
        gate_probs = actor_critic.estimator.mode_probabilities(obs)
    estimate = estimate_scaled.clone()
    estimate[:, :3] /= env.obs_scales.lin_vel
    estimate[:, 3] /= env.obs_scales.height_measurements
    return gate_probs, estimate


def _log_compact_state(logger, env, actions, joint_index):
    robot = 0
    if joint_index in set(int(index) for index in env.wheel_indices.tolist()):
        position_target = None
        velocity_target = actions[robot, joint_index] * env.cfg.control.vel_scale
    else:
        position_target = (
            actions[robot, joint_index] * env.cfg.control.action_scale
            + env.default_dof_pos[robot, joint_index]
        )
        velocity_target = None
    states = {
        "dof_pos": env.dof_pos[robot, joint_index].item(),
        "dof_vel": env.dof_vel[robot, joint_index].item(),
        "dof_torque": env.torques[robot, joint_index].item(),
        "command_x": env.commands[robot, 0].item(),
        "command_y": env.commands[robot, 1].item(),
        "command_yaw": env.commands[robot, 2].item(),
        "base_vel_x": env.base_lin_vel[robot, 0].item(),
        "base_vel_y": env.base_lin_vel[robot, 1].item(),
        "base_vel_z": env.base_lin_vel[robot, 2].item(),
        "base_vel_yaw": env.base_ang_vel[robot, 2].item(),
        "contact_forces_z": env.contact_forces[robot, env.feet_indices, 2].cpu().numpy(),
    }
    if position_target is not None:
        states["dof_pos_target"] = position_target.item()
    if velocity_target is not None:
        states["dof_vel_target"] = velocity_target.item()
    logger.log_states(states)


def play(args):
    args.task = TASK_NAME
    args.num_envs = args.play_num_envs
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)

    if train_cfg.runner_class_name != "MINEOnPolicyRunner":
        raise RuntimeError("play_nezha_mine.py requires MINEOnPolicyRunner")
    if args.play_num_envs < 1 or args.play_steps < 1:
        raise ValueError("play_num_envs and play_steps must be positive")
    if args.terrain not in ("plane", "heightfield", "trimesh"):
        raise ValueError("terrain must be plane, heightfield or trimesh")
    if args.checkpoint_history_order not in ("oldest_first", "newest_first"):
        raise ValueError(
            "checkpoint_history_order must be oldest_first or newest_first"
        )

    env_cfg.env.num_envs = args.play_num_envs
    env_cfg.env.history_order = args.checkpoint_history_order
    train_cfg.policy.history_order = args.checkpoint_history_order
    env_cfg.env.episode_length_s = max(env_cfg.env.episode_length_s, 30)
    env_cfg.sim.max_gpu_contact_pairs = max(2**10, args.play_num_envs * 256)
    env_cfg.noise.add_noise = not args.no_noise
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s * 1000.0
    env_cfg.terrain.mesh_type = args.terrain
    env_cfg.terrain.curriculum = args.terrain != "plane"
    # Keep height sampling enabled on plane as well: the Nezha privileged
    # frame always owns its 187 terrain channels, and the LZHloco base class
    # initializes those sampling points even when the terrain is flat.
    env_cfg.terrain.measure_heights = True
    if not args.enable_domain_rand:
        _disable_domain_randomization(env_cfg)

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
    )
    actor_critic = runner.alg.actor_critic
    if actor_critic.__class__.__name__ != "MINEActorCritic":
        raise RuntimeError("Loaded policy is not MINEActorCritic")
    policy = runner.get_inference_policy(device=env.device)

    command = (args.command_x, args.command_y, args.command_yaw)
    obs = _set_fixed_commands(env, command)

    export_label = (
        "latest" if train_cfg.runner.load_run == -1 else os.path.basename(str(train_cfg.runner.load_run).rstrip("/"))
    )
    export_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        export_label,
    )
    if not args.no_export:
        export_policy_as_jit(actor_critic, export_dir)
        print(f"Integrated MINE policy exported to: {os.path.join(export_dir, 'policy.pt')}")

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    play_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "play",
        timestamp,
    )
    recorder = NezhaPlayRecorder(
        env.dt,
        env.dof_names,
        env.wheel_indices.detach().cpu().tolist(),
    )
    logger = Logger(env.dt)
    joint_index = min(max(args.log_joint, 0), env.num_actions - 1)
    frame_dir = os.path.join(play_dir, "frames")
    if args.record_frames:
        if env.viewer is None:
            raise RuntimeError("--record_frames requires a non-headless viewer")
        os.makedirs(frame_dir, exist_ok=True)

    print("=" * 72)
    print("Nezha gated-modal dual-encoder play")
    print(f"checkpoint run:     {export_label}")
    print(f"command [vx,vy,wz]:{command}")
    print(f"terrain/noise:      {args.terrain} / {env_cfg.noise.add_noise}")
    print(f"history order:      {args.checkpoint_history_order}")
    print(f"domain randomization: {args.enable_domain_rand}")
    print(f"steps/environments: {args.play_steps} / {env.num_envs}")
    print("=" * 72)

    for step in range(args.play_steps):
        obs = _set_fixed_commands(env, command)
        with torch.inference_mode():
            actions = policy(obs.detach())
            gate_probs, estimate = _diagnostics(actor_critic, obs, env)

        obs, _, rewards, dones, infos, _, _ = env.step(actions.detach())
        obs = _set_fixed_commands(env, command)

        if args.gate_log_interval > 0 and step % args.gate_log_interval == 0:
            mode_mean = gate_probs.mean(dim=0).detach().cpu().numpy()
            estimated = estimate[0].detach().cpu().numpy()
            true_velocity = env.base_lin_vel[0].detach().cpu().numpy()
            true_height = env._get_base_heights()[0].item()
            print(
                f"step {step:5d} | mode={MODE_NAMES[int(np.argmax(mode_mean))]} "
                f"gate[w/l/h]={np.array2string(mode_mean, precision=3)} "
                f"| v_est={np.array2string(estimated[:3], precision=3)} "
                f"v_true={np.array2string(true_velocity, precision=3)} "
                f"| h_est={estimated[3]:.3f} h_true={true_height:.3f}"
            )

        if not args.no_csv:
            recorder.record(step, env, actions, gate_probs, estimate)
        _log_compact_state(logger, env, actions, joint_index)

        episode_info = infos.get("episode")
        if episode_info:
            completed_episodes = int(dones.sum().item())
            if completed_episodes:
                logger.log_rewards(episode_info, completed_episodes)

        if args.record_frames and step % 2 == 0:
            filename = os.path.join(frame_dir, f"{step:06d}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
        if args.move_camera and env.viewer is not None:
            target = env.root_states[0, :3].detach().cpu().numpy()
            camera = target + np.asarray(env_cfg.viewer.pos, dtype=np.float64)
            env.set_camera(camera, target)

    if not args.no_csv:
        csv_path = os.path.join(play_dir, "nezha_mine_play.csv")
        recorder.save(csv_path)
        print(f"Full 16-DOF play log saved to: {csv_path}")
    if logger.num_episodes:
        logger.print_rewards()
    if args.plot_states:
        logger.plot_states()


if __name__ == "__main__":
    play(get_args(PLAY_ARGUMENTS))
