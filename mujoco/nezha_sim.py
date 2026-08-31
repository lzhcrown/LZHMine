"""MuJoCo sim-to-sim runner for an exported Nezha MINE policy.

The model must expose a free base and the 16 joints listed in the YAML file.
Torques are written through ``qfrc_applied``, so dedicated MJCF actuators are
not required.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deploy.nezha.policy_runtime import NezhaPolicyRuntime


def _resolve(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _joint_addresses(mujoco, model, joint_names):
    qpos_addresses = []
    dof_addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def _find_free_joint(mujoco, model):
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    raise ValueError(
        "Nezha sim-to-sim requires a floating-base MJCF/URDF. "
        "The supplied model has no free joint."
    )


def run(args):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("Install the optional MuJoCo dependency: pip install mujoco") from exc
    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "The official mujoco package is not installed; the local mujoco directory "
            "was imported as a namespace instead. Run: pip install mujoco"
        )

    cfg = _load_config(args.config)
    model_path = _resolve(args.model or cfg.get("model_path"))
    policy_path = _resolve(args.policy or cfg.get("policy_path"))
    if not model_path:
        raise ValueError("Set model_path in the config or pass --model <nezha.mjcf>")
    if not policy_path:
        raise ValueError("Set policy_path in the config or pass --policy <policy.pt>")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    free_joint_id = _find_free_joint(mujoco, model)
    root_qpos_address = int(model.jnt_qposadr[free_joint_id])
    root_body_id = int(model.jnt_bodyid[free_joint_id])
    qpos_addresses, dof_addresses = _joint_addresses(
        mujoco, model, cfg["joint_names"]
    )

    runtime = NezhaPolicyRuntime(
        policy_path=policy_path,
        default_dof_pos=cfg["default_dof_pos"],
        p_gains=cfg["p_gains"],
        d_gains=cfg["d_gains"],
        action_scale=cfg["action_scale"],
        wheel_velocity_scale=cfg["wheel_velocity_scale"],
        history_steps=int(cfg.get("history_steps", 2)),
        observation_scales=cfg.get("observation_scales"),
        torque_limits=cfg["torque_limits"],
        device=args.device,
    )
    data.qpos[root_qpos_address : root_qpos_address + 7] = [
        0.0,
        0.0,
        cfg.get("initial_base_height", 0.60),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qpos[qpos_addresses] = np.asarray(cfg["default_dof_pos"])
    mujoco.mj_forward(model, data)

    policy_dt = 1.0 / float(cfg.get("control_frequency_hz", 50))
    sim_steps_per_policy = max(1, int(round(policy_dt / model.opt.timestep)))
    commands = [args.vx, args.vy, args.yaw]
    viewer_context = (
        mujoco.viewer.launch_passive(model, data)
        if not args.headless
        else None
    )
    start_time = time.time()
    policy_step = 0
    gate_log_interval = (
        args.gate_log_interval
        if args.gate_log_interval is not None
        else int(cfg.get("gate_log_interval", 100))
    )
    mode_names = tuple(cfg.get("mode_names", ["wheel", "leg", "hybrid"]))
    if len(mode_names) != 3:
        raise ValueError("mode_names must contain wheel, leg and hybrid labels")
    try:
        while viewer_context is None or viewer_context.is_running():
            rotation = data.xmat[root_body_id].reshape(3, 3)
            projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
            angular_velocity = rotation.T @ data.cvel[root_body_id, :3]
            dof_pos = data.qpos[qpos_addresses].copy()
            dof_vel = data.qvel[dof_addresses].copy()
            _, torques = runtime.step(
                angular_velocity,
                projected_gravity,
                commands,
                dof_pos,
                dof_vel,
            )
            if gate_log_interval > 0 and policy_step % gate_log_interval == 0:
                mode_probs = runtime.mode_probabilities().cpu().numpy()
                selected_mode = mode_names[int(np.argmax(mode_probs))]
                print(
                    f"step {policy_step:6d} | mode={selected_mode} "
                    f"gate[w/l/h]={np.array2string(mode_probs, precision=3)}"
                )
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_addresses] = torques.cpu().numpy()
            step_start = time.time()
            for _ in range(sim_steps_per_policy):
                mujoco.mj_step(model, data)
            if viewer_context is not None:
                viewer_context.cam.lookat[:] = data.xpos[root_body_id]
                viewer_context.sync()
                remaining = policy_dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
            elif args.duration > 0 and time.time() - start_time >= args.duration:
                break
            policy_step += 1
    finally:
        if viewer_context is not None:
            viewer_context.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "mujoco", "nezha_config.yaml"),
    )
    parser.add_argument("--model", help="Floating-base Nezha MJCF or MuJoCo-compatible URDF")
    parser.add_argument("--policy", help="Exported policy.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--gate-log-interval",
        type=int,
        default=None,
        help="Print wheel/leg/hybrid gate probabilities every N policy steps; 0 disables it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
