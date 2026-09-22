"""MuJoCo deployment runner for an exported Nezha MINE policy.

The model must expose a free base and the 16 joints listed in the YAML file.
Torques are written through ``qfrc_applied``, so dedicated MJCF actuators are
not required.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np
import torch
import yaml


DEPLOY_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(DEPLOY_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deploy.deploy_mujoco.runtime import NezhaPolicyRuntime


# Keep interactive commands inside the ranges used during training.
COMMAND_MIN = np.asarray([-1.0, -0.6, -1.0], dtype=np.float32)
COMMAND_MAX = np.asarray([1.0, 0.6, 1.0], dtype=np.float32)


class KeyboardCommandController:
    """Thread-safe velocity commands updated by MuJoCo's viewer callback."""

    def __init__(self, initial, linear_step=0.1, yaw_step=0.1):
        self._lock = threading.Lock()
        self._commands = np.clip(
            np.asarray(initial, dtype=np.float32), COMMAND_MIN, COMMAND_MAX
        )
        self._linear_step = float(linear_step)
        self._yaw_step = float(yaw_step)
        self._reset_requested = False

    def get(self):
        with self._lock:
            return self._commands.copy()

    def stop(self):
        with self._lock:
            self._commands.fill(0.0)

    def consume_reset_request(self):
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    def on_key(self, keycode):
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        with self._lock:
            if key == "W":
                self._commands[0] += self._linear_step
            elif key == "S":
                self._commands[0] -= self._linear_step
            elif key == "A":
                self._commands[1] += self._linear_step
            elif key == "D":
                self._commands[1] -= self._linear_step
            elif key == "Q":
                self._commands[2] += self._yaw_step
            elif key == "E":
                self._commands[2] -= self._yaw_step
            elif keycode == 32:  # Space
                self._commands.fill(0.0)
            elif key == "R":
                self._commands.fill(0.0)
                self._reset_requested = True
            else:
                return
            np.clip(self._commands, COMMAND_MIN, COMMAND_MAX, out=self._commands)
            commands = self._commands.copy()
        print(
            f"command [vx, vy, yaw] = "
            f"[{commands[0]:+.2f}, {commands[1]:+.2f}, {commands[2]:+.2f}]"
        )


def _resolve(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.normpath(
        os.path.join(DEPLOY_DIR, path)
    )


def _resolve_config(path):
    if os.path.isabs(path):
        return path
    if os.path.dirname(path):
        return os.path.normpath(os.path.join(DEPLOY_DIR, path))
    config_name = path if path.endswith(".yaml") else f"{path}.yaml"
    return os.path.join(DEPLOY_DIR, "configs", config_name)


def _resolve_policy(path):
    resolved = _resolve(path)
    if resolved and os.path.isfile(resolved):
        return resolved
    # A frequent command-line typo is to omit the final ``t`` in policy.pt.
    # Accept it when the intended file is unambiguous and report the repair.
    if resolved and resolved.endswith(".p"):
        corrected = resolved + "t"
        if os.path.isfile(corrected):
            print(f"policy path ended in '.p'; using: {corrected}")
            return corrected
    # Also allow callers to pass an export directory instead of the file.
    if resolved and os.path.isdir(resolved):
        candidate = os.path.join(resolved, "policy.pt")
        if os.path.isfile(candidate):
            print(f"policy export directory supplied; using: {candidate}")
            return candidate
    if resolved and os.path.basename(os.path.dirname(resolved)) == "latest":
        exported_root = os.path.dirname(os.path.dirname(resolved))
        candidates = [
            candidate
            for candidate in glob.glob(
                os.path.join(exported_root, "*", os.path.basename(resolved))
            )
            if os.path.isfile(candidate)
        ]
        if candidates:
            selected = max(candidates, key=os.path.getmtime)
            print(f"latest export alias is absent; using: {selected}")
            return selected
    return resolved


def _require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{description} does not exist: {path}\n"
            "Paths may be absolute or relative to deploy/deploy_mujoco."
        )


def _load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _terrain_course_spawn(mujoco, model, cfg, initial_base_height):
    """Validate the fixed terrain course and return its flat staging spawn."""
    if model.nhfield == 0:
        return np.asarray([0.0, 0.0, initial_base_height]), None
    metadata_path = _resolve(cfg.get("terrain_metadata_path"))
    if not metadata_path or not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            "Terrain-course metadata is absent. Generate it with: "
            "python deploy/deploy_mujoco/tools/generate_terrain_course.py"
        )
    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    binary_shape = tuple(int(value) for value in metadata["binary_shape"])
    model_shape = (
        int(model.hfield_nrow[0]),
        int(model.hfield_ncol[0]),
    )
    if model_shape != binary_shape:
        raise RuntimeError(
            f"MuJoCo heightfield shape {model_shape} does not match "
            f"training metadata {binary_shape}"
        )

    hfield_address = int(model.hfield_adr[0])
    hfield_count = model_shape[0] * model_shape[1]
    loaded_hfield = np.asarray(
        model.hfield_data[hfield_address : hfield_address + hfield_count],
        dtype=np.float32,
    )
    loaded_digest = hashlib.sha256(loaded_hfield.tobytes()).hexdigest()
    if loaded_digest != metadata["binary_data_sha256"]:
        raise RuntimeError(
            "Loaded MuJoCo height samples differ from the exported training "
            "terrain course; rerun python "
            "deploy/deploy_mujoco/tools/generate_terrain_course.py"
        )

    expected_size = np.asarray(metadata["hfield_size"], dtype=np.float64)
    if not np.allclose(model.hfield_size[0], expected_size, atol=1e-9):
        raise RuntimeError(
            f"MuJoCo hfield size {model.hfield_size[0]} does not match "
            f"training metadata {expected_size}"
        )
    terrain_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "training_terrain"
    )
    if terrain_geom_id < 0:
        raise RuntimeError("MuJoCo scene is missing geom 'training_terrain'")
    expected_position = np.asarray(metadata["hfield_position"], dtype=np.float64)
    if not np.allclose(model.geom_pos[terrain_geom_id], expected_position, atol=1e-9):
        raise RuntimeError(
            f"MuJoCo terrain position {model.geom_pos[terrain_geom_id]} does "
            f"not match training metadata {expected_position}"
        )
    ground_position = np.asarray(
        metadata["spawn_ground_position"], dtype=np.float64
    )
    position = ground_position + np.asarray([0.0, 0.0, initial_base_height])
    description = {
        "profile": metadata.get("profile", "terrain_course"),
        "spawn": position,
        "lanes": metadata["lanes"],
        "sha256": metadata["height_field_sha256"],
    }
    return position, description


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


def _sensor_slice(mujoco, model, name, expected_dim):
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, name
    )
    if sensor_id < 0:
        raise ValueError(f"MuJoCo model is missing sensor: {name}")
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dim:
        raise ValueError(
            f"Sensor {name} has dimension {dimension}, expected {expected_dim}"
        )
    address = int(model.sensor_adr[sensor_id])
    return slice(address, address + dimension)


def _reset_robot(
    mujoco,
    model,
    data,
    runtime,
    root_qpos_address,
    qpos_addresses,
    default_dof_pos,
    initial_base_position,
):
    """Restore the nominal standing pose and clear policy/control history."""
    mujoco.mj_resetData(model, data)
    data.qpos[root_qpos_address : root_qpos_address + 7] = [
        initial_base_position[0],
        initial_base_position[1],
        initial_base_position[2],
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qpos[qpos_addresses] = np.asarray(default_dof_pos)
    data.qvel[:] = 0.0
    data.qfrc_applied[:] = 0.0
    runtime.reset()
    mujoco.mj_forward(model, data)


def _settle_robot(
    mujoco,
    model,
    data,
    runtime,
    qpos_addresses,
    dof_addresses,
    settle_time_s,
):
    """Settle the nominal pose under zero-action PD before opening the viewer."""
    zero_actions = torch.zeros(16, dtype=torch.float32, device=runtime.device)
    settle_steps = max(0, int(round(settle_time_s / model.opt.timestep)))
    for _ in range(settle_steps):
        torques = runtime.compute_torques(
            zero_actions,
            data.qpos[qpos_addresses],
            data.qvel[dof_addresses],
        )
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof_addresses] = torques.cpu().numpy()
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    data.qfrc_applied[:] = 0.0
    runtime.reset()
    mujoco.mj_forward(model, data)


def run(args):
    if args.headless and args.duration <= 0:
        raise ValueError(
            "Headless simulation requires --duration to be greater than zero."
        )
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

    config_path = _resolve_config(args.config)
    _require_file(config_path, "Deployment config")
    cfg = _load_config(config_path)
    model_path = _resolve(args.model or cfg.get("model_path"))
    policy_path = _resolve_policy(args.policy or cfg.get("policy_path"))
    if not model_path:
        raise ValueError("Set model_path in the config or pass --model <nezha.mjcf>")
    if not policy_path:
        raise ValueError("Set policy_path in the config or pass --policy <policy.pt>")
    _require_file(model_path, "MuJoCo model")
    _require_file(policy_path, "Exported policy")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    free_joint_id = _find_free_joint(mujoco, model)
    root_qpos_address = int(model.jnt_qposadr[free_joint_id])
    root_body_id = int(model.jnt_bodyid[free_joint_id])
    qpos_addresses, dof_addresses = _joint_addresses(
        mujoco, model, cfg["joint_names"]
    )
    gyro_slice = _sensor_slice(mujoco, model, "base_gyro", 3)

    runtime = NezhaPolicyRuntime(
        policy_path=policy_path,
        default_dof_pos=cfg["default_dof_pos"],
        p_gains=cfg["p_gains"],
        d_gains=cfg["d_gains"],
        action_scale=cfg["action_scale"],
        wheel_velocity_scale=cfg["wheel_velocity_scale"],
        history_steps=int(cfg.get("history_steps", 2)),
        history_order=args.history_order or cfg.get("history_order", "auto"),
        legacy_history_order=cfg.get(
            "legacy_history_order", "newest_first"
        ),
        zero_wheel_velocity_observations=bool(
            cfg.get("zero_wheel_velocity_observations", True)
        ),
        observation_scales=cfg.get("observation_scales"),
        torque_limits=cfg["torque_limits"],
        device=args.device,
    )
    initial_base_height = float(cfg.get("initial_base_height", 0.60))
    initial_base_position, terrain_description = _terrain_course_spawn(
        mujoco, model, cfg, initial_base_height
    )
    _reset_robot(
        mujoco,
        model,
        data,
        runtime,
        root_qpos_address,
        qpos_addresses,
        cfg["default_dof_pos"],
        initial_base_position,
    )
    settle_time_s = float(cfg.get("settle_time_s", 2.0))
    _settle_robot(
        mujoco,
        model,
        data,
        runtime,
        qpos_addresses,
        dof_addresses,
        settle_time_s,
    )

    policy_dt = 1.0 / float(cfg.get("control_frequency_hz", 50))
    sim_steps_per_policy = max(1, int(round(policy_dt / model.opt.timestep)))
    command_controller = KeyboardCommandController(
        initial=[args.vx, args.vy, args.yaw],
        linear_step=args.linear_step,
        yaw_step=args.yaw_step,
    )
    viewer_context = (
        mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=command_controller.on_key,
        )
        if not args.headless
        else None
    )
    if terrain_description is not None:
        description = terrain_description
        print(
            "Terrain gallery loaded: robot starts on flat staging area at "
            f"{np.array2string(description['spawn'], precision=3)}; "
            f"profile={description['profile']}; "
            f"raw_sha256={description['sha256'][:12]}..."
        )
        print("Lanes across y (use A/D on the staging area to align):")
        for lane in description["lanes"]:
            print(f"  y={lane['y_center_m']:+5.1f} m  {lane['name']}")
    if viewer_context is not None:
        print(
            "Nezha was pre-settled and starts in frozen standby. "
            "Send a nonzero command to release it."
        )
        print("Keyboard: W/S=vx  A/D=vy  Q/E=yaw  Space=stop  R=reset")
    simulated_time = 0.0
    policy_step = 0
    gate_log_interval = (
        args.gate_log_interval
        if args.gate_log_interval is not None
        else int(cfg.get("gate_log_interval", 100))
    )
    mode_names = tuple(cfg.get("mode_names", ["wheel", "leg", "hybrid"]))
    if len(mode_names) != 3:
        raise ValueError("mode_names must contain wheel, leg and hybrid labels")
    standby_enabled = bool(cfg.get("standby_at_zero_command", True)) and not (
        args.run_zero_policy
    )
    standby_qpos = data.qpos.copy()
    in_standby = standby_enabled and np.allclose(
        command_controller.get(), 0.0, atol=1e-8
    )
    try:
        while viewer_context is None or viewer_context.is_running():
            if command_controller.consume_reset_request():
                _reset_robot(
                    mujoco,
                    model,
                    data,
                    runtime,
                    root_qpos_address,
                    qpos_addresses,
                    cfg["default_dof_pos"],
                    initial_base_position,
                )
                _settle_robot(
                    mujoco,
                    model,
                    data,
                    runtime,
                    qpos_addresses,
                    dof_addresses,
                    settle_time_s,
                )
                standby_qpos = data.qpos.copy()
                in_standby = standby_enabled
                policy_step = 0
                print("Robot reset to its settled standby pose; command is zero.")

            commands = command_controller.get()
            zero_command = np.allclose(commands, 0.0, atol=1e-8)
            if standby_enabled and zero_command:
                if not in_standby:
                    standby_qpos = data.qpos.copy()
                    runtime.reset()
                    in_standby = True
                data.qpos[:] = standby_qpos
                data.qvel[:] = 0.0
                data.qfrc_applied[:] = 0.0
                mujoco.mj_forward(model, data)
                step_start = time.time()
                if viewer_context is not None:
                    viewer_context.cam.lookat[:] = data.xpos[root_body_id]
                    viewer_context.sync()
                    remaining = policy_dt - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
                simulated_time += policy_dt
                if args.duration > 0 and simulated_time + 1e-12 >= args.duration:
                    break
                continue
            if in_standby:
                runtime.reset()
                in_standby = False

            rotation = data.xmat[root_body_id].reshape(3, 3)
            projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
            # MuJoCo's gyro sensor is already expressed in the IMU/site body
            # frame, matching Isaac Gym's quat_rotate_inverse observation.
            angular_velocity = data.sensordata[gyro_slice].copy()
            dof_pos = data.qpos[qpos_addresses].copy()
            dof_vel = data.qvel[dof_addresses].copy()
            actions, _ = runtime.step(
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
            step_start = time.time()
            for _ in range(sim_steps_per_policy):
                # Isaac Gym recomputes PD torque at every 5 ms physics step
                # while keeping the 50 Hz policy action fixed.
                torques = runtime.compute_torques(
                    actions,
                    data.qpos[qpos_addresses],
                    data.qvel[dof_addresses],
                )
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[dof_addresses] = torques.cpu().numpy()
                mujoco.mj_step(model, data)
            if viewer_context is not None:
                viewer_context.cam.lookat[:] = data.xpos[root_body_id]
                viewer_context.sync()
                remaining = policy_dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
            simulated_time += sim_steps_per_policy * model.opt.timestep
            policy_step += 1
            if args.duration > 0 and simulated_time + 1e-12 >= args.duration:
                break
    finally:
        if viewer_context is not None:
            viewer_context.close()
    base_position = data.qpos[root_qpos_address : root_qpos_address + 3]
    print(
        f"Simulation finished: simulated_time={simulated_time:.2f}s, "
        f"policy_steps={policy_step}, "
        f"base_xyz={np.array2string(base_position, precision=3)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="nezha_mine.yaml",
        help=(
            "YAML file name in configs/ (default: nezha_mine.yaml), or a "
            "path relative to deploy/deploy_mujoco."
        ),
    )
    parser.add_argument("--model", help="Floating-base Nezha MJCF or MuJoCo-compatible URDF")
    parser.add_argument("--policy", help="Exported policy.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--history-order",
        choices=("auto", "oldest_first", "newest_first"),
        help=(
            "Override history layout. 'auto' reads new export metadata and "
            "falls back to legacy_history_order for old exports."
        ),
    )
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--linear-step",
        type=float,
        default=0.1,
        help="Velocity increment for W/S/A/D in m/s.",
    )
    parser.add_argument(
        "--yaw-step",
        type=float,
        default=0.1,
        help="Yaw-rate increment for Q/E in rad/s.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "Simulated seconds to run; by default the simulation runs until "
            "the viewer closes. Headless mode requires a value greater than zero."
        ),
    )
    parser.add_argument(
        "--run-zero-policy",
        action="store_true",
        help="Disable zero-command standby and expose the policy's raw zero-command behavior.",
    )
    parser.add_argument(
        "--gate-log-interval",
        type=int,
        default=None,
        help="Print wheel/leg/hybrid gate probabilities every N policy steps; 0 disables it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
