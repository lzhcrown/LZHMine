# Adapted from wheel_gym_CQ/wheel_legged_gym/evaluation/geometry.py
"""URDF wheel collision geometry and default-pose FK, without simulator imports."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR


WHEEL_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def _vector(element, key, default="0 0 0"):
    return np.asarray([float(value) for value in
                       (element.get(key, default) if element is not None else default).split()])


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("URDF joint axis cannot be zero")
    x, y, z = axis / norm
    skew = np.asarray([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1. - np.cos(angle)) * (skew @ skew)


def _origin(element):
    result = np.eye(4)
    result[:3, 3] = _vector(element, "xyz")
    roll, pitch, yaw = _vector(element, "rpy")
    result[:3, :3] = (_axis_rotation([0, 0, 1], yaw) @
                      _axis_rotation([0, 1, 0], pitch) @
                      _axis_rotation([1, 0, 0], roll))
    return result


def wheel_geometry_from_cfg(cfg):
    """Return JSON-compatible wheel cylinders and their default-pose placement.

    State tensors provide each wheel link's world pose at runtime. Collision
    offsets/axes below are in that link frame, whereas center_offsets and
    default_rotations describe the reset pose in the robot root frame.
    """
    path = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)).resolve()
    root = ET.parse(str(path)).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    incoming = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
    angles = cfg.init_state.default_joint_angles
    transforms = {}

    def transform(name):
        if name in transforms:
            return transforms[name]
        if name not in incoming:
            result = np.eye(4)
        else:
            joint = incoming[name]
            result = transform(joint.find("parent").attrib["link"]) @ _origin(joint.find("origin"))
            motion = np.eye(4)
            joint_type = joint.attrib["type"]
            value = float(angles.get(joint.attrib["name"], 0.))
            axis = _vector(joint.find("axis"), "xyz", "1 0 0")
            if joint_type in ("revolute", "continuous"):
                motion[:3, :3] = _axis_rotation(axis, value)
            elif joint_type == "prismatic":
                motion[:3, 3] = axis * value
            elif joint_type != "fixed":
                raise ValueError("Unsupported URDF joint type: " + joint_type)
            result = result @ motion
        transforms[name] = result
        return result

    centers, rotations, radii, widths, axes, offsets = [], [], [], [], [], []
    for name in WHEEL_NAMES:
        if name not in links:
            raise ValueError("Expected wheel link is absent from asset: " + name)
        collisions = links[name].findall("collision")
        if len(collisions) != 1 or collisions[0].find("geometry/cylinder") is None:
            raise ValueError("Patch evaluation requires one collision cylinder for " + name)
        cylinder = collisions[0].find("geometry/cylinder")
        collision = _origin(collisions[0].find("origin"))
        pose = transform(name)
        radius, width = float(cylinder.attrib["radius"]), float(cylinder.attrib["length"])
        if radius <= 0 or width <= 0:
            raise ValueError("Wheel cylinder dimensions must be positive")
        centers.append((pose @ collision)[:3, 3])
        rotations.append(pose[:3, :3])
        radii.append(radius)
        widths.append(width)
        axes.append(collision[:3, 2])
        offsets.append(collision[:3, 3])
    centers, rotations = np.asarray(centers), np.asarray(rotations)
    radii, widths, axes = np.asarray(radii), np.asarray(widths), np.asarray(axes)
    default_axes = np.einsum("wij,wj->wi", rotations, axes)
    half_extents = radii[:, None] * np.sqrt(np.maximum(0., 1. - default_axes ** 2)) + widths[:, None] * .5 * np.abs(default_axes)
    return {
        "wheel_names": list(WHEEL_NAMES), "center_offsets": centers.tolist(),
        "default_rotations": rotations.tolist(), "radii": radii.tolist(), "widths": widths.tolist(),
        "collision_axes": axes.tolist(), "collision_offsets": np.asarray(offsets).tolist(),
        "default_axes": default_axes.tolist(), "default_half_extents": half_extents.tolist(),
        "left_wheel_y": float(centers[[0, 2], 1].mean()),
        "right_wheel_y": float(centers[[1, 3], 1].mean()),
        "front_wheel_x": float(centers[[0, 1], 0].mean()),
        "rear_wheel_x": float(centers[[2, 3], 0].mean()),
        "wheel_radius": float(radii.max()),
        "wheel_x_min": float((centers[:, 0] - half_extents[:, 0]).min()),
        "wheel_x_max": float((centers[:, 0] + half_extents[:, 0]).max()),
        "wheel_y_min": float((centers[:, 1] - half_extents[:, 1]).min()),
        "wheel_y_max": float((centers[:, 1] + half_extents[:, 1]).max()),
        "asset_path": str(path), "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
