"""WF observation contract shared by Isaac Gym training and deployment.

All functions use plain PyTorch; importing this module does not load Isaac Gym.
The JSON-compatible schema is stored in checkpoints so changing a training
configuration cannot silently change a deployed policy's input interpretation.
"""

from copy import deepcopy

import torch


def default_observation_config(history_length=4, wheel_indices=None):
    """Return a fresh WF 57 x 4 / 295 observation schema."""
    def term(name, dim, scale=1.0, noise=0.0, clip=100.0):
        return {"name": name, "dim": dim, "scale": scale,
                "noise": noise, "clip": [-clip, clip]}

    proprio = [
        term("base_ang_vel", 3, 0.25, 0.2),
        term("projected_gravity", 3, 1.0, 0.05),
        term("velocity_commands", 3, [2.0, 2.0, 0.25]),
        term("joint_pos", 16, 1.0, 0.03),
        term("joint_vel", 16, 0.05, 2.0),
        term("actions", 16),
    ]
    privileged = [term("base_lin_vel", 3, 2.0)] + deepcopy(proprio)
    for item in privileged:
        item["noise"] = 0.0
        if item["name"] == "velocity_commands":
            item["scale"] = 1.0
    privileged += [
        term("joint_acc", 16, 1e-4),
        term("joint_torque", 16, 0.01),
        term("contact_force", 16, 1e-3),
        term("height_scan", 187, 2.5, clip=1.0),
    ]
    config = {
        "schema_version": 1,
        "history_length": int(history_length),
        "history_order": "oldest_first",
        "history_layout": "term_major",
        "wheel_indices": list([3, 7, 11, 15] if wheel_indices is None else wheel_indices),
        "height_offset": 0.5,
        "height_grid_order": "xy",
        "contact_history_length": 3,
        "proprioceptive_terms": proprio,
        "privileged_terms": privileged,
    }
    validate_observation_config(config)
    return config


def validate_observation_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported CTS observation schema version")
    if config.get("history_order") != "oldest_first" or config.get("history_layout") != "term_major":
        raise ValueError("CTS requires oldest_first, term_major histories")
    if int(config["history_length"]) != config["history_length"] or config["history_length"] < 1:
        raise ValueError("history_length must be a positive integer")
    for group in ("proprioceptive_terms", "privileged_terms"):
        if not config[group]:
            raise ValueError("Observation term groups cannot be empty")
        names = set()
        for term in config[group]:
            if term["name"] in names or not isinstance(term["dim"], int) or term["dim"] < 1:
                raise ValueError("Observation terms need unique names and positive integer dimensions")
            names.add(term["name"])
            if len(term["clip"]) != 2 or term["clip"][0] > term["clip"][1]:
                raise ValueError("Invalid observation clipping bounds")
            if isinstance(term["scale"], (list, tuple)) and len(term["scale"]) != term["dim"]:
                raise ValueError("Vector observation scales must match term dimensions")
            if term.get("noise", 0.0) < 0:
                raise ValueError("Uniform observation noise amplitudes must be nonnegative")
            if term["name"] == "joint_pos" and any(
                index < 0 or index >= term["dim"] for index in config["wheel_indices"]
            ):
                raise ValueError("Wheel indices lie outside the joint position observation")
    return config


def observation_dimensions(config):
    """Return (current proprioception, flattened history, privileged frame)."""
    current = sum(term["dim"] for term in config["proprioceptive_terms"])
    privileged = sum(term["dim"] for term in config["privileged_terms"])
    return current, current * int(config["history_length"]), privileged


def _build_terms(raw, terms, wheel_indices, add_noise, noise_level):
    values = []
    for term in terms:
        name = term["name"]
        value = raw[name]
        if value.shape[-1] != term["dim"]:
            raise ValueError("{} has {} values, expected {}".format(name, value.shape[-1], term["dim"]))
        if name == "joint_pos":
            value = value.clone()
            value[..., wheel_indices] = 0.0
        if add_noise and term.get("noise", 0.0):
            value = value + (2.0 * torch.rand_like(value) - 1.0) * term["noise"] * noise_level
        value = value.clamp(term["clip"][0], term["clip"][1])
        value = value * value.new_tensor(term["scale"])
        # WF also masks observation noise at wheel-position columns.
        if name == "joint_pos":
            value[..., wheel_indices] = 0.0
        values.append(value)
    return torch.cat(values, dim=-1)


def build_proprioception(raw, config=None, add_noise=False, noise_level=1.0):
    """Construct a current frame: uniform noise, clipping, then scaling.

    ``raw['joint_pos']`` is already relative to the robot's default pose.
    Wheel rates are preserved; only wheel positions are masked.
    """
    config = default_observation_config() if config is None else config
    return _build_terms(raw, config["proprioceptive_terms"], config["wheel_indices"], add_noise, noise_level)


def build_privileged_observations(raw, config=None):
    """Construct the clean teacher/critic frame from unscaled quantities."""
    config = default_observation_config() if config is None else config
    return _build_terms(raw, config["privileged_terms"], config["wheel_indices"], False, 0.0)


def flatten_history(history, term_dims):
    """Flatten [..., time, features] in WF's term-major, oldest-first order."""
    if history.shape[-1] != sum(term_dims):
        raise ValueError("History frame width does not match observation term dimensions")
    return torch.cat([part.flatten(start_dim=-2) for part in torch.split(history, term_dims, dim=-1)], dim=-1)


def extract_current_observation(history, history_length, term_dims):
    """Recover the newest frame from a term-major flattened history."""
    if history.shape[-1] != int(history_length) * sum(term_dims):
        raise ValueError("Flattened history width does not match the observation contract")
    values = []
    offset = 0
    for width in term_dims:
        end = offset + int(history_length) * width
        values.append(history[..., end - width:end])
        offset = end
    return torch.cat(values, dim=-1)


def contact_history_features(history):
    """[N, 3, 4, xyz], newest first -> [max4, current4, previous4, oldest4]."""
    norms = history.norm(dim=-1)
    return torch.cat((norms.max(dim=-2).values, norms.flatten(start_dim=-2)), dim=-1)


class CTSObservationHistory:
    """History with independent reset handling for each vector environment."""

    def __init__(self, num_envs, config=None, device="cpu", dtype=torch.float32):
        self.config = deepcopy(default_observation_config() if config is None else config)
        validate_observation_config(self.config)
        self.term_dims = [term["dim"] for term in self.config["proprioceptive_terms"]]
        self.buffer = torch.zeros(num_envs, self.config["history_length"], sum(self.term_dims), device=device, dtype=dtype)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.initialized.zero_()
            self.buffer.zero_()
        else:
            self.initialized[env_ids] = False
            self.buffer[env_ids] = 0.0

    def update(self, frame):
        if frame.shape != self.buffer[:, -1].shape:
            raise ValueError("Current observation frame shape does not match the history buffer")
        self.buffer[:, :-1] = self.buffer[:, 1:].clone()
        self.buffer[:, -1] = frame
        first = ~self.initialized
        self.buffer[first] = frame[first].unsqueeze(1)
        self.initialized.fill_(True)
        return flatten_history(self.buffer, self.term_dims)
