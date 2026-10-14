"""Shared architecture metadata for MINE resume, play and export."""

import copy
import subprocess


OBSERVATION_KEYS = (
    "num_actor_obs", "num_critic_obs", "num_one_step_obs",
    "num_one_step_privileged_obs", "estimation_target_indices", "num_actions",
)


def policy_config_from_checkpoint(checkpoint):
    """Read new metadata, or infer the gate/target view of archived Nezha runs."""
    state = checkpoint["model_state_dict"]
    config = copy.deepcopy(checkpoint.get("policy_config", {}))
    has_gate = "estimator.mode_gate.0.weight" in state
    has_modes = "estimator.mode_prototypes" in state
    if has_gate != has_modes:
        raise ValueError("Checkpoint contains an incomplete MINE mode gate")
    if "use_mode_gate" in config and config["use_mode_gate"] != has_gate:
        raise ValueError("Checkpoint gate metadata does not match its weights")
    config["use_mode_gate"] = has_gate

    target_width = state["estimator.target.0.weight"].shape[1]
    actor_width = config.get("num_one_step_obs", 65)
    privileged_width = config.get("num_one_step_privileged_obs", 375)
    if target_width not in (actor_width, privileged_width):
        raise ValueError(f"Unsupported target encoder input width: {target_width}")
    is_privileged = target_width == privileged_width
    if "is_privileged_obs" in config and config["is_privileged_obs"] != is_privileged:
        raise ValueError("Checkpoint target metadata does not match its weights")
    config["is_privileged_obs"] = is_privileged
    config.setdefault("history_order", "oldest_first")
    return config


def config_to_dict(obj):
    """Plain nested dict of a legged_gym-style config (classes of attributes)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {key: config_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [config_to_dict(value) for value in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if not hasattr(obj, "__dict__"):
        return repr(obj)
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value) and not isinstance(value, type):
            continue
        result[key] = config_to_dict(value)
    return result


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", path, *args],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout


def git_state(path):
    """Commit, branch and dirty flag of the git checkout containing ``path``.

    ``dirty`` covers tracked files only; untracked files are not recorded.
    """
    try:
        return {
            "commit": _git(path, "rev-parse", "HEAD").strip(),
            "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty": bool(_git(path, "status", "--porcelain", "--untracked-files=no").strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None, "dirty": None}


def git_diff(path):
    """Uncommitted changes to tracked files, as a patch against HEAD."""
    try:
        return _git(path, "diff", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return ""
