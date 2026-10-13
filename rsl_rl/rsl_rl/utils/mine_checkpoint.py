"""Shared architecture metadata for MINE resume, play and export."""

import copy


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
