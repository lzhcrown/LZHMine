"""Validate a common non-residual all-MoE starting point for the 2x2 study."""

from .cts_checkpoint import normalize_policy_config


RESIDUAL_POLICY_KEYS = frozenset((
    "residual_enabled", "residual_hidden_dims", "residual_action_scales",
))
BASE_STRUCTURE_KEYS = (
    "student_encoder_type", "expert_num", "actor_type", "actor_expert_num",
    "critic_type", "critic_expert_num",
)


def base_policy_config(config):
    return {key: value for key, value in normalize_policy_config(config).items()
            if key not in RESIDUAL_POLICY_KEYS}


def validate_initialization_source(checkpoint, target_policy):
    """Unlike resume, allow adding a zero-initialized action residual only."""
    source = normalize_policy_config(checkpoint["policy_config"])
    target = normalize_policy_config(target_policy)
    if source["residual_enabled"]:
        raise ValueError("--init_from requires a non-residual all-MoE checkpoint; use --resume for a residual run")
    for name in ("student_encoder_type", "actor_type", "critic_type"):
        if source[name] != "moe" or target[name] != "moe":
            raise ValueError("--init_from requires all-MoE source and target policies")
    for key in BASE_STRUCTURE_KEYS:
        if source[key] != target[key]:
            raise ValueError("Initialization checkpoint differs in " + key)
    saved = checkpoint.get("train_config", {})
    if "algorithm" not in saved or "num_steps_per_env" not in saved.get("runner", {}):
        raise ValueError("Initialization requires a training checkpoint with algorithm and rollout settings")
    return source
