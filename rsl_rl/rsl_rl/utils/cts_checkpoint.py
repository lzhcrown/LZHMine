"""Versioned MoE-CTS checkpoints shared by training, play and export."""

import copy
import torch
from rsl_rl.utils.cts_observations import validate_observation_config


ARCHITECTURE = "nezha_moe_cts"
FORMAT_VERSION = 1
DIMENSION_KEYS = (
    "num_actor_obs", "num_critic_obs", "num_one_step_obs", "num_actions",
    "history_length", "observation_dims",
)
STRUCTURE_KEYS = (
    "student_encoder_type", "expert_num", "actor_type", "actor_expert_num",
    "critic_type", "critic_expert_num",
)


def normalize_policy_config(policy_config):
    """Make all CTS module types explicit, preserving legacy checkpoint defaults."""
    policy = copy.deepcopy(policy_config)
    for type_key, count_key, legacy_type in (
        ("student_encoder_type", "expert_num", "moe"),
        ("actor_type", "actor_expert_num", "mlp"),
        ("critic_type", "critic_expert_num", "mlp"),
    ):
        module_type = policy.setdefault(type_key, legacy_type)
        if module_type not in ("mlp", "moe"):
            raise ValueError("Unsupported CTS {}: {}".format(type_key, module_type))
        if module_type == "mlp":
            policy[count_key] = 0
        else:
            count = policy.setdefault(count_key, 8)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("MoE {} must be a positive integer".format(count_key))
    return policy


def validate_checkpoint_policy(checkpoint, expected_policy):
    """Reject any student/actor/critic mismatch before restoring an experiment."""
    if expected_policy is None:
        return checkpoint
    policy = normalize_policy_config(checkpoint["policy_config"])
    expected = normalize_policy_config(expected_policy)
    for key in STRUCTURE_KEYS:
        if policy[key] != expected[key]:
            raise ValueError(
                "Checkpoint {}={!r} does not match this experiment's {}={!r}. "
                "Use the matching experiment entry point or start a fresh training run."
                .format(key, policy[key], key, expected[key])
            )
    return checkpoint


def validate_checkpoint_actor(checkpoint, expected_actor_type, expected_actor_expert_num=None):
    """Reject cross-experiment checkpoints before restoring or exporting them."""
    if expected_actor_type is None:
        return checkpoint
    policy = normalize_policy_config(checkpoint["policy_config"])
    if policy["actor_type"] != expected_actor_type:
        raise ValueError(
            "Checkpoint actor_type={!r} does not match this experiment's actor_type={!r}. "
            "Use the matching experiment entry point or start a fresh training run."
            .format(policy["actor_type"], expected_actor_type)
        )
    if (expected_actor_expert_num is not None
            and policy["actor_expert_num"] != expected_actor_expert_num):
        raise ValueError(
            "Checkpoint actor_expert_num={} does not match this experiment's actor_expert_num={}."
            .format(policy["actor_expert_num"], expected_actor_expert_num)
        )
    return checkpoint


def validate_cts_checkpoint(checkpoint):
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected a Nezha MoE-CTS checkpoint; other architectures cannot be resumed.")
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported MoE-CTS checkpoint format version")
    for key in ("model_state_dict", "policy_config", "observation_config"):
        if key not in checkpoint:
            raise ValueError("MoE-CTS checkpoint is missing " + key)
    policy = normalize_policy_config(checkpoint["policy_config"])
    # Detect metadata/weight disagreement before constructing any network.
    for type_key, prefix in (
        ("student_encoder_type", "student_moe_encoder.moe."),
        ("actor_type", "actor.moe."),
        ("critic_type", "critic.moe."),
    ):
        has_moe = any(key.startswith(prefix) for key in checkpoint["model_state_dict"])
        if has_moe != (policy[type_key] == "moe"):
            raise ValueError("Checkpoint {} disagrees with saved weights".format(type_key))
    spec = checkpoint["observation_config"]
    validate_observation_config(spec)
    for key in DIMENSION_KEYS:
        if key not in policy:
            raise ValueError("Checkpoint policy_config is missing " + key)
    dims = [int(term["dim"]) for term in spec["proprioceptive_terms"]]
    privileged_dim = sum(int(term["dim"]) for term in spec["privileged_terms"])
    history = int(spec["history_length"])
    if spec.get("history_layout") != "term_major" or spec.get("history_order") != "oldest_first":
        raise ValueError("MoE-CTS requires term-major, oldest-first observations")
    expected = {
        "num_actor_obs": sum(dims) * history,
        "num_critic_obs": privileged_dim,
        "num_one_step_obs": sum(dims),
        "history_length": history,
        "observation_dims": dims,
    }
    for key, value in expected.items():
        if policy[key] != value:
            raise ValueError("Checkpoint observation schema disagrees with policy " + key)
    checkpoint["policy_config"] = policy
    return checkpoint


def load_cts_checkpoint(path, map_location="cpu"):
    # These are local training checkpoints, which also contain optimizer/config data.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return validate_cts_checkpoint(checkpoint)


def policy_from_checkpoint(checkpoint, device="cpu"):
    from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS

    validate_cts_checkpoint(checkpoint)
    policy = ActorCriticMoECTS(**copy.deepcopy(checkpoint["policy_config"])).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.observation_config = copy.deepcopy(checkpoint["observation_config"])
    return policy.eval()
