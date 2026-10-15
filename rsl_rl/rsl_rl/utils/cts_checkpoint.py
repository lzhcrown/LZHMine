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


def validate_cts_checkpoint(checkpoint):
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected a Nezha MoE-CTS checkpoint; other architectures cannot be resumed.")
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported MoE-CTS checkpoint format version")
    for key in ("model_state_dict", "policy_config", "observation_config"):
        if key not in checkpoint:
            raise ValueError("MoE-CTS checkpoint is missing " + key)
    policy = checkpoint["policy_config"]
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
