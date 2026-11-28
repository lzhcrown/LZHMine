"""Versioned MoE-CTS checkpoints shared by training, play and export."""

import copy
import math
import os
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
    "critic_task_count", "teacher_map_attention", "teacher_frozen_terrain", "student_frozen_terrain",
    "residual_enabled", "residual_hidden_dims", "residual_action_scales",
)


def normalize_policy_config(policy_config):
    """Make all CTS module types explicit, preserving legacy checkpoint defaults."""
    policy = copy.deepcopy(policy_config)
    task_count = policy.setdefault("critic_task_count", 1)
    if type(task_count) is not int or task_count not in (1, 2):
        raise ValueError("critic_task_count must be 1 or 2")
    for key in ("teacher_map_attention", "teacher_frozen_terrain", "student_frozen_terrain"):
        if not isinstance(policy.setdefault(key, False), bool):
            raise ValueError(key + " must be a boolean")
    if policy["teacher_map_attention"] and policy["teacher_frozen_terrain"]:
        raise ValueError("teacher_map_attention and teacher_frozen_terrain are mutually exclusive")
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
    enabled = policy.setdefault("residual_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("residual_enabled must be a boolean")
    widths = policy.setdefault("residual_hidden_dims", [128, 64])
    if not isinstance(widths, (list, tuple)) or not widths or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in widths):
        raise ValueError("Residual hidden dimensions must be positive integers")
    policy["residual_hidden_dims"] = list(widths)
    scales = policy.setdefault("residual_action_scales", None)
    if enabled:
        if policy["actor_type"] != "moe":
            raise ValueError("Action residual experiments require a MoE Actor")
        if not isinstance(scales, (list, tuple)) or not scales:
            raise ValueError("Enabled residual requires explicit per-action scales")
        if any(isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0
               for scale in scales):
            raise ValueError("Residual scales must be finite and strictly positive")
        if "num_actions" in policy and len(scales) != policy["num_actions"]:
            raise ValueError("Residual action scales must have one entry per action")
        policy["residual_action_scales"] = [float(scale) for scale in scales]
    elif scales is not None:
        raise ValueError("Disabled residual must not configure action scales")
    return policy


def resolve_map_encoder_checkpoint(frozen, path, resuming):
    """A fresh frozen map encoder needs stage-one weights. Resume carries its own copy."""
    if path is None:
        if frozen and not resuming:
            raise ValueError("A fresh frozen terrain policy requires --map_encoder_checkpoint")
        return None
    if not frozen:
        raise ValueError("--map_encoder_checkpoint requires a frozen terrain policy")
    if resuming:
        raise ValueError("--resume restores its own terrain encoder; omit --map_encoder_checkpoint")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError("Map encoder checkpoint does not exist: " + path)
    return path


def validate_checkpoint_policy(checkpoint, expected_policy):
    """Reject any student/actor/critic mismatch before restoring an experiment."""
    if expected_policy is None:
        return checkpoint
    policy = normalize_policy_config(checkpoint["policy_config"])
    expected = normalize_policy_config(expected_policy)
    spec = checkpoint.get("observation_config", {})
    has_skill = any(term["name"] == "skill_id" for term in spec.get("proprioceptive_terms", ()))
    if ("observation_dims" in expected and expected["observation_dims"] != policy.get("observation_dims")
            or has_skill and "observation_dims" not in expected):
        if has_skill and expected["teacher_map_attention"] and policy["teacher_map_attention"]:
            raise ValueError("Checkpoint observation_dims include the legacy student skill_id. "
                             "Use a compatible explicit --init_from migration before changing student inputs.")
        raise ValueError("Checkpoint observation_dims do not match the selected experiment")
    if (expected["teacher_map_attention"] or expected["teacher_frozen_terrain"]) and not any(
            term["name"] == "skill_id" for term in spec.get("privileged_terms", ())):
        raise ValueError("The terrain teacher requires skill_id in the checkpoint's privileged observations")
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
    state = checkpoint["model_state_dict"]
    has_map_attention = any(key.startswith("teacher_encoder.map_attention.") for key in state)
    if has_map_attention != policy["teacher_map_attention"]:
        raise ValueError("Checkpoint teacher_map_attention disagrees with saved weights")
    has_frozen_terrain = any(key.startswith("teacher_encoder.terrain_encoder.") for key in state)
    if has_frozen_terrain != policy["teacher_frozen_terrain"]:
        raise ValueError("Checkpoint teacher_frozen_terrain disagrees with saved weights")
    student_map_prefix = "student_moe_encoder.terrain_encoder."
    has_student_map = any(key.startswith(student_map_prefix) for key in state)
    if has_student_map != policy["student_frozen_terrain"]:
        raise ValueError("Checkpoint student_frozen_terrain disagrees with saved weights")
    if has_frozen_terrain and has_student_map:
        # v1 shares one physical encoder. A conflicting second copy must not
        # overwrite the first as load_state_dict traverses the module tree.
        teacher_prefix = "teacher_encoder.terrain_encoder."
        teacher_map = {k[len(teacher_prefix):]: v for k, v in state.items() if k.startswith(teacher_prefix)}
        student_map = {k[len(student_map_prefix):]: v for k, v in state.items() if k.startswith(student_map_prefix)}
        if teacher_map.keys() != student_map.keys() or any(
                not torch.equal(v, student_map[k]) for k, v in teacher_map.items()):
            raise ValueError("Checkpoint contains conflicting weights for the shared frozen terrain encoder")
    has_residual = any(key.startswith("actor.residual.") for key in state)
    has_scales = "actor.residual_action_scales" in state
    if has_residual != policy["residual_enabled"] or has_scales != policy["residual_enabled"]:
        raise ValueError("Checkpoint residual configuration disagrees with saved weights")
    if has_scales:
        scales = state["actor.residual_action_scales"].detach().cpu()
        expected_scales = torch.tensor(policy["residual_action_scales"], dtype=scales.dtype)
        if scales.shape != expected_scales.shape or not torch.allclose(
                scales, expected_scales, rtol=1e-6, atol=1e-8):
            raise ValueError("Checkpoint residual action scales disagree with metadata")
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
    policy = ActorCriticMoECTS(**copy.deepcopy(checkpoint["policy_config"]),
                              observation_config=checkpoint["observation_config"]).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.observation_config = copy.deepcopy(checkpoint["observation_config"])
    return policy.eval()
