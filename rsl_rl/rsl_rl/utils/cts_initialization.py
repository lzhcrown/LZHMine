"""Validate a common non-residual all-MoE starting point for the 2x2 study."""

from .cts_checkpoint import normalize_policy_config
from .cts_observations import add_skill_id, observation_dimensions


RESIDUAL_POLICY_KEYS = frozenset((
    "residual_enabled", "residual_hidden_dims", "residual_action_scales",
))
BASE_STRUCTURE_KEYS = (
    "student_encoder_type", "expert_num", "actor_type", "actor_expert_num",
    "critic_type", "critic_expert_num",
)


def base_policy_config(config):
    return {key: value for key, value in normalize_policy_config(config).items()
            if key not in RESIDUAL_POLICY_KEYS and key != "critic_task_count"}


def skill_input_extension(checkpoint, target_policy, target_schema):
    """Check that any observation change consists only of appended skill inputs."""
    source_schema = checkpoint["observation_config"]
    base = base_policy_config(checkpoint["policy_config"])
    extend = source_schema != target_schema
    if extend:
        if add_skill_id(source_schema) != target_schema:
            raise ValueError("Initialization observation schema differs from checkpoint")
        current, history, privileged = observation_dimensions(target_schema)
        base.update(num_one_step_obs=current, num_actor_obs=history, num_critic_obs=privileged,
                    observation_dims=[term["dim"] for term in target_schema["proprioceptive_terms"]])
    if base != base_policy_config(target_policy):
        raise ValueError("Initialization must preserve the complete base policy configuration")
    return extend


def extend_skill_input_weights(state, history_length):
    """Copy the trained inputs and initialize appended skill columns to zero.

    The student consumes a term-major history, while teacher, actor and critic
    consume one frame. Actor/critic latent features precede the frame.
    """
    from torch.nn.functional import pad

    key = "teacher_encoder.0.network.0.weight"
    state[key] = pad(state[key], (0, 1))
    for module, columns in (("student_moe_encoder", history_length), ("actor", 1), ("critic", 1)):
        for layer in ("experts.backbone", "gating_network.0"):
            key = "{}.moe.{}.network.0.weight".format(module, layer)
            state[key] = pad(state[key], (0, columns))


def validate_initialization_source(checkpoint, target_policy):
    """Initialize a compatible all-MoE arm from a non-residual source."""
    source = normalize_policy_config(checkpoint["policy_config"])
    target = normalize_policy_config(target_policy)
    if (any(term["name"] == "skill_id" for term in checkpoint["observation_config"]["proprioceptive_terms"])
            and target.get("observation_dims") != source["observation_dims"]):
        raise ValueError("Initialization cannot discard trained skill_id inputs")
    if source["residual_enabled"]:
        raise ValueError("--init_from requires a non-residual all-MoE checkpoint; use --resume for a residual run")
    if source["critic_task_count"] > target["critic_task_count"]:
        raise ValueError("Initialization cannot discard a trained task value head")
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
