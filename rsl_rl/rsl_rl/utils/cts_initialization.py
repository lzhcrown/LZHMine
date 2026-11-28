"""Validate and transfer a compatible non-residual all-MoE starting point."""

from rsl_rl.modules.frozen_terrain_encoder import TERRAIN_LATENT_DIM

from .cts_checkpoint import normalize_policy_config
from .cts_observations import add_skill_id, observation_dimensions


RESIDUAL_POLICY_KEYS = frozenset((
    "residual_enabled", "residual_hidden_dims", "residual_action_scales",
))
WARM_START_TARGET_KEYS = RESIDUAL_POLICY_KEYS | {
    "critic_task_count", "teacher_map_attention", "student_frozen_terrain",
}
BASE_STRUCTURE_KEYS = (
    "student_encoder_type", "expert_num", "actor_type", "actor_expert_num",
    "critic_type", "critic_expert_num",
)


def base_policy_config(config):
    return {key: value for key, value in normalize_policy_config(config).items()
            if key not in WARM_START_TARGET_KEYS}


def transfer_skill_inputs(state, checkpoint, target_policy, target_schema):
    """Transfer only canonical, trailing skill inputs between compatible schemas.

    Appended columns start at zero. Dropping the student's ID preserves its
    source prediction when the ID is fixed to zero throughout the history.
    The critic consumes the privileged frame even on student rollouts.
    """
    from torch.nn.functional import pad

    source_schema = checkpoint["observation_config"]
    base = base_policy_config(checkpoint["policy_config"])
    current, history, privileged = observation_dimensions(target_schema)
    source_current, source_history, source_privileged = observation_dimensions(source_schema)
    if source_schema != target_schema:
        student_has_id = any(term["name"] == "skill_id" for term in target_schema["proprioceptive_terms"])
        privileged_only = target_policy["teacher_map_attention"] and not student_has_id
        if add_skill_id(source_schema, privileged_only=privileged_only) != target_schema:
            raise ValueError("Initialization observation schema differs from checkpoint")
        base.update(num_one_step_obs=current, num_actor_obs=history, num_critic_obs=privileged,
                    observation_dims=[term["dim"] for term in target_schema["proprioceptive_terms"]])
    if base != base_policy_config(target_policy):
        raise ValueError("Initialization must preserve the complete base policy configuration")

    privileged_delta = privileged - source_privileged
    if privileged_delta:
        teacher_key = "teacher_encoder.0.network.0.weight"
        state[teacher_key] = pad(state[teacher_key], (0, privileged_delta))
        query_key = "teacher_encoder.map_attention.query.weight"
        if query_key in state:
            state[query_key] = pad(state[query_key], (0, privileged_delta))
    for module, columns in (("student_moe_encoder", history - source_history),
                            ("actor", current - source_current), ("critic", privileged_delta)):
        if columns:
            for layer in ("experts.backbone", "gating_network.0"):
                key = "{}.moe.{}.network.0.weight".format(module, layer)
                state[key] = pad(state[key], (0, columns))

    # First remove any obsolete student ID columns, then append independent
    # zero columns for the frozen map code. Old histories retain their outputs.
    if target_policy["student_frozen_terrain"] and not checkpoint["policy_config"].get("student_frozen_terrain", False):
        for layer in ("experts.backbone", "gating_network.0"):
            key = "student_moe_encoder.moe.{}.network.0.weight".format(layer)
            state[key] = pad(state[key], (0, TERRAIN_LATENT_DIM))


def validate_initialization_source(checkpoint, target_policy):
    """Initialize a compatible all-MoE arm from a non-residual source."""
    source = normalize_policy_config(checkpoint["policy_config"])
    target = normalize_policy_config(target_policy)
    if source["teacher_frozen_terrain"] or target["teacher_frozen_terrain"]:
        raise ValueError("--init_from is unsupported for frozen terrain policies")
    if (any(term["name"] == "skill_id" for term in checkpoint["observation_config"]["proprioceptive_terms"])
            and target.get("observation_dims") != source["observation_dims"]
            and not target["teacher_map_attention"]):
        raise ValueError("Initialization cannot discard trained skill_id inputs")
    if source["residual_enabled"]:
        raise ValueError("--init_from requires a non-residual all-MoE checkpoint; use --resume for a residual run")
    if source["critic_task_count"] > target["critic_task_count"]:
        raise ValueError("Initialization cannot discard a trained task value head")
    if source["teacher_map_attention"] and not target["teacher_map_attention"]:
        raise ValueError("Initialization cannot discard trained teacher map attention")
    if source["student_frozen_terrain"] and not target["student_frozen_terrain"]:
        raise ValueError("Initialization cannot discard the trained student geometry inputs")
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
