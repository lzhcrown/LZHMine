"""Train a geometry-conditioned student with a jointly trained map-attention teacher."""

from experiments.g12_v5.config import POLICY as V5_POLICY, configure_environment as configure_v5
from experiments.registry import SKILL_V2_EXPERIMENT


KEY = SKILL_V2_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
POLICY = dict(V5_POLICY, observation_dims=[3, 3, 3, 16, 16, 16],
              student_frozen_terrain=True, teacher_map_attention=True)
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "Student uses pretrained terrain features. Teacher learns map attention with privileged state and skill ID.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    from rsl_rl.utils.cts_observations import add_skill_id, default_observation_config, observation_dimensions

    configure_v5(cfg)
    schema = cfg.env.observation_config
    if schema is None:
        schema = default_observation_config(history_length=cfg.env.frame_stack)
    schema = add_skill_id(schema, privileged_only=True)
    cfg.env.observation_config = schema
    current, history, privileged = observation_dimensions(schema)
    cfg.env.num_one_step_observations = current
    cfg.env.num_observations = history
    cfg.env.num_privileged_obs = cfg.env.num_one_step_privileged_obs = privileged
