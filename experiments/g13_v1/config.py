"""Condition the G12 v1 low-level policy on ordinary or platform locomotion."""

from experiments.g12_v1.config import POLICY as V1_POLICY, configure_environment as configure_v1
from experiments.registry import SKILL_V1_EXPERIMENT


KEY = SKILL_V1_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
POLICY = dict(V1_POLICY, observation_dims=[3, 3, 3, 16, 16, 16, 1])
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "G12 v1 rewards with an explicit ordinary/platform skill ID for low-level CTS training.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    from rsl_rl.utils.cts_observations import add_skill_id, default_observation_config, observation_dimensions

    configure_v1(cfg)
    schema = cfg.env.observation_config
    if schema is None:
        schema = default_observation_config(history_length=cfg.env.frame_stack)
    schema = add_skill_id(schema)
    cfg.env.observation_config = schema
    current, history, privileged = observation_dimensions(schema)
    cfg.env.num_one_step_observations = current
    cfg.env.num_observations = history
    cfg.env.num_privileged_obs = cfg.env.num_one_step_privileged_obs = privileged
