"""Use a 0.54 m base clearance target for ordinary and platform locomotion."""

from experiments.g12_v3.config import POLICY, configure_environment as configure_v3
from experiments.registry import PLATFORM_V4_EXPERIMENT


KEY = PLATFORM_V4_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "G12 v3 with a 0.54 m target base clearance on all terrains.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    configure_v3(cfg)
    cfg.rewards.base_height_target = 0.54
    cfg.rewards.platform_base_height_target = cfg.rewards.base_height_target
