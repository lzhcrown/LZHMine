"""Use the G12 v1 policy with a level-9 ceiling for ordinary terrain."""

from experiments.g12_v1.config import POLICY, configure_environment as configure_v1
from experiments.registry import PLATFORM_V5_EXPERIMENT


KEY = PLATFORM_V5_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "G12 v1 curriculum with levels 0-9 on ordinary terrain and 0-19 on platform blocks and pits.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    configure_v1(cfg)
    cfg.terrain.ordinary_max_level = 9
