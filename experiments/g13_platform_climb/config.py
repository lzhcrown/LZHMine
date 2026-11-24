"""G12 policy and terrain with directional climbing rewards and advancement."""

from experiments.g12_high_platform.config import POLICY, configure_environment as configure_platform_task
from experiments.registry import PLATFORM_CLIMB_EXPERIMENT


KEY = PLATFORM_CLIMB_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "Shared G12 policy with direction-aware platform progress and completion-based advancement.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    from legged_gym.terrains.platform_rewards import platform_climb_settings

    configure_platform_task(cfg)
    cfg.rewards.platform_climb = True
    cfg.rewards.platform_climb_settings = platform_climb_settings()
