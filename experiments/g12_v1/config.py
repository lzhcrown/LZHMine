"""Train G12 with M20 platform rewards and MUJICA's roll/yaw rate penalty."""

from experiments.g12_high_platform.config import POLICY, configure_environment as configure_g12
from experiments.registry import PLATFORM_V1_EXPERIMENT


KEY = PLATFORM_V1_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "M20 terrain progression and high-platform rewards with MUJICA vertical speed and roll/yaw rate costs.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    from legged_gym.terrains.platform_rewards import platform_v1_reward_scales

    configure_g12(cfg)
    cfg.terrain.curriculum_rule = "dreamwaq"
    cfg.rewards.platform_scales = platform_v1_reward_scales()
    cfg.rewards.base_height_target = 0.50
    cfg.rewards.platform_base_height_target = 0.52
