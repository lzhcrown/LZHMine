"""Train Nezha on platform tiles with M20 rewards and MUJICA's roll/yaw rate penalty."""

from experiments.g12_v1.config import POLICY, configure_environment as configure_v1
from experiments.registry import PLATFORM_V2_EXPERIMENT


KEY = PLATFORM_V2_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": (
        "G12 v1 commands and curriculum with Dreamwaq/highplatform M20 rewards, "
        "MUJICA's roll/yaw rate penalty and a 0.54 m clearance target on all terrains."
    ),
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    from legged_gym.terrains.platform_rewards import platform_v2_reward_scales

    configure_v1(cfg)
    cfg.rewards.platform_scales = platform_v2_reward_scales()
    cfg.rewards.platform_reward_profile = "dreamwaq_highplatform"
    cfg.rewards.platform_heading_target = 0.0
    cfg.rewards.base_height_target = 0.54
    cfg.rewards.platform_base_height_target = cfg.rewards.base_height_target
