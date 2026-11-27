"""Train all terrains with 0.55 m clearance and stronger posture penalties."""

from experiments.g12_v4.config import POLICY, configure_environment as configure_v4
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
        "Shared v4 platform rewards with 0.55 m clearance and hip, standing posture, "
        "and tilt penalties at 75% of the G7 ordinary-terrain weights. "
        "The roll and yaw rate penalty weight is -0.15."
    ),
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    configure_v4(cfg)
    cfg.rewards.platform_reward_profile = "shared_platform"
    cfg.rewards.platform_scales.update(
        hip_default=-6.0,
        stand_still=-2.25,
        orientation=-1.5,
        ang_vel_xz=-0.15,
    )
    cfg.rewards.base_height_target = 0.55
    cfg.rewards.platform_base_height_target = cfg.rewards.base_height_target
