"""Train Nezha on platform tiles with M20 rewards and MUJICA's roll/yaw rate penalty."""

from experiments.g12_high_platform.config import POLICY, configure_environment as configure_g12
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
        "Dreamwaq/highplatform M20 rewards with MUJICA's roll/yaw rate penalty "
        "and Nezha's standing clearance."
    ),
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    from legged_gym.terrains.platform_rewards import platform_v2_reward_scales

    configure_g12(cfg)
    cfg.terrain.curriculum_rule = "dreamwaq"
    platform_commands = cfg.commands.wf_settings["terrain_max_command_ranges"]["high_platform"]
    platform_commands.update(lin_vel_x=[0.0, 1.0], resampling_time=10.0)
    cfg.rewards.platform_scales = platform_v2_reward_scales()
    cfg.rewards.platform_reward_profile = "dreamwaq_highplatform"
    cfg.rewards.platform_heading_target = 0.0
    # Trunk-to-ground distance with the URDF's four wheels on level ground
    # at Nezha's default leg angles (hip 0, thigh 0.8, calf -1.6 rad).
    cfg.rewards.platform_base_height_target = 0.556
