"""G7 Actor and CTS student with terrain-specific rewards and value outputs."""

from experiments.g7_all_moe.config import POLICY as G7_POLICY
from experiments.registry import PLATFORM_EXPERIMENT


KEY = PLATFORM_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
POLICY = dict(G7_POLICY, critic_task_count=2)
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "Shared G7 Actor with separate ordinary and high-platform rewards and values.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    """Apply the task settings after loading the source environment on warm start."""
    from legged_gym.terrains.platform_rewards import platform_reward_scales
    from legged_gym.terrains.wf import high_platform_wf_sub_terrains

    # A warm start restores reward settings saved by the source experiment.
    for name in ("platform_climb", "platform_climb_settings", "platform_base_height_target"):
        if name in vars(cfg.rewards):
            delattr(cfg.rewards, name)
    cfg.terrain.curriculum_rule = "wf"
    cfg.terrain.wf_sub_terrains = high_platform_wf_sub_terrains()
    cfg.terrain.terrain_proportions = [
        spec["proportion"] for spec in cfg.terrain.wf_sub_terrains.values()
    ]
    cfg.commands.wf_settings["terrain_max_command_ranges"]["high_platform"] = {
        "lin_vel_x": [0.5, 1.0], "lin_vel_y": [0.0, 0.0], "ang_vel_yaw": [0.0, 0.0],
    }
    cfg.rewards.platform_scales = platform_reward_scales()
