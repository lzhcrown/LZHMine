"""Task registration for the Nezha training framework."""

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.nezha import NezhaMoECTSCfg, NezhaMoECTSCfgPPO, NezhaMoECTSEnv
from legged_gym.utils.task_registry import task_registry


task_registry.register(
    "nezha_moe_cts",
    NezhaMoECTSEnv,
    NezhaMoECTSCfg(),
    NezhaMoECTSCfgPPO(),
)

__all__ = [
    "LeggedRobot",
    "NezhaMoECTSCfg",
    "NezhaMoECTSCfgPPO",
    "NezhaMoECTSEnv",
]
