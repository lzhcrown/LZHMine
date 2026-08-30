"""Task registration for the Nezha training framework."""

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.nezha import NezhaMINECfg, NezhaMINECfgPPO, NezhaMINEEnv
from legged_gym.utils.task_registry import task_registry


task_registry.register(
    "nezha_mine",
    NezhaMINEEnv,
    NezhaMINECfg(),
    NezhaMINECfgPPO(),
)

__all__ = [
    "LeggedRobot",
    "NezhaMINECfg",
    "NezhaMINECfgPPO",
    "NezhaMINEEnv",
]
