"""Task registration for the Nezha training framework."""

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.nezha import NezhaMoECTSCfg, NezhaMoECTSCfgPPO, NezhaMoECTSEnv
from legged_gym.utils.task_registry import task_registry
from experiments.registry import EXPERIMENT_KEYS, configure_experiment, get_experiment


for _key in EXPERIMENT_KEYS:
    _experiment = get_experiment(_key)
    task_registry.register(
        _experiment.TASK_NAME,
        NezhaMoECTSEnv,
        NezhaMoECTSCfg(),
        configure_experiment(NezhaMoECTSCfgPPO(), _key),
    )

__all__ = [
    "LeggedRobot",
    "NezhaMoECTSCfg",
    "NezhaMoECTSCfgPPO",
    "NezhaMoECTSEnv",
]
