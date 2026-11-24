"""Task registration for the Nezha training framework."""

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.nezha import NezhaMoECTSCfg, NezhaMoECTSCfgPPO, NezhaMoECTSEnv
from legged_gym.utils.task_registry import task_registry
from experiments.registry import (EXPERIMENT_KEYS, PLATFORM_EXPERIMENT_KEYS,
                                  clone_config, configure_experiment, get_experiment)


for _key in EXPERIMENT_KEYS:
    _experiment = get_experiment(_key)
    _env_cfg = NezhaMoECTSCfg()
    if _key in PLATFORM_EXPERIMENT_KEYS:
        _env_cfg = clone_config(_env_cfg)
        _experiment.configure_environment(_env_cfg)
    task_registry.register(
        _experiment.TASK_NAME,
        NezhaMoECTSEnv,
        _env_cfg,
        configure_experiment(NezhaMoECTSCfgPPO(), _key),
    )

__all__ = [
    "LeggedRobot",
    "NezhaMoECTSCfg",
    "NezhaMoECTSCfgPPO",
    "NezhaMoECTSEnv",
]
