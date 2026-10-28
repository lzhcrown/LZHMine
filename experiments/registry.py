"""Pure-Python experiment catalog; importing it does not load Isaac Gym."""

import copy
import importlib
from pathlib import Path


EXPERIMENT_KEYS = ("exp01_baseline", "exp02_actor_moe")
REPO_ROOT = Path(__file__).resolve().parents[1]


def clone_config(config):
    """Copy public settings, including mutable defaults inherited from classes.

    Gym's BaseConfig instantiates nested classes but leaves list/dict defaults
    on their classes. A plain deepcopy of those instances still shares those
    defaults, so materialize public fields on each copied instance as well.
    """
    if not hasattr(config, "__dict__"):
        return copy.deepcopy(config)
    result = copy.deepcopy(config)
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if not callable(value):
            setattr(result, name, clone_config(value))
    return result


def get_experiment(key):
    if key not in EXPERIMENT_KEYS:
        raise ValueError("Unknown CTS experiment: {}".format(key))
    return importlib.import_module("experiments.{}.config".format(key))


def configure_experiment(train_cfg, key):
    """Create isolated config instances while keeping common PPO defaults."""
    experiment = get_experiment(key)
    train_cfg = clone_config(train_cfg)
    for section, overrides in (("policy", experiment.POLICY), ("runner", experiment.RUNNER)):
        target = getattr(train_cfg, section)
        for name, value in overrides.items():
            setattr(target, name, copy.deepcopy(value))
    return train_cfg


def export_path(key):
    return str(REPO_ROOT / get_experiment(key).EXPORT_PATH)
