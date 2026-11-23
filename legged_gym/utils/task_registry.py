# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import copy
import os
from datetime import datetime

from rsl_rl.runners import MoECTSOnPolicyRunner
from rsl_rl.utils.cts_checkpoint import DIMENSION_KEYS, load_cts_checkpoint, validate_checkpoint_policy
from rsl_rl.utils.cts_initialization import RESIDUAL_POLICY_KEYS, validate_initialization_source

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.terrains.wf import PLATFORM_TYPES
from experiments.registry import PLATFORM_EXPERIMENT, RESIDUAL_EXPERIMENT_KEYS, clone_config, get_experiment
from .helpers import (
    get_args, update_cfg_from_args, class_to_dict, update_class_from_dict,
    get_load_path, set_seed, parse_sim_params,
)


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = clone_config(env_cfg)
        self.train_cfgs[name] = clone_config(train_cfg)

    def get_task_class(self, name):
        return self.task_classes[name]

    def get_cfgs(self, name):
        # Play/resume overrides must not mutate defaults for subsequent envs.
        env_cfg = copy.deepcopy(self.env_cfgs[name])
        train_cfg = copy.deepcopy(self.train_cfgs[name])
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    @staticmethod
    def _resume_path(train_cfg, log_root="default"):
        if not train_cfg.runner.resume:
            return None
        if train_cfg.runner.resume_path:
            return train_cfg.runner.resume_path
        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        if log_root is None:
            raise ValueError("Resume requires a log root or runner.resume_path")
        return get_load_path(log_root, train_cfg.runner.load_run, train_cfg.runner.checkpoint)

    @staticmethod
    def _validate_experiment_checkpoint(train_cfg, checkpoint):
        experiment_key = getattr(train_cfg.runner, "experiment_key", None)
        if experiment_key:
            validate_checkpoint_policy(checkpoint, class_to_dict(train_cfg.policy))
        if experiment_key in RESIDUAL_EXPERIMENT_KEYS:
            saved = checkpoint.get("train_config", {})
            source_key = saved.get("runner", {}).get("experiment_key")
            if source_key != experiment_key:
                raise ValueError(
                    "Comparison resume requires the same experiment_key: "
                    "checkpoint={!r}, target={!r}. Use --init_from for a new comparison arm."
                    .format(source_key, experiment_key)
                )
            factors = class_to_dict(getattr(train_cfg.runner, "comparison_factors", {}))
            expected_temporal = factors.get("temporal")
            temporal_coef = saved.get("algorithm", {}).get("temporal_loss_coef", 0.0)
            if not isinstance(expected_temporal, bool) or (temporal_coef > 0) != expected_temporal:
                raise ValueError(
                    "Checkpoint temporal_loss_coef={!r} disagrees with target comparison "
                    "factor temporal={!r}".format(temporal_coef, expected_temporal)
                )

    @staticmethod
    def _restore_asset_path(env_cfg):
        """Relocate a stale checkout path without replacing an existing asset."""
        asset = getattr(env_cfg, "asset", None)
        configured_path = getattr(asset, "file", None)
        if configured_path is None:
            return
        resolved_path = os.fspath(configured_path).format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        if os.path.isfile(resolved_path):
            return
        marker = os.sep + "resources" + os.sep
        if os.path.isabs(resolved_path) and marker in resolved_path:
            suffix = resolved_path.split(marker, 1)[1]
            relocated_path = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", suffix)
            if os.path.isfile(relocated_path):
                asset.file = relocated_path
                return
            raise FileNotFoundError(
                "Checkpoint asset does not exist: {}. Relocated asset also missing: {}"
                .format(resolved_path, relocated_path)
            )
        raise FileNotFoundError("Checkpoint asset does not exist: " + resolved_path)

    @staticmethod
    def _initialization_path(train_cfg, args):
        path = getattr(args, "cts_init_from", None) or getattr(args, "init_from", None)
        if path is None:
            return None
        if train_cfg.runner.resume or getattr(args, "from_scratch", False):
            raise ValueError("--init_from cannot be combined with --resume or --from_scratch")
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError("Initialization checkpoint does not exist: " + path)
        return path

    @staticmethod
    def restore_environment_config(env_cfg, saved, warm_start=False):
        """Load the saved WF map for replay or use the new map on a legacy warm start."""
        saved_terrain = saved.get("terrain", {})
        target = getattr(env_cfg, "terrain", None)
        legacy_wf = (saved_terrain.get("generator_profile") == "legbot_wf"
                     and getattr(target, "generator_profile", None) == "legbot_wf"
                     and "num_rows" in saved_terrain and "difficulty_reference_rows" not in saved_terrain)
        if legacy_wf:
            saved_terrain = saved_terrain.copy()
            if warm_start:
                # New runs use the target curriculum; other source settings
                # (including tuned commands and rewards) still transfer.
                for key in ("num_rows", "wf_sub_terrains", "terrain_proportions"):
                    saved_terrain.pop(key, None)
                target_ranges = env_cfg.commands.wf_settings["terrain_max_command_ranges"]
            else:
                saved_terrain["difficulty_reference_rows"] = saved_terrain["num_rows"]
            saved = dict(saved, terrain=saved_terrain)
        elif warm_start and saved_terrain.get("generator_profile") == "legbot_wf":
            # The first 20-row maps still used the original 10 cm stair grid.
            # Transfer any tuned heights while adopting the target tread geometry.
            specs = saved_terrain.get("wf_sub_terrains", {}).copy()
            target_specs = getattr(target, "wf_sub_terrains", {})
            for kind in ("stairs_up", "stairs_down"):
                if (kind in specs and "horizontal_scale" not in specs[kind]
                        and "horizontal_scale" in target_specs.get(kind, {})):
                    specs[kind] = dict(specs[kind], step_width=target_specs[kind]["step_width"],
                                       horizontal_scale=target_specs[kind]["horizontal_scale"])
            if specs:
                saved = dict(saved, terrain=dict(saved_terrain, wf_sub_terrains=specs))
        update_class_from_dict(env_cfg, saved)
        if legacy_wf and warm_start:
            settings = env_cfg.commands.wf_settings
            ranges = settings["terrain_max_command_ranges"].copy()
            for kind in PLATFORM_TYPES:
                if kind not in ranges:
                    limits = ranges["obstacles"] if "obstacles" in ranges else target_ranges[kind]
                    ranges[kind] = copy.deepcopy(limits)
            env_cfg.commands.wf_settings = dict(settings, terrain_max_command_ranges=ranges)

    def make_env(self, name, args=None, env_cfg=None):
        args = get_args() if args is None else args
        if name not in self.task_classes:
            raise ValueError("Task with name: {} was not registered".format(name))
        default_env_cfg, train_cfg = self.get_cfgs(name)
        if env_cfg is None:
            env_cfg = default_env_cfg
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        init_path = self._initialization_path(train_cfg, args)
        resume_path = self._resume_path(train_cfg)
        source_path = init_path or resume_path
        if source_path is not None:
            checkpoint = load_cts_checkpoint(source_path)
            if init_path:
                validate_initialization_source(checkpoint, class_to_dict(train_cfg.policy))
            else:
                self._validate_experiment_checkpoint(train_cfg, checkpoint)
            if "environment_config" in checkpoint:
                self.restore_environment_config(env_cfg, checkpoint["environment_config"], warm_start=bool(init_path))
            if init_path and getattr(train_cfg.runner, "experiment_key", None) == PLATFORM_EXPERIMENT:
                # A G7 checkpoint carries its old terrain mix and command ranges.
                get_experiment(PLATFORM_EXPERIMENT).configure_environment(env_cfg)
            self._restore_asset_path(env_cfg)
            # Restore the schema before simulator buffers are allocated.
            env_cfg.env.observation_config = copy.deepcopy(checkpoint["observation_config"])
            # Keep the resolved file even after a new run directory exists.
            if init_path:
                args.cts_init_from = init_path
            else:
                args.cts_resume_path = resume_path
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
        env = self.task_classes[name](
            cfg=env_cfg, sim_params=sim_params, physics_engine=args.physics_engine,
            sim_device=args.sim_device, headless=args.headless,
        )
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        args = get_args() if args is None else args
        if getattr(args, "horovod", False):
            raise ValueError("MoE-CTS currently supports single-device training; remove --horovod")
        if train_cfg is None:
            if name is None:
                raise ValueError("Either name or train_cfg must be provided")
            _, train_cfg = self.get_cfgs(name)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        if train_cfg.runner_class_name != "MoECTSOnPolicyRunner":
            raise ValueError("This branch trains Nezha with MoECTSOnPolicyRunner")
        init_path = self._initialization_path(train_cfg, args)
        resume_path = getattr(args, "cts_resume_path", None) or self._resume_path(train_cfg, log_root)
        if resume_path is not None:
            checkpoint = load_cts_checkpoint(resume_path)
            self._validate_experiment_checkpoint(train_cfg, checkpoint)
            # Architecture was checked before restoring the saved training settings.
            for key, value in checkpoint["policy_config"].items():
                if key not in DIMENSION_KEYS:
                    setattr(train_cfg.policy, key, copy.deepcopy(value))
            update_class_from_dict(train_cfg.algorithm, checkpoint["train_config"]["algorithm"])
            saved_runner = checkpoint["train_config"]["runner"]
            train_cfg.runner.num_steps_per_env = saved_runner["num_steps_per_env"]
            if (getattr(args, "run_name", None) is None
                    and not getattr(train_cfg.runner, "experiment_key", None)):
                train_cfg.runner.run_name = saved_runner.get("run_name") or (
                    "cts_actor_moe" if train_cfg.policy.actor_type == "moe" else "cts_mlp_baseline"
                )
            # Keep this experiment's labels, including when resuming a compatible
            # legacy checkpoint whose old directory is no longer an active group.
            if not getattr(train_cfg.runner, "experiment_key", None):
                for key in ("wandb_tags", "wandb_notes"):
                    if key in saved_runner:
                        setattr(train_cfg.runner, key, copy.deepcopy(saved_runner[key]))
        if init_path is not None:
            checkpoint = load_cts_checkpoint(init_path)
            validate_initialization_source(checkpoint, class_to_dict(train_cfg.policy))
            # Restore the base architecture/hyperparameters in all four arms;
            # retain only the intended residual and temporal experiment factors.
            for key, value in checkpoint["policy_config"].items():
                if key not in DIMENSION_KEYS and key not in RESIDUAL_POLICY_KEYS and key != "critic_task_count":
                    setattr(train_cfg.policy, key, copy.deepcopy(value))
            temporal_coef = getattr(train_cfg.algorithm, "temporal_loss_coef", 0.0)
            update_class_from_dict(train_cfg.algorithm, checkpoint["train_config"]["algorithm"])
            train_cfg.algorithm.temporal_loss_coef = temporal_coef
            train_cfg.runner.num_steps_per_env = checkpoint["train_config"]["runner"]["num_steps_per_env"]
        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        log_dir = None if log_root is None else os.path.join(
            log_root, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + train_cfg.runner.run_name,
        )
        runner = MoECTSOnPolicyRunner(env, class_to_dict(train_cfg), log_dir, device=args.rl_device)
        if resume_path is not None:
            print("Loading model from: " + resume_path)
            runner.load(resume_path)
        elif init_path is not None:
            print("Initializing all-MoE comparison with fresh optimizers from: " + init_path)
            runner.initialize_from(init_path)
        return runner, train_cfg


task_registry = TaskRegistry()
