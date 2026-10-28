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
from rsl_rl.utils.cts_checkpoint import DIMENSION_KEYS, load_cts_checkpoint, validate_checkpoint_actor

from legged_gym import LEGGED_GYM_ROOT_DIR
from experiments.registry import clone_config
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
        if getattr(train_cfg.runner, "experiment_key", None):
            validate_checkpoint_actor(
                checkpoint, train_cfg.policy.actor_type, train_cfg.policy.actor_expert_num,
            )

    def make_env(self, name, args=None, env_cfg=None):
        args = get_args() if args is None else args
        if name not in self.task_classes:
            raise ValueError("Task with name: {} was not registered".format(name))
        default_env_cfg, train_cfg = self.get_cfgs(name)
        if env_cfg is None:
            env_cfg = default_env_cfg
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        resume_path = self._resume_path(train_cfg)
        if resume_path is not None:
            checkpoint = load_cts_checkpoint(resume_path)
            self._validate_experiment_checkpoint(train_cfg, checkpoint)
            if "environment_config" in checkpoint:
                update_class_from_dict(env_cfg, checkpoint["environment_config"])
            # Restore the schema before simulator buffers are allocated.
            env_cfg.env.observation_config = copy.deepcopy(checkpoint["observation_config"])
            # Keep the resolved file even after a new run directory exists.
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
        resume_path = getattr(args, "cts_resume_path", None) or self._resume_path(train_cfg, log_root)
        if resume_path is not None:
            checkpoint = load_cts_checkpoint(resume_path)
            self._validate_experiment_checkpoint(train_cfg, checkpoint)
            # The loader makes legacy actor_type='mlp' explicit, so restoring
            # its config also overrides a newer task default of actor_type='moe'.
            for key, value in checkpoint["policy_config"].items():
                if key not in DIMENSION_KEYS:
                    setattr(train_cfg.policy, key, copy.deepcopy(value))
            update_class_from_dict(train_cfg.algorithm, checkpoint["train_config"]["algorithm"])
            saved_runner = checkpoint["train_config"]["runner"]
            train_cfg.runner.num_steps_per_env = saved_runner["num_steps_per_env"]
            if getattr(args, "run_name", None) is None:
                train_cfg.runner.run_name = saved_runner.get("run_name") or (
                    "cts_actor_moe" if train_cfg.policy.actor_type == "moe" else "cts_mlp_baseline"
                )
            actor_is_moe = train_cfg.policy.actor_type == "moe"
            default_tags = ["nezha", "moe-cts", "isaac-gym"] + (["actor-moe"] if actor_is_moe else [])
            train_cfg.runner.wandb_tags = copy.deepcopy(saved_runner.get("wandb_tags", default_tags))
            train_cfg.runner.wandb_notes = saved_runner.get("wandb_notes", (
                "CTS student MoE + MoE actor; MLP critic on Nezha" if actor_is_moe
                else "WF MoE concurrent teacher-student training with an MLP actor on Nezha"
            ))
        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        log_dir = None if log_root is None else os.path.join(
            log_root, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + train_cfg.runner.run_name,
        )
        runner = MoECTSOnPolicyRunner(env, class_to_dict(train_cfg), log_dir, device=args.rl_device)
        if resume_path is not None:
            print("Loading model from: " + resume_path)
            runner.load(resume_path)
        return runner, train_cfg


task_registry = TaskRegistry()
