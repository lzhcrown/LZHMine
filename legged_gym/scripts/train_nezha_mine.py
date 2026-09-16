#!/usr/bin/env python
"""Dedicated trainer for the Nezha gated-modal dual-encoder policy.

Network, loss and rollout construction remain in rsl_rl; this entry point
deliberately locks the task and validates the complete MINE training stack.

Usage:
    python legged_gym/scripts/train_nezha_mine.py --headless
    python legged_gym/scripts/train_nezha_mine.py --resume \
        --load_run <run> --checkpoint -1 --headless
"""

from legged_gym.envs import *  # noqa: F401,F403 - registers nezha_mine
from legged_gym.utils import get_args, task_registry


TASK_NAME = "nezha_mine"
EXPECTED_RUNNER = "MINEOnPolicyRunner"
EXPECTED_POLICY = "MINEActorCritic"
EXPECTED_ALGORITHM = "MINEPPO"


def train(args):
    """Build the Nezha environment and start synchronized PPO/MINE training."""
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)

    if train_cfg.runner_class_name != EXPECTED_RUNNER:
        raise RuntimeError(
            f"{TASK_NAME} must use {EXPECTED_RUNNER}, got "
            f"{train_cfg.runner_class_name}"
        )
    if train_cfg.runner.policy_class_name != EXPECTED_POLICY:
        raise RuntimeError(
            f"{TASK_NAME} must use {EXPECTED_POLICY}, got "
            f"{train_cfg.runner.policy_class_name}"
        )
    if train_cfg.runner.algorithm_class_name != EXPECTED_ALGORITHM:
        raise RuntimeError(
            f"{TASK_NAME} must use {EXPECTED_ALGORITHM}, got "
            f"{train_cfg.runner.algorithm_class_name}"
        )

    env, env_cfg = task_registry.make_env(
        name=TASK_NAME,
        args=args,
        env_cfg=env_cfg,
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
    )

    print("=" * 72)
    print("Nezha gated-modal dual-encoder training")
    print(f"task:               {TASK_NAME}")
    print(f"runner/algorithm:   {EXPECTED_RUNNER} / {EXPECTED_ALGORITHM}")
    print(f"environments:       {env.num_envs}")
    print(f"actor observations: {env.num_obs} ({env.history_length} x {env.num_one_step_obs})")
    print(f"critic observations:{env.num_privileged_obs}")
    target_width = (
        env.num_one_step_privileged_obs
        if train_cfg.policy.is_privileged_obs
        else env.num_one_step_obs
    )
    print(f"target encoder obs: {target_width}")
    print(
        "semantic gate loss: "
        f"{train_cfg.policy.mode_semantic_cfg['enabled']}"
    )
    print(f"actions:            {env.num_actions}")
    print(f"device:             {args.rl_device}")
    print(f"log directory:      {runner.log_dir}")
    print(
        "wandb:              "
        f"{train_cfg.runner.wandb_enabled} / "
        f"{train_cfg.runner.wandb_project} / "
        f"{train_cfg.runner.wandb_mode}"
    )
    print("=" * 72)

    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
