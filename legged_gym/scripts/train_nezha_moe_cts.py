#!/usr/bin/env python
"""Train WF concurrent teacher/student MoE on Nezha in Isaac Gym."""

import isaacgym  # noqa: F401 - Gym must initialize before torch
from legged_gym.envs import *  # noqa: F401,F403 - registers the task
from legged_gym.utils import get_args, task_registry


def train(args):
    args.task = "nezha_moe_cts"
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, cfg = task_registry.make_alg_runner(env, name=args.task, args=args)
    print("Nezha MoE-CTS | current={} history={} privileged={} actions={}".format(
        env.num_one_step_obs, env.num_obs, env.num_privileged_obs, env.num_actions,
    ))
    print("Teacher envs={} | Student envs={} | Experts={} | Logs={}".format(
        runner.alg.teacher_env_idxs.numel(), runner.alg.student_env_idxs.numel(),
        runner.alg.actor_critic.expert_num, runner.log_dir,
    ))
    runner.learn(cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train(get_args())
