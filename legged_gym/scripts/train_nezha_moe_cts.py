#!/usr/bin/env python
"""Train WF concurrent teacher/student MoE on Nezha in Isaac Gym."""

if __package__:
    from ._bootstrap import configure_runtime
else:
    from _bootstrap import configure_runtime

configure_runtime(isaac_gym=True)

import isaacgym  # noqa: F401 - Gym must initialize before torch
from legged_gym.envs import *  # noqa: F401,F403 - registers the task
from legged_gym.utils import get_args, task_registry


def train(args, task_name="nezha_moe_cts"):
    if args.num_envs is not None and args.num_envs < 2:
        raise ValueError("CTS training requires at least two environments")
    if args.max_iterations is not None and args.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    args.task = task_name
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    runner, cfg = task_registry.make_alg_runner(env, name=args.task, args=args)
    print("Experiment={} | Task={}".format(getattr(cfg.runner, "experiment_key", "legacy"), args.task))
    print("Nezha MoE-CTS | current={} history={} privileged={} actions={}".format(
        env.num_one_step_obs, env.num_obs, env.num_privileged_obs, env.num_actions,
    ))
    print("Teacher envs={} | Student envs={} | Student experts={} | Actor={} | Actor experts={} | Logs={}".format(
        runner.alg.teacher_env_idxs.numel(), runner.alg.student_env_idxs.numel(),
        runner.alg.actor_critic.expert_num, runner.alg.actor_critic.actor_type,
        runner.alg.actor_critic.actor_expert_num, runner.log_dir,
    ))
    print("Terrain={} | Curriculum={} | Speed ranges={} | Episode={} s | TensorBoard=enabled".format(
        env_cfg.terrain.generator_profile,
        getattr(env_cfg.commands, "curriculum_profile", "legacy")
        if env.wf_curriculum is not None else "legacy",
        env.wf_curriculum.range_curriculum if env.wf_curriculum is not None else "nezha_reward",
        env_cfg.env.episode_length_s,
    ))
    runner.learn(cfg.runner.max_iterations, init_at_random_ep_len=True)


def training_args():
    args = get_args([
        {"name": "--viewer", "action": "store_true",
         "help": "Show the training viewer (training is headless by default)."},
    ])
    args.headless = args.headless or not args.viewer
    return args


if __name__ == "__main__":
    train(training_args())
