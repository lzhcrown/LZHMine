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
from experiments.registry import RESIDUAL_EXPERIMENT_KEYS, get_experiment


def train(args, task_name="nezha_cts_g1_student_moe"):
    modes = [bool(getattr(args, key, False)) for key in ("init_from", "resume", "from_scratch")]
    if sum(modes) > 1:
        raise ValueError("Choose only one of --init_from, --resume or --from_scratch")
    comparison_tasks = {get_experiment(key).TASK_NAME for key in RESIDUAL_EXPERIMENT_KEYS}
    if task_name in comparison_tasks and not any(modes):
        raise ValueError("Residual comparison groups require --init_from G7_CHECKPOINT, "
                         "--resume, or an explicit --from_scratch")
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
    print("Teacher envs={} | Student envs={} | Student={} ({}) | Actor={} ({}) | Critic={} ({}) | Logs={}".format(
        runner.alg.teacher_env_idxs.numel(), runner.alg.student_env_idxs.numel(),
        runner.alg.actor_critic.student_encoder_type, runner.alg.actor_critic.expert_num,
        runner.alg.actor_critic.actor_type, runner.alg.actor_critic.actor_expert_num,
        runner.alg.actor_critic.critic_type, runner.alg.actor_critic.critic_expert_num, runner.log_dir,
    ))
    print("Action residual={} | Temporal coefficient={} | Start iteration={} | Additional iterations={}".format(
        runner.alg.actor_critic.residual_enabled, runner.alg.temporal_loss_coef,
        runner.current_learning_iteration, cfg.runner.max_iterations,
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
        {"name": "--init_from", "type": str,
         "help": "Initialize from a non-residual all-MoE training checkpoint; reset both optimizers and preserve curriculum progress."},
        {"name": "--from_scratch", "action": "store_true", "default": False,
         "help": "Explicitly train a comparison group without a pretrained starting point."},
    ])
    args.headless = args.headless or not args.viewer
    return args


if __name__ == "__main__":
    train(training_args())
