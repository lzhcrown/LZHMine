#!/usr/bin/env python
"""Real Isaac Gym stage-0 smoke test; not a locomotion-quality experiment.

Uses 16 environments, shortened episodes and a small terrain grid to force
resets. Runs two updates, loads their checkpoint into a new runner, then runs
one more update. Both actor/critic stored inputs are checked exactly.
"""

import copy
import json
import math
from pathlib import Path

from legged_gym.envs import *  # noqa: F401,F403; Isaac Gym before torch
from legged_gym.utils import get_args, task_registry
from rsl_rl.runners.mine_on_policy_runner import MINEOnPolicyRunner
import torch


def attach_checks(runner):
    acted, results = [], []
    act, update = runner.alg.act, runner.alg.update

    def recorded_act(obs, critic):
        acted.append((obs.clone(), critic.clone()))
        return act(obs, critic)

    def checked_update():
        storage = runner.alg.storage
        for step, (obs, critic) in enumerate(acted):
            assert torch.equal(storage.observations[step], obs), "Actor observation changed"
            assert torch.equal(storage.privileged_observations[step], critic), "Critic observation changed"
        assert storage.dones.any(), "Smoke rollout did not exercise resets"
        assert (~storage.dones.bool()).any(), "Smoke rollout lacks non-reset samples"
        losses = update()
        assert all(math.isfinite(value) for value in losses.values()), "Nonfinite diagnostics"
        for group in ("", "_reset", "_non_reset"):
            assert losses["pre_update_kl" + group] < 1e-3, losses
        assert losses["pre_update_value_max_abs_error"] < 1e-4, losses
        results.append(losses)
        acted.clear()
        return losses

    runner.alg.act, runner.alg.update = recorded_act, checked_update
    return results


def main():
    args = get_args()
    if args.resume:
        raise ValueError("This smoke test always starts fresh and tests its own resume")
    args.headless, args.no_wandb = True, True
    args.num_envs = args.num_envs or 16
    args.run_name = args.run_name or "phase0_smoke"
    args.experiment_name = "phase0_smoke"
    args.max_iterations = 2
    env_cfg, train_cfg = task_registry.get_cfgs("nezha_mine")
    # All overrides are local to this process and recorded in the checkpoint.
    env_cfg.env.episode_length_s = 0.2
    env_cfg.terrain.num_rows, env_cfg.terrain.num_cols = 2, 5
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.border_size = 5
    train_cfg.runner.save_interval = 1
    train_cfg.runner.best_policy_min_episodes = 1
    train_cfg.algorithm.learning_rate = 1e-5
    train_cfg.policy.estimator_learning_rate = 1e-5
    env, _ = task_registry.make_env("nezha_mine", args=args, env_cfg=env_cfg)
    runner = None
    resumed = None
    try:
        runner, _ = task_registry.make_alg_runner(env, name="nezha_mine", args=args, train_cfg=train_cfg)
        # Play steps outside inference_mode; initialization must support that
        # path too, even though training collects rollouts in inference_mode.
        with torch.no_grad():
            env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        results = attach_checks(runner)
        runner.learn(2)
        directory = Path(runner.log_dir)
        checkpoint = directory / "model_2.pt"
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        assert saved["iter"] == 2
        best = torch.load(str(directory / "best_policy.pt"), map_location="cpu", weights_only=False)
        assert best["iter"] == best["infos"]["iteration"]
        for key in ("git", "train_cfg", "env_cfg"):
            assert key in saved

        resumed = MINEOnPolicyRunner(
            env, copy.deepcopy(runner.train_cfg), str(directory / "resume"), device=args.rl_device
        )
        resumed.load(str(checkpoint))
        rate = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
        estimator_rate = saved["estimator_optimizer_state_dict"]["param_groups"][0]["lr"]
        assert resumed.alg.learning_rate == rate
        assert resumed.alg.actor_critic.estimator.learning_rate == estimator_rate
        resume_results = attach_checks(resumed)
        resumed.learn(1)
        assert resumed.current_learning_iteration == 3
        expected = min(1e-2, rate * 1.5)
        assert math.isclose(resumed.alg.last_minibatch_diagnostics[0]["learning_rate"], expected)
        report = {
            "status": "PASS",
            "gate": runner.alg.actor_critic.use_mode_gate,
            "checkpoint": str(checkpoint),
            "resumed_checkpoint": str(directory / "resume" / "model_3.pt"),
            "restored_learning_rate": rate,
            "reset_samples": [row["pre_update_reset_count"] for row in results + resume_results],
            "pre_update_kl": [row["pre_update_kl"] for row in results + resume_results],
            "observation_equality": "exact, actor and critic, every transition",
        }
        (directory / "stage0_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        for item in (runner, resumed):
            if item is not None and item.writer is not None:
                item.writer.close()
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
