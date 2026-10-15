"""Training-pipeline regression tests (rollout integrity, diagnostics, checkpoints).

Run with: PYTHONPATH=.:rsl_rl pytest -q legged_gym/tests/test_mine_training_pipeline.py
(or python -m unittest legged_gym.tests.test_mine_training_pipeline).
"""

import json
import os
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from rsl_rl.modules import MINEEstimator
from rsl_rl.runners.mine_on_policy_runner import MINEOnPolicyRunner
from legged_gym.scripts.summarize_training_diagnostics import summarize


class InPlaceResetEnv:
    """Minimal VecEnv with NezhaMINEEnv's history layout.

    With ``in_place_reset=True`` it clears the histories of reset envs inside
    the tensors it returned on the previous step, as NezhaMINEEnv.reset_idx
    did before the fix. The runner must still store the observation each
    action was actually computed from.
    """

    def __init__(self, num_envs=8, episode_length=3, in_place_reset=True):
        self.num_envs = num_envs
        self.num_one_step_obs = 10
        self.num_one_step_privileged_obs = 12
        self.num_obs = 2 * self.num_one_step_obs
        self.num_privileged_obs = 2 * self.num_one_step_privileged_obs
        self.num_actions = 4
        self.max_episode_length = episode_length
        self.in_place_reset = in_place_reset
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(estimation_target_indices=[10, 11])
        )
        self.generator = torch.Generator().manual_seed(0)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.obs_buf = torch.randn(num_envs, self.num_obs, generator=self.generator)
        self.privileged_obs_buf = torch.randn(
            num_envs, self.num_privileged_obs, generator=self.generator
        )
        self.num_resets = 0

    def reset(self):
        pass

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def step(self, actions):
        del actions
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        env_ids = dones.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.privileged_obs_buf[env_ids].clone()
        if env_ids.numel() > 0:
            self.num_resets += env_ids.numel()
            self.episode_length_buf[env_ids] = 0
            if self.in_place_reset:
                self.obs_buf[env_ids] = 0.0
                self.privileged_obs_buf[env_ids] = 0.0
            else:
                self.obs_buf = self.obs_buf.index_fill(0, env_ids, 0.0)
                self.privileged_obs_buf = self.privileged_obs_buf.index_fill(
                    0, env_ids, 0.0
                )
        frame = torch.randn(
            self.num_envs, self.num_one_step_obs, generator=self.generator
        )
        privileged_frame = torch.cat(
            (frame, torch.randn(self.num_envs, 2, generator=self.generator)), dim=-1
        )
        self.obs_buf = torch.cat(
            (self.obs_buf[:, self.num_one_step_obs :], frame), dim=-1
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.privileged_obs_buf[:, self.num_one_step_privileged_obs :],
                privileged_frame,
            ),
            dim=-1,
        )
        rewards = torch.rand(self.num_envs, generator=self.generator)
        infos = {"time_outs": dones.clone()}
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            rewards,
            dones,
            infos,
            env_ids,
            termination_privileged_obs,
        )


def make_runner(env, use_mode_gate=True, log_dir=None, **runner_overrides):
    torch.manual_seed(0)
    config = dict(
        policy=dict(
            use_mode_gate=use_mode_gate,
            actor_hidden_dims=[32],
            critic_hidden_dims=[32],
            enc_hidden_dims=[32],
            tar_hidden_dims=[32],
            latent_dim=6,
        ),
        algorithm=dict(num_learning_epochs=2, num_mini_batches=2),
        runner=dict(
            policy_class_name="MINEActorCritic",
            algorithm_class_name="MINEPPO",
            num_steps_per_env=6,
            save_interval=1000,
        ),
    )
    config["runner"].update(runner_overrides)
    return MINEOnPolicyRunner(env, config, log_dir=log_dir)


class TestRolloutIntegrity(unittest.TestCase):
    def _collect(self, env, expect_resets=True):
        runner = make_runner(env)
        acted, stored, results = [], {}, {}
        act, update = runner.alg.act, runner.alg.update

        def recording_act(obs, critic_obs):
            acted.append((obs.clone(), critic_obs.clone()))
            return act(obs, critic_obs)

        def snapshot_update():
            storage = runner.alg.storage
            stored["obs"] = storage.observations.clone()
            stored["critic"] = storage.privileged_observations.clone()
            results["losses"] = update()
            return results["losses"]

        runner.alg.act = recording_act
        runner.alg.update = snapshot_update
        runner.learn(1)
        if expect_resets:
            self.assertGreater(env.num_resets, 0, "the rollout must contain resets")
        return acted, stored, results["losses"]

    def test_storage_keeps_the_observations_each_action_used(self):
        acted, stored, _ = self._collect(InPlaceResetEnv(in_place_reset=True))
        for step, (obs, critic_obs) in enumerate(acted):
            self.assertTrue(torch.equal(stored["obs"][step], obs), f"obs step {step}")
            self.assertTrue(
                torch.equal(stored["critic"][step], critic_obs), f"critic step {step}"
            )

    def test_first_minibatch_kl_is_zero_for_a_consistent_rollout(self):
        # Before any parameter update the re-evaluated policy must reproduce
        # the stored action distribution exactly, resets included.
        for in_place in (True, False):
            with self.subTest(in_place_reset=in_place):
                _, _, losses = self._collect(InPlaceResetEnv(in_place_reset=in_place))
                self.assertLess(losses["kl_first"], 1e-3)
                for name in ("kl_mean", "kl_max", "kl_estimator_shift"):
                    self.assertTrue(torch.isfinite(torch.tensor(losses[name])), name)
                self.assertGreaterEqual(losses["kl_max"], losses["kl_mean"])

    def test_whole_rollout_kl_covers_both_groups(self):
        _, _, losses = self._collect(InPlaceResetEnv())
        self.assertEqual(losses["pre_update_reset_count"], 16)
        self.assertEqual(losses["pre_update_non_reset_count"], 32)
        for name in ("pre_update_kl", "pre_update_kl_reset", "pre_update_kl_non_reset"):
            self.assertLess(losses[name], 1e-3)
        self.assertLess(losses["pre_update_value_max_abs_error"], 1e-6)
        self.assertEqual(losses["update_reset_count"], 32)  # two PPO epochs
        self.assertEqual(losses["update_non_reset_count"], 64)
        self.assertIn("post_update_kl_reset", losses)

    def test_absent_groups_are_omitted_not_nan_or_zero(self):
        for length, absent in ((100, "reset"), (1, "non_reset")):
            _, _, losses = self._collect(InPlaceResetEnv(episode_length=length), False)
            self.assertEqual(losses["pre_update_" + absent + "_count"], 0)
            self.assertNotIn("pre_update_kl_" + absent, losses)
            self.assertNotIn("kl_" + absent, losses)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))

    def test_audit_detects_reset_corruption_without_consuming_rng(self):
        runner = make_runner(InPlaceResetEnv())
        def audit_instead_of_update():
            state = torch.get_rng_state().clone()
            before = runner.alg.rollout_diagnostics("pre", True, chunk_size=7)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            storage = runner.alg.storage
            reset = storage.dones.squeeze(-1).bool()
            storage.mu[reset] += 3.0
            after = runner.alg.rollout_diagnostics("pre", True, chunk_size=7)
            self.assertGreater(after["pre_kl_reset"], 1.0)
            self.assertAlmostEqual(before["pre_kl_non_reset"], after["pre_kl_non_reset"])
            storage.clear()
            return {}
        runner.alg.update = audit_instead_of_update
        runner.learn(1)

    def test_shuffle_keeps_reset_labels_aligned(self):
        runner = make_runner(InPlaceResetEnv())
        storage = runner.alg.storage
        ids = torch.arange(48).reshape(6, 8)
        storage.observations[:, :, 0] = ids
        storage.dones[:, :, 0] = ids % 3 == 0
        batches = list(storage.mini_batch_generator(2, 2, include_dones=True))
        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertTrue(torch.equal(batch[0][:, 0].long() % 3 == 0, batch[-1].flatten().bool()))
        self.assertEqual(len(next(storage.mini_batch_generator(2))), 10)


class TestEstimatorDiagnostics(unittest.TestCase):
    def test_per_dimension_and_median_statistics(self):
        latent = torch.tensor([[1., 0., 0.], [1., 2., 0.], [1., 0., 3.]])
        metrics = MINEEstimator.latent_statistics(latent)
        for i, value in enumerate(latent.std(dim=0, unbiased=False)):
            self.assertAlmostEqual(metrics[f"latent_std_dim_{i:02d}"], value.item())
        self.assertEqual(metrics["latent_std_min"], 0.)
        direction = torch.nn.functional.normalize(latent.mean(0), dim=0)
        self.assertAlmostEqual(metrics["latent_cos_to_mean_median"], (latent @ direction).median().item())
        single = MINEEstimator.latent_statistics(latent[:1])
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in single.values()))

    def _estimator(self, use_mode_gate):
        torch.manual_seed(0)
        return MINEEstimator(
            temporal_steps=2,
            num_one_step_obs=10,
            num_one_step_privileged_obs=12,
            estimation_target_indices=[10, 11],
            use_mode_gate=use_mode_gate,
            enc_hidden_dims=[32],
            tar_hidden_dims=[32],
            latent_dim=6,
        )

    def test_latent_health_separates_collapsed_from_varied_latents(self):
        estimator = self._estimator(use_mode_gate=False)
        estimator.update(torch.randn(256, 20), torch.randn(256, 24))
        varied = dict(estimator.last_diagnostics)
        constant = torch.randn(1, 20).expand(256, 20).contiguous()
        estimator.update(constant, torch.randn(256, 24))
        collapsed = dict(estimator.last_diagnostics)

        self.assertGreater(varied["latent_std"], 10 * collapsed["latent_std"])
        self.assertLess(varied["latent_cos_to_mean"], 0.99)
        self.assertGreater(collapsed["latent_cos_to_mean"], 0.999)
        self.assertGreaterEqual(collapsed["swav_perplexity"], 1.0)
        self.assertAlmostEqual(collapsed["swav_perplexity"], 1.0, places=4)
        self.assertLessEqual(varied["swav_perplexity"], 32.0)
        self.assertGreater(varied["swap_vs_uniform"], 0.0)
        self.assertNotIn("gate_entropy", varied)

    def test_gate_diagnostics_are_reported_only_for_gated_estimators(self):
        estimator = self._estimator(use_mode_gate=True)
        estimator.update(torch.randn(64, 20), torch.randn(64, 24))
        diagnostics = estimator.last_diagnostics
        log3 = torch.log(torch.tensor(3.0)).item()
        self.assertTrue(0.0 <= diagnostics["gate_entropy"] <= log3 + 1e-6)
        self.assertTrue(
            diagnostics["gate_entropy"] <= diagnostics["gate_batch_entropy"] + 1e-6
        )
        self.assertTrue(0.0 <= diagnostics["mode_proto_max_cos"] <= 1.0 + 1e-6)
        with torch.no_grad():
            estimator.mode_prototypes.copy_(
                torch.tensor([[1.0], [2.0], [-0.5]]) * torch.ones(3, 6)
            )
        estimator.update(torch.randn(64, 20), torch.randn(64, 24))
        self.assertGreater(estimator.last_diagnostics["mode_proto_max_cos"], 0.99)


class TestCheckpoints(unittest.TestCase):
    def test_provenance_handles_ascii_locale(self):
        # The actual GPU launcher uses a C locale; Chinese README diffs must
        # remain readable and writable without depending on Python UTF-8 mode.
        code = (
            "from tempfile import TemporaryDirectory; "
            "from unittest.mock import patch; "
            "from legged_gym.tests.test_mine_training_pipeline import make_runner, InPlaceResetEnv; "
            "from rsl_rl.utils.mine_checkpoint import git_diff; "
            "git_diff('.'); "
            "directory = TemporaryDirectory(); "
            "runner = make_runner(InPlaceResetEnv(), log_dir=directory.name); "
            "runner.git_state['dirty'] = True; "
            "mock = patch('rsl_rl.runners.mine_on_policy_runner.git_diff', return_value='\\u4e2d\\u6587'); "
            "mock.start(); runner._write_run_info(); mock.stop(); directory.cleanup()"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True,
            env=dict(os.environ, LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0"),
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))

    def test_resume_restores_scheduler_and_estimator_learning_rates(self):
        with TemporaryDirectory() as directory:
            runner = make_runner(InPlaceResetEnv())
            runner.alg.learning_rate = 1e-5
            runner.alg.optimizer.param_groups[0]["lr"] = 1e-5
            estimator = runner.alg.actor_critic.estimator
            estimator.learning_rate = 2e-5
            estimator.optimizer.param_groups[0]["lr"] = 2e-5
            runner.current_learning_iteration = 7
            path = os.path.join(directory, "model_7.pt")
            runner.save(path)
            restored = make_runner(InPlaceResetEnv())
            restored.load(path)
            self.assertEqual(restored.alg.learning_rate, 1e-5)
            self.assertEqual(restored.alg.actor_critic.estimator.learning_rate, 2e-5)
            restored.alg.schedule = "adaptive"
            restored.alg.desired_kl = 0.01
            restored.learn(1)
            self.assertEqual(restored.current_learning_iteration, 8)
            # A consistent first mini-batch raises 1e-5 by 1.5, not 1e-3.
            self.assertAlmostEqual(restored.alg.last_minibatch_diagnostics[0]["learning_rate"], 1.5e-5)
            fresh = make_runner(InPlaceResetEnv())
            fresh.load(path, load_optimizer=False)
            self.assertEqual(fresh.alg.learning_rate, 1e-3)
            self.assertEqual(fresh.alg.optimizer.param_groups[0]["lr"], 1e-3)

    def test_iteration_numbers_logs_and_minibatch_metrics_agree(self):
        with TemporaryDirectory() as directory:
            runner = make_runner(InPlaceResetEnv(), log_dir=directory,
                                 save_interval=2, best_policy_min_episodes=1)
            # Exercise floor fractions without changing the optimization schedule.
            runner.alg.learning_rate = 1e-5
            runner.alg.optimizer.param_groups[0]["lr"] = 1e-5
            runner.learn(3)
            self.assertFalse(os.path.exists(os.path.join(directory, "model_0.pt")))
            for iteration in (2, 3):
                saved = torch.load(os.path.join(directory, f"model_{iteration}.pt"), weights_only=False)
                self.assertEqual(saved["iter"], iteration)
            best = torch.load(os.path.join(directory, "best_policy.pt"), weights_only=False)
            self.assertEqual(best["iter"], best["infos"]["iteration"])
            with open(os.path.join(directory, "training_diagnostics.jsonl")) as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual([row["iteration"] for row in records], [1, 2, 3])
            for record in records:
                rows, metrics = record["minibatches"], record["metrics"]
                self.assertEqual(len(rows), 4)
                self.assertEqual(metrics["LearningRate/floor_fraction"], 1.)
                self.assertEqual(sum(row["reset_count"] for row in rows), 32)
                weighted = sum(row.get("kl_reset", 0.) * row["reset_count"] for row in rows) / 32
                self.assertAlmostEqual(metrics["Diagnostics/kl_reset"], weighted)
                self.assertIn("Latent/latent_std_dim_05", metrics)
            diagnostic_file = os.path.join(directory, "training_diagnostics.jsonl")
            summary = summarize(diagnostic_file, start=2, end=3)
            self.assertTrue(summary["integrity_pass"])
            self.assertEqual(summary["lr_floor_iteration_fraction"], 1.)
            self.assertEqual(summary["lr_floor_minibatch_fraction"], 1.)
            self.assertFalse(summary["lr_target_pass"])
            with self.assertRaisesRegex(ValueError, "complete"):
                summarize(diagnostic_file, start=2, end=4)
            resumed = make_runner(InPlaceResetEnv(), log_dir=directory, save_interval=2)
            resumed.load(os.path.join(directory, "model_3.pt"))
            resumed.learn(1)
            with open(os.path.join(directory, "training_diagnostics.jsonl")) as stream:
                self.assertEqual([json.loads(line)["iteration"] for line in stream], [1, 2, 3, 4])
            final = torch.load(os.path.join(directory, "model_4.pt"), weights_only=False)
            self.assertEqual(final["tot_timesteps"], 4 * 48)

    def test_best_policy_and_provenance_are_saved(self):
        with TemporaryDirectory() as directory:
            env = InPlaceResetEnv(in_place_reset=False)
            runner = make_runner(env, log_dir=directory, best_policy_min_episodes=16)
            runner.learn(3)
            files = set(os.listdir(directory))
            self.assertIn("best_policy.pt", files)
            self.assertIn("model_3.pt", files)
            self.assertIn("run_info.json", files)

            best = torch.load(
                os.path.join(directory, "best_policy.pt"), weights_only=False
            )
            self.assertIn("mean_reward", best["infos"])
            self.assertIn("iteration", best["infos"])
            final = torch.load(os.path.join(directory, "model_3.pt"), weights_only=False)
            self.assertEqual(final["iter"], 3)
            for checkpoint in (best, final):
                self.assertEqual(checkpoint["train_cfg"]["runner"]["num_steps_per_env"], 6)
                self.assertEqual(
                    checkpoint["env_cfg"]["env"]["estimation_target_indices"], [10, 11]
                )
                self.assertIn("commit", checkpoint["git"])
            with open(os.path.join(directory, "run_info.json")) as stream:
                run_info = json.load(stream)
            self.assertEqual(run_info["git"], final["git"])
            if final["git"].get("dirty"):
                self.assertIn("git_diff.patch", files)


if __name__ == "__main__":
    unittest.main()
