"""Training-pipeline regression tests (rollout integrity, diagnostics, checkpoints).

Run with: PYTHONPATH=.:rsl_rl pytest -q legged_gym/tests/test_mine_training_pipeline.py
(or python -m unittest legged_gym.tests.test_mine_training_pipeline).
"""

import json
import os
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from rsl_rl.modules import MINEEstimator
from rsl_rl.runners.mine_on_policy_runner import MINEOnPolicyRunner


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
            **runner_overrides,
        ),
    )
    return MINEOnPolicyRunner(env, config, log_dir=log_dir)


class TestRolloutIntegrity(unittest.TestCase):
    def _collect(self, env):
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


class TestEstimatorDiagnostics(unittest.TestCase):
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
