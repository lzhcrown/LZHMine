"""Regression checks for the single-factor Phase 1 estimator-order experiment.

Run: PYTHONPATH=.:rsl_rl OMP_NUM_THREADS=1 python -m unittest -v \
    legged_gym.tests.test_mine_update_schedule
"""

import copy
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from legged_gym.tests.test_mine_training_pipeline import InPlaceResetEnv, make_runner
from rsl_rl.algorithms.mine_ppo import MINEPPO
from rsl_rl.utils.mine_checkpoint import estimator_update_schedule_from_checkpoint


class TestMINEUpdateSchedule(unittest.TestCase):
    def _assert_parameters_equal(self, before, module, estimator):
        after = dict(module.named_parameters())
        for name, value in before.items():
            if name.startswith("estimator.") == estimator:
                self.assertTrue(torch.equal(value, after[name]), name)

    @staticmethod
    def _parameters(module):
        return {name: value.detach().clone() for name, value in module.named_parameters()}

    def _trace(self, schedule, use_mode_gate=True, iterations=2):
        runner = make_runner(InPlaceResetEnv(), use_mode_gate=use_mode_gate)
        alg = runner.alg
        alg.estimator_update_schedule = schedule
        alg.schedule = "adaptive"
        policy = alg.actor_critic
        steps_per_iteration = alg.num_learning_epochs * alg.num_mini_batches
        events, batches, ppo_rates, estimator_calls, updates = [], [], [], [], []
        expected_phases = []
        generator = alg.storage.mini_batch_generator
        ppo_step, estimator_update, algorithm_update = (
            alg.optimizer.step, policy.estimator.update, alg.update
        )

        def recording_generator(*args, **kwargs):
            for batch in generator(*args, **kwargs):
                batches.append((batch[0].clone(), batch[3].clone()))
                yield batch

        def recording_ppo_step(*args, **kwargs):
            before = self._parameters(policy)
            # The PPO optimizer owns estimator parameters for checkpoint
            # compatibility. Stale estimator gradients must not enter it,
            # including the first update after the previous delayed phase.
            self.assertTrue(all(p.grad is None for p in policy.estimator.parameters()))
            ppo_rates.append(alg.optimizer.param_groups[0]["lr"])
            events.append("ppo")
            result = ppo_step(*args, **kwargs)
            self._assert_parameters_equal(before, policy, estimator=True)
            return result

        def recording_estimator_update(obs, next_critic, learning_rate=None):
            ordinal = len(estimator_calls) % steps_per_iteration
            before = self._parameters(policy)
            with torch.no_grad():
                batch_mean = policy.act_inference(obs).clone()
                batch_sigma = policy.std.clone()
                if schedule == "after_ppo" and ordinal == 0:
                    rollout_obs = alg.storage.observations.flatten(0, 1)
                    expected_phases.append({
                        "mean": policy.act_inference(rollout_obs).clone(),
                        "sigma": policy.std.clone(),
                        "post_ppo": alg.rollout_diagnostics("post_ppo"),
                    })
            events.append("estimator")
            result = estimator_update(obs, next_critic, learning_rate=learning_rate)
            self._assert_parameters_equal(before, policy, estimator=False)
            with torch.no_grad():
                shift = ((policy.act_inference(obs) - batch_mean).square()
                         / (2 * batch_sigma.square())).sum(-1).mean().item()
                if schedule == "after_ppo" and ordinal == steps_per_iteration - 1:
                    phase = expected_phases[-1]
                    rollout_obs = alg.storage.observations.flatten(0, 1)
                    shift_samples = (
                        (policy.act_inference(rollout_obs) - phase["mean"]).square()
                        / (2 * phase["sigma"].square())
                    ).sum(-1)
                    phase["metrics"] = alg._grouped_kl(shift_samples, alg.storage.dones)
            estimator_calls.append((obs.clone(), next_critic.clone(), learning_rate, shift))
            return result

        def recording_algorithm_update():
            result = algorithm_update()
            updates.append((dict(result), copy.deepcopy(alg.last_minibatch_diagnostics)))
            return result

        alg.storage.mini_batch_generator = recording_generator
        alg.optimizer.step = recording_ppo_step
        policy.estimator.update = recording_estimator_update
        alg.update = recording_algorithm_update
        with patch("torch.randperm", wraps=torch.randperm) as randperm:
            runner.learn(iterations)
        # Delayed estimator updates replay the original indices, not a second
        # random permutation or another pass through the shuffled generator.
        self.assertEqual(randperm.call_count, iterations)
        count = steps_per_iteration * iterations
        self.assertEqual(len(batches), count)
        self.assertEqual(len(ppo_rates), count)
        self.assertEqual(len(estimator_calls), count)
        expected_events = (
            ["estimator", "ppo"] * steps_per_iteration
            if schedule == "interleaved"
            else ["ppo"] * steps_per_iteration + ["estimator"] * steps_per_iteration
        )
        self.assertEqual(events, expected_events * iterations)
        for batch, rate, call in zip(batches, ppo_rates, estimator_calls):
            self.assertTrue(torch.equal(batch[0], call[0]))
            self.assertTrue(torch.equal(batch[1], call[1]))
            self.assertEqual(rate, call[2])
        for iteration, (metrics, rows) in enumerate(updates):
            self.assertEqual(metrics["update_ppo_count"], steps_per_iteration)
            self.assertEqual(metrics["update_estimator_count"], steps_per_iteration)
            for j, row in enumerate(rows):
                call = estimator_calls[iteration * steps_per_iteration + j]
                self.assertAlmostEqual(row["kl_estimator_shift"], call[3], places=8)
        return updates, expected_phases

    def test_constructor_accepts_only_explicit_supported_schedules(self):
        policy = make_runner(InPlaceResetEnv()).alg.actor_critic
        self.assertEqual(MINEPPO(policy).estimator_update_schedule, "interleaved")
        for schedule in ("interleaved", "after_ppo"):
            self.assertEqual(
                MINEPPO(policy, estimator_update_schedule=schedule).estimator_update_schedule,
                schedule,
            )
        for schedule in ("after", "", None):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                MINEPPO(policy, estimator_update_schedule=schedule)

    def test_interleaved_order_and_parameter_isolation_across_rollouts(self):
        for gated in (False, True):
            with self.subTest(use_mode_gate=gated):
                self._trace("interleaved", use_mode_gate=gated)

    def test_delayed_order_replays_identical_inputs_and_rates_without_reshuffle(self):
        for gated in (False, True):
            with self.subTest(use_mode_gate=gated):
                self._trace("after_ppo", use_mode_gate=gated)

    def test_delayed_phase_kl_is_measured_separately_for_both_reset_groups(self):
        updates, expected_phases = self._trace("after_ppo")
        for (metrics, _), expected in zip(updates, expected_phases):
            for name, value in expected["post_ppo"].items():
                self.assertAlmostEqual(metrics[name], value, places=7)
            for name, value in expected["metrics"].items():
                self.assertAlmostEqual(metrics["estimator_phase_" + name], value, places=8)
            self.assertEqual(metrics["estimator_phase_reset_count"], 16)
            self.assertEqual(metrics["estimator_phase_non_reset_count"], 32)
            self.assertGreater(metrics["estimator_phase_kl"], 0.)
            self.assertLess(metrics["pre_update_kl"], 1e-3)

    def test_noop_delayed_estimator_has_exactly_zero_phase_kl(self):
        runner = make_runner(InPlaceResetEnv())
        runner.alg.estimator_update_schedule = "after_ppo"
        runner.alg.actor_critic.estimator.update = lambda *args, **kwargs: (0.,) * 13
        result = {}
        update = runner.alg.update

        def capture():
            result.update(update())
            return result

        runner.alg.update = capture
        runner.learn(1)
        for name in ("estimator_phase_kl", "estimator_phase_kl_reset",
                     "estimator_phase_kl_non_reset", "kl_estimator_shift"):
            self.assertEqual(result[name], 0., name)
        # The legacy rollout KL has a small positive numerical offset; the
        # new estimator-only metric must not inherit it.
        self.assertGreater(result["pre_update_kl"], 0.)

    def test_default_and_explicit_interleaved_have_identical_state_and_rng(self):
        results = []
        for explicit in (False, True):
            runner = make_runner(InPlaceResetEnv())
            if explicit:
                runner.alg.estimator_update_schedule = "interleaved"
            runner.learn(2)
            results.append((
                copy.deepcopy(runner.alg.actor_critic.state_dict()),
                copy.deepcopy(runner.alg.optimizer.state_dict()),
                copy.deepcopy(runner.alg.actor_critic.estimator.optimizer.state_dict()),
                torch.get_rng_state().clone(),
            ))

        def assert_equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_equal(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    assert_equal(a, b)
            else:
                self.assertEqual(left, right)

        assert_equal(*results)

    def test_fixed_rate_estimator_state_is_identical_between_schedules(self):
        states = []
        random_states = []
        for schedule in ("interleaved", "after_ppo"):
            runner = make_runner(InPlaceResetEnv())
            runner.alg.estimator_update_schedule = schedule
            self.assertEqual(runner.alg.schedule, "fixed")
            # This environment intentionally ignores actions, so both runs
            # have identical rollouts. Holding LR fixed isolates ordering
            # from the adaptive scheduler's response to policy KL.
            runner.learn(2)
            states.append(copy.deepcopy(runner.alg.actor_critic.estimator.state_dict()))
            random_states.append(torch.get_rng_state().clone())
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)
        self.assertTrue(torch.equal(*random_states))

    def test_checkpoint_records_schedule_and_rejects_cross_schedule_load(self):
        template = make_runner(InPlaceResetEnv())

        def runner_for(schedule):
            config = copy.deepcopy(template.train_cfg)
            config["algorithm"]["estimator_update_schedule"] = schedule
            return type(template)(InPlaceResetEnv(), config, log_dir=None)

        with TemporaryDirectory() as directory:
            for schedule in ("interleaved", "after_ppo"):
                with self.subTest(schedule=schedule):
                    source = runner_for(schedule)
                    source.learn(1)
                    path = os.path.join(directory, schedule + ".pt")
                    source.save(path)
                    saved = torch.load(path, weights_only=False)
                    self.assertEqual(estimator_update_schedule_from_checkpoint(saved), schedule)
                    resumed = runner_for(schedule)
                    resumed.load(path)
                    self.assertEqual(resumed.current_learning_iteration, 1)
                    for name, value in source.alg.actor_critic.state_dict().items():
                        self.assertTrue(torch.equal(value, resumed.alg.actor_critic.state_dict()[name]))
                    other = "after_ppo" if schedule == "interleaved" else "interleaved"
                    mismatched = runner_for(other)
                    before = self._parameters(mismatched.alg.actor_critic)
                    for load_optimizer in (False, True):
                        with self.assertRaisesRegex(ValueError, "schedule differs"):
                            mismatched.load(path, load_optimizer=load_optimizer)
                        self._assert_parameters_equal(before, mismatched.alg.actor_critic, True)
                        self._assert_parameters_equal(before, mismatched.alg.actor_critic, False)
                    if schedule == "interleaved":
                        del saved["train_cfg"]["algorithm"]["estimator_update_schedule"]
                        legacy_path = os.path.join(directory, "legacy_A.pt")
                        torch.save(saved, legacy_path)
                        legacy_resumed = runner_for("interleaved")
                        legacy_resumed.load(legacy_path)
                        self.assertEqual(legacy_resumed.current_learning_iteration, 1)
                        for name, value in source.alg.actor_critic.state_dict().items():
                            self.assertTrue(torch.equal(
                                value, legacy_resumed.alg.actor_critic.state_dict()[name]
                            ))
                        with self.assertRaisesRegex(ValueError, "schedule differs"):
                            runner_for("after_ppo").load(legacy_path)

    def test_legacy_checkpoint_defaults_to_interleaved_and_invalid_value_fails(self):
        self.assertEqual(estimator_update_schedule_from_checkpoint({}), "interleaved")
        self.assertEqual(
            estimator_update_schedule_from_checkpoint({"train_cfg": {"algorithm": {}}}),
            "interleaved",
        )
        with self.assertRaises(ValueError):
            estimator_update_schedule_from_checkpoint({
                "train_cfg": {"algorithm": {"estimator_update_schedule": "invalid"}}
            })


if __name__ == "__main__":
    unittest.main()
