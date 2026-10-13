"""Gate-mode regression tests; run with python -m unittest legged_gym.tests.test_mine_modes."""

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from deploy.deploy_mujoco.runtime import NezhaPolicyRuntime
from legged_gym.scripts.export_nezha_mine import _build_policy, export
from rsl_rl.algorithms import MINEPPO
from rsl_rl.modules import MINEActorCritic, MINEPolicyExporter
from rsl_rl.runners.mine_on_policy_runner import MINEOnPolicyRunner
from rsl_rl.utils.mine_checkpoint import policy_config_from_checkpoint


class LegacyGatedExport(torch.nn.Module):
    """Old public interface, without the new architecture metadata methods."""

    def __init__(self, policy):
        super().__init__()
        self.exporter = MINEPolicyExporter(policy)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.exporter(observations)

    @torch.jit.export
    def get_mode_probabilities(self, observations: torch.Tensor) -> torch.Tensor:
        return self.exporter.get_mode_probabilities(observations)

    @torch.jit.export
    def get_history_order_code(self) -> int:
        return self.exporter.get_history_order_code()


class TestMINEModes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)

    def test_ungated_training_updates_source_target_and_actor_without_gate(self):
        policy = _build_policy("oldest_first", use_mode_gate=False,
                               mode_semantic_cfg={"enabled": True, "loss_coef": 1.0})
        estimator = policy.estimator
        self.assertFalse(estimator.mode_semantic_enabled)
        self.assertFalse(any("mode_gate" in name or "mode_prototypes" in name
                             for name, _ in policy.named_parameters()))
        obs = torch.randn(8, 130)
        parts = estimator.encoder(obs)
        expected_latent = F.normalize(parts[:, 4:], dim=-1)
        expected_actions = policy.actor(torch.cat((obs[:, -65:], parts[:, :4], expected_latent), -1))
        self.assertTrue(torch.allclose(policy.act_inference(obs), expected_actions, atol=1e-6))
        self.assertEqual(estimator.mode_probabilities(obs).shape, (8, 0))
        before = {name: value.clone() for name, value in policy.state_dict().items()}
        algorithm = MINEPPO(policy, num_learning_epochs=1, num_mini_batches=2)
        algorithm.init_storage(8, 2, [130], [1125], [16])
        privileged = torch.randn(8, 1125)
        for _ in range(2):
            with torch.inference_mode():
                algorithm.act(obs, privileged)
                privileged = torch.randn(8, 1125)
                algorithm.process_env_step(torch.randn(8), torch.zeros(8), {}, privileged)
            obs = torch.randn(8, 130)
        with torch.inference_mode():
            algorithm.compute_returns(privileged)
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        for name in ("mode", "semantic_mode", "wheel_gate_probability",
                     "leg_gate_probability", "hybrid_gate_probability"):
            self.assertEqual(losses[name], 0)
        for name in ("estimator.encoder.0.weight", "estimator.target.0.weight", "actor.0.weight"):
            self.assertFalse(torch.equal(before[name], policy.state_dict()[name]), name)

    def test_archived_checkpoints_export_in_both_modes_and_target_widths(self):
        with TemporaryDirectory() as directory:
            for gate in (True, False):
                for privileged in (True, False):
                    with self.subTest(gate=gate, privileged=privileged):
                        policy = _build_policy("oldest_first", use_mode_gate=gate,
                                               is_privileged_obs=privileged)
                        checkpoint = {"model_state_dict": policy.state_dict()}
                        config = policy_config_from_checkpoint(checkpoint)
                        self.assertEqual(config["use_mode_gate"], gate)
                        self.assertEqual(config["is_privileged_obs"], privileged)
                        path = Path(directory) / "checkpoint.pt"
                        output = Path(directory) / "policy.pt"
                        torch.save(checkpoint, path)
                        export(Namespace(checkpoint=str(path), output=str(output), history_order=None))
                        scripted = torch.jit.load(str(output))
                        obs = torch.randn(4, 130)
                        self.assertEqual(scripted.get_use_mode_gate(), gate)
                        self.assertEqual(scripted.get_mode_probabilities(obs).shape, (4, 3 if gate else 0))
                        self.assertTrue(torch.allclose(scripted(obs), policy.act_inference(obs), atol=1e-6))
                        if not gate:
                            self.assertFalse(any("mode_gate" in name or "mode_prototypes" in name
                                                 for name, _ in scripted.named_parameters()))

    def test_runtime_detects_modes_validates_config_and_supports_legacy_exports(self):
        with TemporaryDirectory() as directory:
            for gate, legacy in ((False, False), (True, False), (True, True)):
                with self.subTest(gate=gate, legacy=legacy):
                    policy = _build_policy("oldest_first", use_mode_gate=gate)
                    module = LegacyGatedExport(policy) if legacy else MINEPolicyExporter(policy)
                    path = str(Path(directory) / "policy.pt")
                    torch.jit.script(module).save(path)
                    kwargs = dict(policy_path=path, default_dof_pos=[0.] * 16,
                                  p_gains=[1.] * 16, d_gains=[1.] * 16)
                    runtime = NezhaPolicyRuntime(**kwargs)
                    self.assertEqual(runtime.use_mode_gate, gate)
                    self.assertEqual(runtime.observations.history_order, "oldest_first")
                    for _ in range(2):
                        actions, torques = runtime.step([0.] * 3, [0., 0., -1.],
                                                       [0.5, 0., 0.], [0.] * 16, [0.] * 16)
                        self.assertEqual(actions.shape, (16,))
                        self.assertTrue(torch.isfinite(torques).all())
                    self.assertEqual(runtime.mode_probabilities().numel(), 3 if gate else 0)
                    runtime.reset()
                    self.assertEqual(runtime.mode_probabilities().numel(), 3 if gate else 0)
                    with self.assertRaisesRegex(ValueError, "conflicts"):
                        NezhaPolicyRuntime(**kwargs, mode_gate="off" if gate else "on")
                    if not legacy:
                        with self.assertRaisesRegex(ValueError, "policy expects"):
                            NezhaPolicyRuntime(**kwargs, history_steps=3)

    def test_checkpoint_metadata_preserves_architecture_and_optimizer_resume(self):
        env = SimpleNamespace(
            num_envs=4, num_obs=130, num_privileged_obs=1125, num_one_step_obs=65,
            num_one_step_privileged_obs=375, num_actions=16, reset=lambda: None,
            cfg=SimpleNamespace(env=SimpleNamespace(estimation_target_indices=[371, 372, 373, 374])),
        )
        with TemporaryDirectory() as directory:
            for gate in (True, False):
                config = dict(
                    policy=dict(use_mode_gate=gate, history_order="newest_first",
                                actor_hidden_dims=[32], critic_hidden_dims=[32],
                                enc_hidden_dims=[32], tar_hidden_dims=[32], latent_dim=8),
                    algorithm={},
                    runner=dict(policy_class_name="MINEActorCritic", algorithm_class_name="MINEPPO",
                                num_steps_per_env=2, save_interval=1),
                )
                runner = MINEOnPolicyRunner(env, config)
                runner.alg.actor_critic.estimator.update(torch.randn(4, 130), torch.randn(4, 1125))
                runner.current_learning_iteration = 7
                path = str(Path(directory) / "checkpoint.pt")
                runner.save(path)
                saved = torch.load(path, weights_only=False)
                settings = policy_config_from_checkpoint(saved)
                self.assertEqual(settings["use_mode_gate"], gate)
                self.assertEqual(settings["latent_dim"], 8)
                restored = MINEActorCritic(**settings)
                restored.load_state_dict(saved["model_state_dict"])
                runner.load(path)
                self.assertEqual(runner.current_learning_iteration, 7)
                self.assertTrue(runner.alg.actor_critic.estimator.optimizer.state)
                output = str(Path(directory) / "policy.pt")
                export(Namespace(checkpoint=path, output=output, history_order=None))
                scripted = torch.jit.load(output)
                self.assertEqual(scripted.get_history_order_code(), 0)
                obs = torch.randn(4, 130)
                self.assertTrue(torch.allclose(scripted(obs), restored.act_inference(obs), atol=1e-6))
                config["policy"]["use_mode_gate"] = not gate
                other_runner = MINEOnPolicyRunner(env, config)
                with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
                    other_runner.load(path)

    def test_checkpoint_rejects_inconsistent_gate_metadata(self):
        policy = _build_policy("oldest_first", use_mode_gate=False)
        with self.assertRaisesRegex(ValueError, "metadata"):
            policy_config_from_checkpoint({"model_state_dict": policy.state_dict(),
                                           "policy_config": {"use_mode_gate": True}})


if __name__ == "__main__":
    unittest.main()
