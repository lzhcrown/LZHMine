"""On-policy runner for the Nezha gated-modal training pipeline."""

import os
import copy
import json
import statistics
import sys
import time
from collections import deque
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.mine_ppo import MINEPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules.mine_actor_critic import MINEActorCritic
from rsl_rl.utils.mine_checkpoint import (
    config_to_dict,
    git_diff,
    git_state,
    policy_config_from_checkpoint,
)

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


class MINEOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.train_cfg = train_cfg
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        num_critic_obs = env.num_privileged_obs or env.num_obs

        if self.cfg["policy_class_name"] != "MINEActorCritic":
            raise ValueError("MINEOnPolicyRunner requires MINEActorCritic")
        if self.cfg["algorithm_class_name"] != "MINEPPO":
            raise ValueError("MINEOnPolicyRunner requires MINEPPO")

        actor_critic = MINEActorCritic(
            env.num_obs,
            num_critic_obs,
            env.num_one_step_obs,
            env.num_one_step_privileged_obs,
            list(env.cfg.env.estimation_target_indices),
            env.num_actions,
            **self.policy_cfg,
        ).to(device)
        self.alg = MINEPPO(actor_critic, device=device, **self.alg_cfg)
        self.checkpoint_policy_config = copy.deepcopy(self.policy_cfg)
        self.checkpoint_policy_config.update(
            num_actor_obs=env.num_obs,
            num_critic_obs=num_critic_obs,
            num_one_step_obs=env.num_one_step_obs,
            num_one_step_privileged_obs=env.num_one_step_privileged_obs,
            estimation_target_indices=list(env.cfg.env.estimation_target_indices),
            num_actions=env.num_actions,
            use_mode_gate=actor_critic.use_mode_gate,
            is_privileged_obs=actor_critic.estimator.is_privileged_obs,
            history_order=actor_critic.history_order,
        )
        # Provenance stored in every checkpoint: the code and configuration
        # the run was launched with.
        self.git_state = git_state(SOURCE_DIR)
        self.env_cfg_dict = config_to_dict(env.cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.best_policy_min_episodes = int(
            self.cfg.get("best_policy_min_episodes", 100)
        )
        self.best_mean_reward = -float("inf")
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [env.num_obs],
            [env.num_privileged_obs],
            [env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.wandb_run = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        # Reusing an environment after rollout also resets tensors allocated
        # by step() under inference_mode. Match the collection context.
        with torch.inference_mode():
            self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            self._write_run_info()
        if self.log_dir is not None and self.wandb_run is None:
            self._init_wandb()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        critic_obs = critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        current_rewards = torch.zeros(self.env.num_envs, device=self.device)
        current_lengths = torch.zeros(self.env.num_envs, device=self.device)

        first_iteration = self.current_learning_iteration
        self._learn_start_time = self.tot_time
        final_iteration = first_iteration + num_learning_iterations
        # All externally recorded iterations count completed PPO updates.
        for iteration in range(first_iteration + 1, final_iteration + 1):
            collection_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    step_result = self.env.step(actions)
                    (
                        obs,
                        privileged_obs,
                        rewards,
                        dones,
                        infos,
                        termination_ids,
                        termination_privileged_obs,
                    ) = step_result
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    next_critic_obs = critic_obs.clone()
                    if (
                        termination_ids is not None
                        and termination_privileged_obs is not None
                        and termination_ids.numel() > 0
                    ):
                        next_critic_obs[termination_ids.to(self.device)] = (
                            termination_privileged_obs.to(self.device)
                        )
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        current_rewards += rewards
                        current_lengths += 1
                        done_ids = dones.nonzero(as_tuple=False).flatten()
                        rewbuffer.extend(current_rewards[done_ids].cpu().tolist())
                        lenbuffer.extend(current_lengths[done_ids].cpu().tolist())
                        current_rewards[done_ids] = 0
                        current_lengths[done_ids] = 0

                collection_time = time.time() - collection_start
                self.alg.compute_returns(critic_obs)

            learning_start = time.time()
            losses = self.alg.update()
            self.current_learning_iteration = iteration
            learning_time = time.time() - learning_start
            if self.log_dir is not None:
                self._log(
                    iteration,
                    first_iteration,
                    final_iteration,
                    losses,
                    collection_time,
                    learning_time,
                    ep_infos,
                    rewbuffer,
                    lenbuffer,
                )
                self._save_best_policy(iteration, rewbuffer, lenbuffer)
                if iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{iteration}.pt",
                        )
                    )
            ep_infos.clear()

        self.current_learning_iteration = final_iteration
        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )
        if self.writer is not None:
            self.writer.flush()
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    def _write_run_info(self):
        """Record how the run was launched; uncommitted changes go to a patch."""
        run_info = {
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "command": sys.argv,
            "git": self.git_state,
            "train_cfg": self.train_cfg,
            "env_cfg": self.env_cfg_dict,
        }
        with open(os.path.join(self.log_dir, "run_info.json"), "w", encoding="utf-8") as stream:
            json.dump(run_info, stream, indent=2, default=str)
        if self.git_state.get("dirty"):
            with open(os.path.join(self.log_dir, "git_diff.patch"), "w", encoding="utf-8") as stream:
                stream.write(git_diff(SOURCE_DIR))

    def _save_best_policy(self, iteration, rewbuffer, lenbuffer):
        """Keep best_policy.pt at the highest mean reward over recent episodes.

        Judged only once rewbuffer holds best_policy_min_episodes episodes, so
        early high-variance means cannot claim it. Under the terrain
        curriculum the mean reward also depends on the assigned terrain, so
        evaluate best_policy.pt before deploying it.
        """
        if len(rewbuffer) < min(self.best_policy_min_episodes, rewbuffer.maxlen):
            return
        mean_reward = statistics.mean(rewbuffer)
        if mean_reward <= self.best_mean_reward:
            return
        self.best_mean_reward = mean_reward
        self.save(
            os.path.join(self.log_dir, "best_policy.pt"),
            infos={
                "iteration": iteration,
                "mean_reward": mean_reward,
                "mean_episode_length": statistics.mean(lenbuffer),
            },
        )

    def _init_wandb(self):
        if not self.cfg.get("wandb_enabled", False):
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but the 'wandb' package is not installed. "
                "Install it with: pip install wandb"
            ) from exc

        mode = os.environ.get(
            "WANDB_MODE", self.cfg.get("wandb_mode", "online")
        ).lower()
        if mode not in ("online", "offline", "disabled"):
            raise ValueError(
                "wandb_mode must be 'online', 'offline', or 'disabled', "
                f"got {mode!r}"
            )
        project = os.environ.get(
            "WANDB_PROJECT", self.cfg.get("wandb_project", "LZHMine")
        )
        entity = os.environ.get("WANDB_ENTITY") or self.cfg.get("wandb_entity")
        group = self.cfg.get("wandb_group") or None
        tags = self.cfg.get("wandb_tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        run_name = os.path.basename(os.path.normpath(self.log_dir))
        environment_cfg = {
            "num_envs": int(self.env.num_envs),
            "num_observations": int(self.env.num_obs),
            "num_privileged_observations": int(
                self.env.num_privileged_obs or 0
            ),
            "num_actions": int(self.env.num_actions),
            "control_dt": float(self.env.dt),
            "max_episode_length": int(self.env.max_episode_length),
        }
        try:
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                tags=list(tags),
                notes=self.cfg.get("wandb_notes") or None,
                job_type="train",
                dir=self.log_dir,
                mode=mode,
                config={
                    "training": self.train_cfg,
                    "environment": environment_cfg,
                    "git": self.git_state,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize W&B. On the training machine run "
                "'wandb login', or set WANDB_API_KEY. To train without W&B, "
                "pass --no_wandb."
            ) from exc
        print(
            f"W&B logging: project={project}, run={run_name}, mode={mode}, "
            f"url={getattr(self.wandb_run, 'url', None)}"
        )

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().mean().item()
        return float(value)

    def _episode_metrics(self, ep_infos):
        values = {}
        for episode_info in ep_infos:
            for name, value in episode_info.items():
                values.setdefault(name, []).append(self._scalar(value))
        return {
            f"Episode/{name}": statistics.mean(samples)
            for name, samples in values.items()
        }

    def _log(
        self,
        iteration,
        first_iteration,
        final_iteration,
        losses,
        collection_time,
        learning_time,
        ep_infos,
        rewbuffer,
        lenbuffer,
    ):
        steps = self.num_steps_per_env * self.env.num_envs
        iteration_time = collection_time + learning_time
        self.tot_timesteps += steps
        self.tot_time += iteration_time
        fps = int(steps / max(iteration_time, 1e-8))

        metrics = {}
        for name in ("value", "surrogate", "estimation", "swap", "mode", "semantic_mode"):
            metrics[f"Loss/{name}"] = self._scalar(losses[name])
        metrics["Policy/entropy"] = self._scalar(losses["entropy"])
        for name in (
            "semantic_valid_ratio",
            "wheel_target_ratio",
            "leg_target_ratio",
            "hybrid_target_ratio",
            "wheel_activity",
            "leg_activity",
        ):
            metrics[f"ModeSemantic/{name}"] = self._scalar(losses[name])
        for name in (
            "wheel_gate_probability",
            "leg_gate_probability",
            "hybrid_gate_probability",
        ):
            metrics[f"ModeGate/{name}"] = self._scalar(losses[name])
        for name in ("gate_entropy", "gate_batch_entropy", "mode_proto_max_cos"):
            if name in losses:
                metrics[f"ModeGate/{name}"] = self._scalar(losses[name])
        for name, value in losses.items():
            if name.startswith(("kl_", "pre_update_", "post_update_", "update_")):
                metrics[f"Diagnostics/{name}"] = self._scalar(value)
            elif name.startswith("learning_rate_"):
                metrics[f"LearningRate/{name[len('learning_rate_'):]}"] = self._scalar(value)
            elif name.startswith("latent_") or name in ("swap_vs_uniform", "swav_perplexity"):
                metrics[f"Latent/{name}"] = self._scalar(value)
        if not self.alg.actor_critic.use_mode_gate:
            metrics = {
                name: value for name, value in metrics.items()
                if not name.startswith(("ModeSemantic/", "ModeGate/"))
                and name not in ("Loss/mode", "Loss/semantic_mode")
            }
        metrics.update(
            {
                "Loss/learning_rate": float(self.alg.learning_rate),
                "Policy/mean_noise_std": (
                    self.alg.actor_critic.std.detach().mean().item()
                ),
                "Perf/total_fps": fps,
                "Perf/collection_time": collection_time,
                "Perf/learning_time": learning_time,
                "Perf/iteration_time": iteration_time,
                "Perf/total_timesteps": self.tot_timesteps,
                "Perf/total_time": self.tot_time,
            }
        )
        if rewbuffer:
            metrics["Train/mean_reward"] = statistics.mean(rewbuffer)
            metrics["Train/mean_episode_length"] = statistics.mean(lenbuffer)
        episode_metrics = self._episode_metrics(ep_infos)
        metrics.update(episode_metrics)

        # One append per iteration, with every mini-batch retained for audits.
        # Missing reset/non-reset groups have count=0 and no corresponding KL.
        record = {
            "iteration": iteration,
            "metrics": metrics,
            "minibatches": self.alg.last_minibatch_diagnostics,
        }
        with open(os.path.join(self.log_dir, "training_diagnostics.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, iteration)
        if self.wandb_run is not None:
            self.wandb_run.log(
                {"iteration": iteration, **metrics}, step=iteration
            )

        width = 92
        pad = 44
        completed_iterations = iteration - first_iteration
        remaining_iterations = max(final_iteration - iteration, 0)
        eta = (
            (self.tot_time - self._learn_start_time)
            / max(completed_iterations, 1) * remaining_iterations
        )
        title = f" Learning iteration {iteration}/{final_iteration} "
        log_lines = [
            "#" * width,
            f"\033[1m{title.center(width)}\033[0m",
            "",
            (
                f"{'Computation:':>{pad}} {fps} steps/s "
                f"(collection: {collection_time:.3f}s, "
                f"learning: {learning_time:.3f}s)"
            ),
            f"{'Value function loss:':>{pad}} {losses['value']:.4f}",
            f"{'Surrogate loss:':>{pad}} {losses['surrogate']:.4f}",
            f"{'Policy entropy:':>{pad}} {losses['entropy']:.4f}",
            f"{'Estimator loss:':>{pad}} {losses['estimation']:.4f}",
            f"{'Swap loss:':>{pad}} {losses['swap']:.4f}",
            (
                f"{'Mean action noise std:':>{pad}} "
                f"{self.alg.actor_critic.std.detach().mean().item():.2f}"
            ),
            f"{'Learning rate:':>{pad}} {self.alg.learning_rate:.6f}",
            (
                f"{'KL mean / max / first minibatch:':>{pad}} "
                f"{losses['kl_mean']:.4f} / {losses['kl_max']:.4f} / "
                f"{losses['kl_first']:.2e}"
            ),
            f"{'KL from estimator step:':>{pad}} {losses['kl_estimator_shift']:.4f}",
            (
                f"{'Pre-update KL / reset samples:':>{pad}} "
                f"{losses['pre_update_kl']:.2e} / {losses['pre_update_reset_count']}"
            ),
            f"{'Minibatch LR floor fraction:':>{pad}} {losses['learning_rate_floor_fraction']:.1%}",
            (
                f"{'Latent std / cos to mean:':>{pad}} "
                f"{losses['latent_std']:.4f} / {losses['latent_cos_to_mean']:.4f}"
            ),
            (
                f"{'Swap vs uniform / SwAV perplexity:':>{pad}} "
                f"{losses['swap_vs_uniform']:.3f} / {losses['swav_perplexity']:.1f}"
            ),
        ]
        if self.alg.actor_critic.use_mode_gate:
            log_lines.extend(
                [
                    (
                        f"{'Gate entropy (sample / batch):':>{pad}} "
                        f"{losses['gate_entropy']:.3f} / "
                        f"{losses['gate_batch_entropy']:.3f}"
                    ),
                    (
                        f"{'Mode prototype max |cos|:':>{pad}} "
                        f"{losses['mode_proto_max_cos']:.3f}"
                    ),
                    f"{'Mode loss:':>{pad}} {losses['mode']:.4f}",
                    f"{'Semantic mode loss:':>{pad}} {losses['semantic_mode']:.4f}",
                    (
                        f"{'Semantic valid ratio:':>{pad}} "
                        f"{losses['semantic_valid_ratio']:.4f}"
                    ),
                    (
                        f"{'Target ratio (wheel/leg/hybrid):':>{pad}} "
                        f"{losses['wheel_target_ratio']:.3f} / "
                        f"{losses['leg_target_ratio']:.3f} / "
                        f"{losses['hybrid_target_ratio']:.3f}"
                    ),
                    (
                        f"{'Gate probability (wheel/leg/hybrid):':>{pad}} "
                        f"{losses['wheel_gate_probability']:.3f} / "
                        f"{losses['leg_gate_probability']:.3f} / "
                        f"{losses['hybrid_gate_probability']:.3f}"
                    ),
                    (
                        f"{'Activity (wheel/leg):':>{pad}} "
                        f"{losses['wheel_activity']:.4f} / "
                        f"{losses['leg_activity']:.4f}"
                    ),
                ]
            )
        if rewbuffer:
            log_lines.extend(
                [
                    (
                        f"{'Mean reward:':>{pad}} "
                        f"{statistics.mean(rewbuffer):.4f}"
                    ),
                    (
                        f"{'Mean episode length:':>{pad}} "
                        f"{statistics.mean(lenbuffer):.2f}"
                    ),
                ]
            )
        log_lines.extend(
            f"{('Mean episode ' + name[len('Episode/'):] + ':'):>{pad}} "
            f"{value:.4f}"
            for name, value in episode_metrics.items()
        )
        log_lines.extend(
            [
                "-" * width,
                f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
                f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
                f"{'Total time:':>{pad}} {self.tot_time:.2f}s",
                f"{'ETA:':>{pad}} {eta:.1f}s",
            ]
        )
        print("\n".join(log_lines))

    def save(self, path, infos=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "policy_config": self.checkpoint_policy_config,
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "estimator_optimizer_state_dict": (
                    self.alg.actor_critic.estimator.optimizer.state_dict()
                ),
                "iter": self.current_learning_iteration,
                "iteration_semantics": "completed_updates",
                "tot_timesteps": self.tot_timesteps,
                "tot_time": self.tot_time,
                "best_mean_reward": self.best_mean_reward,
                "infos": infos,
                "train_cfg": self.train_cfg,
                "env_cfg": self.env_cfg_dict,
                "git": self.git_state,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = policy_config_from_checkpoint(checkpoint)
        if config["use_mode_gate"] != self.alg.actor_critic.use_mode_gate:
            raise ValueError("Gate mode differs from checkpoint; start a new run to change it")
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # The scheduler reads this attribute and overwrites optimizer lr
            # on its next step. Restore it from the optimizer, including for
            # legacy checkpoints which have no explicit learning-rate field.
            self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]["lr"])
            estimator_state = checkpoint.get("estimator_optimizer_state_dict")
            if estimator_state is not None:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(
                    estimator_state
                )
            estimator = self.alg.actor_critic.estimator
            estimator.learning_rate = float(estimator.optimizer.param_groups[0]["lr"])
        self.current_learning_iteration = checkpoint.get("iter", 0)
        self.best_mean_reward = checkpoint.get("best_mean_reward", -float("inf"))
        self.tot_timesteps = checkpoint.get("tot_timesteps", 0)
        self.tot_time = checkpoint.get("tot_time", 0.0)
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
