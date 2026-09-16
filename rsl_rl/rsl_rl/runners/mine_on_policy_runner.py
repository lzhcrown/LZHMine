"""On-policy runner for the Nezha gated-modal training pipeline."""

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.mine_ppo import MINEPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules.mine_actor_critic import MINEActorCritic


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
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
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
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
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
        final_iteration = first_iteration + num_learning_iterations
        for iteration in range(first_iteration, final_iteration):
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
                if iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{iteration}.pt",
                        )
                    )
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
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

        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, iteration)
        if self.wandb_run is not None:
            self.wandb_run.log(
                {"iteration": iteration, **metrics}, step=iteration
            )

        width = 92
        pad = 44
        completed_iterations = iteration - first_iteration + 1
        remaining_iterations = max(final_iteration - iteration - 1, 0)
        eta = (
            self.tot_time / max(completed_iterations, 1) * remaining_iterations
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
            (
                f"{'Mean action noise std:':>{pad}} "
                f"{self.alg.actor_critic.std.detach().mean().item():.2f}"
            ),
            f"{'Learning rate:':>{pad}} {self.alg.learning_rate:.6f}",
        ]
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
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "estimator_optimizer_state_dict": (
                    self.alg.actor_critic.estimator.optimizer.state_dict()
                ),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            estimator_state = checkpoint.get("estimator_optimizer_state_dict")
            if estimator_state is not None:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(
                    estimator_state
                )
        self.current_learning_iteration = checkpoint.get("iter", 0)
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
