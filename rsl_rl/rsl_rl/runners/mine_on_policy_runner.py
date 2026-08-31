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
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.best_mean_reward = -float("inf")
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
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
                    obs = torch.nan_to_num(obs.to(self.device))
                    critic_obs = torch.nan_to_num(critic_obs.to(self.device))
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
            self.current_learning_iteration = iteration + 1
            if self.log_dir is not None:
                self._log(
                    iteration,
                    losses,
                    collection_time,
                    learning_time,
                    ep_infos,
                    rewbuffer,
                    lenbuffer,
                )
                if self.current_learning_iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.current_learning_iteration}.pt",
                        )
                    )
                if rewbuffer and statistics.mean(rewbuffer) > self.best_mean_reward:
                    self.best_mean_reward = statistics.mean(rewbuffer)
                    self.save(os.path.join(self.log_dir, "best_policy.pt"))
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )

    def _log(
        self,
        iteration,
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

        if self.writer is not None:
            for name in ("value", "surrogate", "estimation", "swap", "mode", "semantic_mode"):
                self.writer.add_scalar(f"Loss/{name}", losses[name], iteration)
            for name in (
                "semantic_valid_ratio",
                "wheel_target_ratio",
                "leg_target_ratio",
                "hybrid_target_ratio",
                "wheel_activity",
                "leg_activity",
            ):
                self.writer.add_scalar(f"ModeSemantic/{name}", losses[name], iteration)
            for name in (
                "wheel_gate_probability",
                "leg_gate_probability",
                "hybrid_gate_probability",
            ):
                self.writer.add_scalar(f"ModeGate/{name}", losses[name], iteration)
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
            self.writer.add_scalar(
                "Policy/mean_noise_std",
                self.alg.actor_critic.std.mean().item(),
                iteration,
            )
            self.writer.add_scalar("Perf/total_fps", fps, iteration)
            if rewbuffer:
                self.writer.add_scalar(
                    "Train/mean_reward", statistics.mean(rewbuffer), iteration
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length", statistics.mean(lenbuffer), iteration
                )
            for episode_info in ep_infos:
                for name, value in episode_info.items():
                    if isinstance(value, torch.Tensor):
                        value = value.float().mean().item()
                    self.writer.add_scalar(f"Episode/{name}", value, iteration)

        reward_text = (
            f" reward={statistics.mean(rewbuffer):.2f}"
            if rewbuffer
            else ""
        )
        print(
            f"MINE iteration {iteration} | {fps} steps/s | "
            f"value={losses['value']:.4f} policy={losses['surrogate']:.4f} "
            f"estimate={losses['estimation']:.4f} swap={losses['swap']:.4f} "
            f"mode={losses['mode']:.4f} semantic={losses['semantic_mode']:.4f} "
            f"valid={losses['semantic_valid_ratio']:.3f} "
            f"target[w/l/h]=({losses['wheel_target_ratio']:.2f}/"
            f"{losses['leg_target_ratio']:.2f}/{losses['hybrid_target_ratio']:.2f}) "
            f"gate[w/l/h]=({losses['wheel_gate_probability']:.2f}/"
            f"{losses['leg_gate_probability']:.2f}/"
            f"{losses['hybrid_gate_probability']:.2f}){reward_text}"
        )

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
