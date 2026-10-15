"""Isaac Gym runner for concurrent teacher/student PPO and MoE distillation."""

import copy
import json
import os
import statistics
import time
from collections import deque

import torch

from rsl_rl.algorithms.moe_cts import MoECTS
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.utils.cts_checkpoint import ARCHITECTURE, FORMAT_VERSION, load_cts_checkpoint


def _config_dict(value):
    """Serialize Gym config objects without importing Isaac Gym into the runner."""
    if isinstance(value, dict):
        return {key: _config_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_dict(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _config_dict(getattr(value, key)) for key in dir(value)
                if not key.startswith("_") and not callable(getattr(value, key))}
    return value


class MoECTSOnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.device = device
        self.train_cfg = copy.deepcopy(train_cfg)
        self.cfg = self.train_cfg["runner"]
        self.log_dir = log_dir
        self.observation_config = copy.deepcopy(env.observation_config)
        self.policy_config = copy.deepcopy(self.train_cfg["policy"])
        self.policy_config.update(
            num_actor_obs=env.num_obs,
            num_critic_obs=env.num_privileged_obs,
            num_one_step_obs=env.num_one_step_obs,
            num_actions=env.num_actions,
            history_length=int(self.observation_config["history_length"]),
            observation_dims=[term["dim"] for term in self.observation_config["proprioceptive_terms"]],
        )
        if self.cfg.get("policy_class_name", "ActorCriticMoECTS") != "ActorCriticMoECTS":
            raise ValueError("MoECTSOnPolicyRunner requires ActorCriticMoECTS")
        if self.cfg.get("algorithm_class_name", "MoECTS") != "MoECTS":
            raise ValueError("MoECTSOnPolicyRunner requires MoECTS")
        policy = ActorCriticMoECTS(**self.policy_config).to(device)
        policy.observation_config = copy.deepcopy(self.observation_config)
        self.alg = MoECTS(policy, num_envs=env.num_envs, device=device, **self.train_cfg["algorithm"])
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg.get("save_interval", 50))
        if self.num_steps_per_env < 1 or self.save_interval < 1:
            raise ValueError("Rollout length and save interval must be positive")
        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env,
            [env.num_obs], [env.num_privileged_obs], [env.num_actions],
        )
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.writer = None
        self.wandb_run = None
        self.env.reset()

    def _open_logs(self):
        if self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)
            with open(os.path.join(self.log_dir, "config.json"), "w", encoding="utf-8") as stream:
                json.dump({
                    "training": self.train_cfg,
                    "observations": self.observation_config,
                    "policy": self.policy_config,
                }, stream, ensure_ascii=False, indent=2)
        if self.cfg.get("wandb_enabled", False) and self.wandb_run is None:
            import wandb
            self.wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", self.cfg.get("wandb_project", "NezhaMoECTS")),
                entity=os.environ.get("WANDB_ENTITY") or self.cfg.get("wandb_entity"),
                name=os.path.basename(os.path.normpath(self.log_dir)),
                group=self.cfg.get("wandb_group", "nezha_moe_cts"),
                tags=self.cfg.get("wandb_tags", ["nezha", "moe-cts", "isaac-gym"]),
                dir=self.log_dir,
                mode=os.environ.get("WANDB_MODE", self.cfg.get("wandb_mode", "online")),
                config={"training": self.train_cfg, "observations": self.observation_config},
            )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        count = int(num_learning_iterations)
        if count < 1:
            raise ValueError("num_learning_iterations must be positive")
        self._open_logs()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length),
            )
        obs = self.env.get_observations().to(self.device)
        critic_obs = self.env.get_privileged_observations().to(self.device)
        self.alg.actor_critic.train()
        rewards_by_env = torch.zeros(self.env.num_envs, device=self.device)
        lengths_by_env = torch.zeros(self.env.num_envs, device=self.device)
        is_teacher = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        is_teacher[self.alg.teacher_env_idxs] = True
        buffers = {name: deque(maxlen=100) for name in (
            "teacher_reward", "student_reward", "teacher_length", "student_length",
        )}
        final_iteration = self.current_learning_iteration + count
        for iteration in range(self.current_learning_iteration, final_iteration):
            start = time.perf_counter()
            episodes = []
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    result = self.env.step(actions)
                    obs, critic_obs, rewards, dones, infos = result[:5]
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    # CTS distils the stored same-time observation. Terminal
                    # privileged observations (Gym's two extra return fields)
                    # must never replace the next episode's current observation.
                    self.alg.process_env_step(rewards, dones, infos)
                    rewards_by_env += rewards
                    lengths_by_env += 1
                    done_mask = dones.bool().reshape(-1)
                    for name, group in (("teacher", is_teacher), ("student", ~is_teacher)):
                        selected = done_mask & group
                        buffers[name + "_reward"].extend(rewards_by_env[selected].cpu().tolist())
                        buffers[name + "_length"].extend(lengths_by_env[selected].cpu().tolist())
                    rewards_by_env[done_mask] = 0
                    lengths_by_env[done_mask] = 0
                    if "episode" in infos:
                        episodes.append(infos["episode"])
                self.alg.compute_returns(obs, critic_obs)
            collected = time.perf_counter() - start
            start = time.perf_counter()
            losses = self.alg.update()
            learned = time.perf_counter() - start
            self.current_learning_iteration = iteration + 1
            self.tot_timesteps += self.env.num_envs * self.num_steps_per_env
            self.tot_time += collected + learned
            self._log(iteration, losses, collected, learned, buffers, episodes)
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        if self.writer is not None:
            self.writer.flush()
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    def _log(self, iteration, losses, collected, learned, buffers, episodes):
        metrics = {}
        for key, value in losses.items():
            prefix = "Experts" if key.startswith(("expert_", "gate_")) else "Loss"
            metrics[prefix + "/" + key] = float(value)
        for name, values in buffers.items():
            if values:
                metrics["Train/" + name] = statistics.mean(values)
        episode_values = {}
        for episode in episodes:
            for name, value in episode.items():
                scalar = value.detach().float().mean().item() if isinstance(value, torch.Tensor) else float(value)
                episode_values.setdefault(name, []).append(scalar)
        for name, values in episode_values.items():
            metrics["Episode/" + name] = statistics.mean(values)
        metrics.update({
            "Perf/fps": self.num_steps_per_env * self.env.num_envs / max(collected + learned, 1e-8),
            "Perf/collection_seconds": collected,
            "Perf/learning_seconds": learned,
            "Perf/total_timesteps": self.tot_timesteps,
            "Policy/learning_rate": self.alg.optimizer.param_groups[0]["lr"],
            "Policy/student_learning_rate": self.alg.optimizer_stu_enc.param_groups[0]["lr"],
        })
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, iteration)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=iteration)
        print("MoE-CTS iteration {} | {:.0f} steps/s | {}".format(
            iteration + 1, metrics["Perf/fps"],
            " | ".join("{}={:.5f}".format(k, float(v)) for k, v in losses.items() if not k.startswith("expert_")),
        ))

    def _control_config(self):
        env = self.env
        if not hasattr(env, "cfg") or not hasattr(env.cfg, "control"):
            return {}
        result = {
            "action_scale": env.cfg.control.action_scale,
            "wheel_velocity_scale": env.cfg.control.vel_scale,
            "control_dt": env.dt,
            "clip_actions": env.cfg.normalization.clip_actions,
        }
        for key in ("default_dof_pos", "p_gains", "d_gains", "torque_limits", "wheel_indices"):
            if hasattr(env, key):
                result[key] = getattr(env, key).detach().cpu().reshape(-1).tolist()
        if hasattr(env, "dof_names"):
            result["joint_names"] = list(env.dof_names)
        return result

    def save(self, path, infos=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "architecture": ARCHITECTURE,
            "format_version": FORMAT_VERSION,
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "policy_config": self.policy_config,
            "observation_config": self.observation_config,
            "train_config": self.train_cfg,
            "environment_config": _config_dict(self.env.cfg) if hasattr(self.env, "cfg") else {},
            "control_config": self._control_config(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "student_optimizer_state_dict": self.alg.optimizer_stu_enc.state_dict(),
            "iter": self.current_learning_iteration,
            "next_iter": self.current_learning_iteration,
            "total_timesteps": self.tot_timesteps,
            "total_time": self.tot_time,
            "env_common_step_counter": getattr(self.env, "common_step_counter", 0),
            "infos": infos,
        }
        # Atomic replacement prevents a partial checkpoint being treated as latest.
        temporary = str(path) + ".tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def load(self, path, load_optimizer=True):
        checkpoint = load_cts_checkpoint(path, map_location=self.device)
        if checkpoint["policy_config"] != self.policy_config:
            raise ValueError("Policy configuration differs from checkpoint; restore it before constructing the runner")
        if checkpoint["observation_config"] != self.observation_config:
            raise ValueError("Observation configuration differs from checkpoint")
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.optimizer_stu_enc.load_state_dict(checkpoint["student_optimizer_state_dict"])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = int(checkpoint.get("next_iter", checkpoint["iter"]))
        self.tot_timesteps = int(checkpoint.get("total_timesteps", 0))
        self.tot_time = float(checkpoint.get("total_time", 0.0))
        if hasattr(self.env, "common_step_counter"):
            self.env.common_step_counter = int(checkpoint.get("env_common_step_counter", 0))
        # Physics and per-environment histories are intentionally reinitialised.
        self.env.reset()
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        policy = self.alg.actor_critic.eval()
        if device is not None:
            policy.to(device)
        return policy.act_inference
