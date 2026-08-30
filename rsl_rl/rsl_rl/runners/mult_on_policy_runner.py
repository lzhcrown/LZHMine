import time
import os
from collections import deque
import statistics

import torch
from rsl_rl.algorithms.mult_ppo import MultPPO
from rsl_rl.modules.mult_actor_critic import MultActorCritic
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

class MultOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        
        # Load experts if distillation is enabled
        self.distill = train_cfg.get("distill", False)
        if self.distill:
            expert1_path = train_cfg.get("expert1_path", "")
            expert2_path = train_cfg.get("expert2_path", "")
            
            # Initialize experts
            from rsl_rl.modules.him_actor_critic import HIMActorCritic
            num_actor_obs = self.env.num_obs
            num_critic_obs = self.env.num_privileged_obs
            num_actions = self.env.num_actions
            
            self.expert1 = HIMActorCritic(num_actor_obs, num_critic_obs, self.env.num_one_step_obs, num_actions, **self.policy_cfg).to(self.device)
            self.expert2 = HIMActorCritic(num_actor_obs, num_critic_obs, self.env.num_one_step_obs, num_actions, **self.policy_cfg).to(self.device)
            
            if os.path.exists(expert1_path):
                self.expert1.load_state_dict(torch.load(expert1_path, map_location=self.device))
                print(f"Loaded expert 1 from {expert1_path}")
            else:
                print(f"Warning: Expert 1 path {expert1_path} does not exist.")
                
            if os.path.exists(expert2_path):
                self.expert2.load_state_dict(torch.load(expert2_path, map_location=self.device))
                print(f"Loaded expert 2 from {expert2_path}")
            else:
                print(f"Warning: Expert 2 path {expert2_path} does not exist.")
                
            # Re-initialize algorithm with experts
            self.alg = MultPPO(self.alg.actor_critic, self.expert1, self.expert2, device=self.device, **self.cfg_train["algorithm"])
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Similar to OnPolicyRunner.learn, but we can log distill loss
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Track best reward for model checkpointing
        best_mean_reward = -float('inf')

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)
                    
                    # Use critic_obs as next_critic_obs (already updated by env including resets)
                    self.alg.process_env_step(rewards, dones, infos, critic_obs)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            if hasattr(self.alg, 'expert1'):
                mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_distill_loss = self.alg.update()
            else:
                mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
                mean_distill_loss = 0.0
                
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                
                # Save checkpoint every 200 iterations
                if (it - self.current_learning_iteration + 1) % 200 == 0:
                    self.save(os.path.join(self.log_dir, f'checkpoint_{it}.pt'))
                
                # Save best model based on mean reward
                if len(rewbuffer) > 0:
                    current_mean_reward = statistics.mean(rewbuffer)
                    if current_mean_reward > best_mean_reward:
                        best_mean_reward = current_mean_reward
                        self.save(os.path.join(self.log_dir, 'best_policy.pt'))
                
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        if hasattr(self.alg, 'expert1'):
            print(f" \033[1m Distill Loss: {locs['mean_distill_loss']:.4f} \033[0m")
