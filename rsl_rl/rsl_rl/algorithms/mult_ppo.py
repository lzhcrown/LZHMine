import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms.him_ppo import HIMPPO

class MultPPO(HIMPPO):
    def __init__(self, actor_critic, expert1, expert2, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.expert1 = expert1
        self.expert2 = expert2
        
        if self.expert1 is not None:
            self.expert1.eval()
            for param in self.expert1.parameters():
                param.requires_grad = False
                
        if self.expert2 is not None:
            self.expert2.eval()
            for param in self.expert2.parameters():
                param.requires_grad = False

        self.distill_loss_coef = 10.0

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_distill_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                estimation_loss, swap_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch, lr=self.learning_rate)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Distillation loss with improved fallen state detection
                distill_loss = torch.tensor(0.0, device=self.device)
                if self.expert1 is not None and self.expert2 is not None:
                    with torch.no_grad():
                        expert1_action = self.expert1.act(obs_batch)
                        expert2_action = self.expert2.act(obs_batch)
                    
                    # Extract components from obs_batch for fallen state detection
                    # base_ang_vel is 3D (0:3), projected_gravity is 3D (3:6)
                    projected_gravity = obs_batch[:, 3:6]
                    proj_grav_z = projected_gravity[:, 2]
                    proj_grav_xy_norm = torch.norm(projected_gravity[:, :2], dim=1)
                    
                    # Default thresholds (these match config defaults)
                    proj_grav_z_threshold = -0.5
                    proj_grav_xy_threshold = 0.5
                    
                    # Multiple fallen state criteria:
                    # 1. Projected gravity z component (upside down)
                    criterion_z = proj_grav_z > proj_grav_z_threshold
                    # 2. Body tilt (projected gravity xy magnitude)
                    criterion_xy = proj_grav_xy_norm > proj_grav_xy_threshold
                    
                    # Combine criteria: fallen if ANY is true
                    is_fallen = (criterion_z | criterion_xy).float().unsqueeze(1)
                    
                    target_action = is_fallen * expert1_action + (1.0 - is_fallen) * expert2_action
                    distill_loss = nn.MSELoss()(mu_batch, target_action)

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.distill_loss_coef * distill_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss
                mean_distill_loss += distill_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        mean_distill_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_distill_loss
