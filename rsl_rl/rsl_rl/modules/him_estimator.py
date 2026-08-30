import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class HIMEstimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0):
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        
        # Check for NaN in raw encoder output
        if torch.isnan(vel).any() or torch.isnan(z).any():
            print(f"Warning: NaN in encoder output. vel: {torch.isnan(vel).any()}, z: {torch.isnan(z).any()}")
        
        # Safe normalize to prevent NaN when input is zero
        z_norm = torch.norm(z, p=2, dim=-1, keepdim=True)
        z_norm = torch.where(z_norm > 1e-8, z_norm, torch.ones_like(z_norm))
        z = z / z_norm
        
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        z_s = self.encode(obs_history)
        z_t = self.target(next_obs)
        # z_s is (vel, z) tuple, pred_vel is already extracted, z_s should be the latent
        pred_vel, z_s = z_s

        # Safe normalize
        z_s_norm = torch.norm(z_s, p=2, dim=-1, keepdim=True)
        z_s_norm = torch.where(z_s_norm > 1e-8, z_s_norm, torch.ones_like(z_s_norm))
        z_s = z_s / z_s_norm
        
        z_t_norm = torch.norm(z_t, p=2, dim=-1, keepdim=True)
        z_t_norm = torch.where(z_t_norm > 1e-8, z_t_norm, torch.ones_like(z_t_norm))
        z_t = z_t / z_t_norm
        
        # Check for NaN
        if torch.isnan(pred_vel).any() or torch.isnan(z_s).any() or torch.isnan(z_t).any():
            print(f"Warning: NaN in update. pred_vel: {torch.isnan(pred_vel).any()}, z_s: {torch.isnan(z_s).any()}, z_t: {torch.isnan(z_t).any()}")
            return 0.0, 0.0
        
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w_norm = torch.norm(w, p=2, dim=-1, keepdim=True)
            w_norm = torch.where(w_norm > 1e-8, w_norm, torch.ones_like(w_norm))
            w = w / w_norm
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        
        # Check for NaN in losses
        if torch.isnan(estimation_loss) or torch.isnan(swap_loss):
            print(f"Warning: NaN detected in estimator loss. Skipping this update.")
            return 0.0, 0.0
        
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    # Clamp out to prevent overflow
    out = torch.clamp(out, min=-100, max=100)
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True) + 1e-10
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-10
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None