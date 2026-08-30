import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import ActorCritic, get_activation

class MultActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, **kwargs)
        
        # We can load experts later in the algorithm
