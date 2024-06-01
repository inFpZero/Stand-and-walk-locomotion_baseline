'''
from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.nn import functional as F

class Estimator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Estimator, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc21 = nn.Linear(64, latent_dim)  # Mean μ layer
        self.fc22 = nn.Linear(64, latent_dim)  # Log-variance log(σ²) layer
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 48)
        self.fc4 = nn.Linear(48, 48)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return h3

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 19))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
'''
import torch
from torch import nn
from torch.nn import functional as F
# from .types_ import *
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch.distributions import Normal

class Estimator(nn.Module):
    num_iter = 0 # Global static variable to keep track of iterations
    def __init__(self,
                num_vae_one_obs: int,
                num_history: int ,
                activation,
                adapt_hidden_dims: List = None,
                vae_latent_dims: List = None,
                beta: int = 4,
                **kwargs) -> None:
        super(Estimator, self).__init__()

        self.vae_latent_dims = vae_latent_dims
        self.beta = beta
        self.num_history = num_history
        
        # Build adapt
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_vae_one_obs * num_history, adapt_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(adapt_hidden_dims)-1):
            if l == len(adapt_hidden_dims) - 2:
                encoder_layers.append(nn.Linear(adapt_hidden_dims[l], adapt_hidden_dims[l + 1]))
            else:
                encoder_layers.append(nn.Linear(adapt_hidden_dims[l], adapt_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.adaptor = nn.Sequential(*encoder_layers)

        # Build encoder
        self.fc1 = nn.Linear(19, 128)
        self.fc21 = nn.Linear(128, 64)  # Mean μ layer
        self.fc22 = nn.Linear(128, 64)  # Log-variance log(σ²) layer
        # Decoder layers
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 41)

    def adapt(self, history_obs) :
        result = self.adaptor(history_obs)
        return result

    def encode(self, x):
        h1 = F.elu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.elu(self.fc3(z))
        return self.fc4(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, history_obs, **kwargs):
        latent = self.adapt(history_obs)
        z_mu, z_var = self.encode(latent)
        z = self.reparameterize(z_mu, z_var)
        pre = self.decode(z)
        return latent, z_mu, z_var, pre 
    
    def loss_function(self, history_obs, next_obs, real_v, kld_weight = 1.0) :
        latent, z_mu, z_var, pre  = self.forward(history_obs)
        recons_obs_loss =F.mse_loss(pre, next_obs)
        # print(pre[0], next_obs[0])
        # print(recons_obs_loss)
        est_loss = F.mse_loss(latent[:,:3], real_v)
        # print(est_loss)
        KL_loss = -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - z_var.exp())
        # print(KL_loss)
        VAE_loss = recons_obs_loss + self.beta * kld_weight * KL_loss

        CE_loss = est_loss + VAE_loss
        return CE_loss

    def sample(self, history_obs):
        estimation = self.adapt(history_obs)
        return estimation
