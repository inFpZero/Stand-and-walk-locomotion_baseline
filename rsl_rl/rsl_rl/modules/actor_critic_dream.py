# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .vae import Estimator

class ActorCriticDream(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_obs_history,
                        num_dim_depth = None,
                        adapt_hidden_dims=[128,64,19],
                        estimate_hidden_dims = [64,128,47],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        bootstrap_threshold=0.1,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDream, self).__init__()

        activation = get_activation(activation)
        mlp_input_dim_a = num_actor_obs            
        mlp_input_dim_c = num_critic_obs            
        mlp_input_dim_d = num_obs_history * mlp_input_dim_a
        
        
            
        # parameter for AdaBoot
        self.bootstrap_threshold = bootstrap_threshold
        '''
        # Adaptation module
        adapt_layers = []
        adapt_layers.append(nn.Linear(mlp_input_dim_d, adapt_hidden_dims[0]))
        adapt_layers.append(activation)
        for l in range(len(adapt_hidden_dims)-1):
            if l == len(adapt_hidden_dims)-2 :
                adapt_layers.append(nn.Linear(adapt_hidden_dims[l], adapt_hidden_dims[l + 1]))
            else:
                adapt_layers.append(nn.Linear(adapt_hidden_dims[l], adapt_hidden_dims[l + 1]))
                adapt_layers.append(activation)
        self.adapt = nn.Sequential(*adapt_layers)

        #self.estimator = VAE(19, 128)
        '''
        # estimator
        self.estimator = Estimator(
            num_vae_one_obs = mlp_input_dim_a,
            num_history = num_obs_history,
            activation= activation,
            adapt_hidden_dims = adapt_hidden_dims,
            vae_latent_dims = estimate_hidden_dims,
            beta = 1,
        )
        # Policy
        actor_layers = []
        if num_dim_depth is not None:
            mlp_input_dim_a = num_actor_obs + num_dim_depth
        actor_layers.append(nn.Linear(mlp_input_dim_a+19, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c + mlp_input_dim_a, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Estimator Module: {self.estimator}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, obs_history):
        latent = self.estimator.sample(obs_history)
        mean = self.actor(torch.cat((observations, latent), dim=-1))
        self.distribution = Normal(mean, mean*0. + self.std)

    def adapt_bootstrap_probability(self, rewards):
        cv = torch.std(rewards) / torch.mean(rewards)
        bootstrap_prob = 1 - torch.tanh(cv)
        return bootstrap_prob.item()
    
    def act(self, observations, obs_history,**kwargs):
        self.update_distribution(observations, obs_history)
        '''
        bootstrap_prob = self.adapt_bootstrap_probability(rew_buf)
        if bootstrap_prob > self.bootstrap_threshold:
            return self.distribution.sample()
        else:
            return self.distribution.mean
        '''
        return self.distribution.sample()
        
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        latent = self.estimator.sample(obs_history)
        actions_mean = self.actor(torch.cat((observations, latent), dim=-1))
        return actions_mean

    def evaluate(self, obs, critic_observations, **kwargs):
        value = self.critic(torch.cat((obs, critic_observations), dim=-1))
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
