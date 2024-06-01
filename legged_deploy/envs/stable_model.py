import torch.nn as nn
import torch
import os


class VelocityPredictor(nn.Module):
    def __init__(self, num_obs_history, num_privileged_obs):
        super().__init__()
                # Adaptation module
        self.net = nn.Sequential(
            nn.Linear(num_obs_history, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, num_privileged_obs)
        )
       
    def forward(self, x):
        return self.net(x)
    
class ForwardModel(nn.Module):
    def __init__(self, num_obs, num_actions, pi):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_actions),
        )
        #self.net.to(self.device)

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def array2tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def forward(self, x):
        return self.net(x)

class Model_1(nn.Module):
    def __init__(self, num_obs, num_actions, sigma=1, pi=11e-3):
        super().__init__()
        # std of the noise to add
        self.sigma = sigma
        # score function scaled by sigma (reparametrization)
        self.net = nn.Sequential(
            nn.Linear(num_obs + num_actions, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, num_actions),
        )

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def array2tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def forward(self, x):
        return self.net(x)

class Model_2(nn.Module):
    def __init__(self, num_obs, pi, sigma=1):
        super().__init__()
        # std of the noise to add
        self.sigma = sigma
        # score function scaled by sigma (reparametrization)
        self.net = nn.Sequential(
            nn.Linear(num_obs, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, num_obs),
        )

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def array2tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def forward(self, x):
        return self.net(x)
