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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot


class Wow(LeggedRobot):


    def check_termination(self):
        """ Check if environments need to be reset
            termination condition for h1 only
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        self.reset_orientation = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) >0.65
        self.reset_buf |= self.reset_orientation

        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.reset_base = self.base_height < 0.63
        self.reset_buf |= self.reset_base

    # def get_target_angles(self):
    #     self.target_angles = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    #     for i in range(self.num_dof):
    #         name = self.dof_names[i]
    #         angle = self.cfg.init_state.target_angles[name]
    #         self.target_angles[i] = angle
    #     self.target_angles = self.target_angles.unsqueeze(0)


    def _reward_xy_velocity_stand(self):
        # Penalize velocity when standing still
        lin_vel_error = torch.sum(torch.abs(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error*5)*(torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_xy_velocity_walk(self):
        # Reward for tracking velocity when walking
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error*5)*(torch.norm(self.commands[:, :3], dim=1) > 0.1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.abs(self.root_states[:, 2].unsqueeze(1) - self.measured_heights)
    #     return torch.exp(-20*torch.abs(base_height - self.cfg.rewards.base_height_target))


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.exp(-20*torch.abs(base_height - self.cfg.rewards.base_height_target))

    # def _reward_feet_air_time_stand(self):
    #     # Reward 1 when standing
    #     return 1*(torch.norm(self.commands[:, :2], dim=1) < 0.1)
    # def _reward_feet_air_time_walk(self):
    #     # Reward long steps when walking
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     self.contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * self.contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~self.contact_filt
    #     return rew_airTime*(torch.norm(self.commands[:, :2], dim=1) > 0.1)
    
    # def _reward_feet_contact_stand(self):
    #     # Reward 1 when standing
    #     return 1*(torch.norm(self.commands[:, :2], dim=1) < 0.1)
    # def _reward_feet_contact_walk(self):
    #     # Reward one feet contact when walking beyond 0.3s
    #     one_feet_contact_filt= torch.logical_xor(self.contact_filt[:,0],self.contact_filt[:,1])
    #     fly_time_filt=torch.abs(self.feet_air_time[:,0]-self.feet_air_time[:,1])>0.3
    #     return 1*one_feet_contact_filt*fly_time_filt

    def _reward_foot_position_stand(self):
        # Penalize feet position away from target when standing
        # self.get_target_angles()
        return -1*torch.sum(torch.abs(self.dof_pos-self.default_dof_pos),dim=1) \
            *(torch.norm(self.commands[:, :3], dim=1) < 0.08)
    
    # def _reward_foot_position_walk(self):
    #     return 1*(torch.norm(self.commands[:, :2], dim=1) >0.1)


    # def _reward_arm_position(self):
    #     # Penalize arm pose away from target
    #     self.get_target_angles()
    #     return torch.exp(-3*torch.sum(torch.square(self.dof_pos[:,10:]-self.target_angles[:,10:]),dim=1))
    
    # def _reward_arm_position_stand(self):
    #     # Penalize arm pose away from target
    #     self.get_target_angles()
    #     return torch.exp(-3*torch.sum(torch.square(self.dof_pos[:,10:]-self.target_angles[:,10:]),dim=1))*(torch.norm(self.commands[:, :2], dim=1) < 0.08)


    def _reward_base_acc(self):
        # Penalize base accelerations
        return torch.exp(-0.01*torch.sum(torch.abs((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=1))

    def _reward_action_difference(self):
        # Penalize changes in actions
        # print(torch.exp(-0.02*torch.sum(torch.abs(self.last_actions - self.actions), dim=1)))
        return torch.exp(-0.02*torch.sum(torch.abs(self.last_actions - self.actions), dim=1))

    def _reward_torques(self):
        # penalize torques too close to the limit
        return torch.exp(-0.02/self.num_dof*torch.sum((torch.abs(self.torques)) / (self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.),dim=1))

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    def _reward_foot_height(self):
        # Reward foot away from ground
        self.foot_positions = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)
        
        self.get_foot_height=torch.exp(-10*(torch.sum(foot_height**2,dim=1)))
        # print(self.get_foot_height)
        return self.get_foot_height*(torch.norm(self.commands[:, :3], dim=1) > 0.15)
    
    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (x axes)
        x_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-x_vel_error/self.cfg.rewards.tracking_sigma)
    

    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (y axes)
        y_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-y_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    