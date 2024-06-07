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
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 10 if not args.save else 10
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.height = [0.02, 0.02]
    # env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
    #                     "rough slope up": 0.,
    #                     "rough slope down": 0.,
    #                     "rough stairs up": 0., 
    #                     "rough stairs down": 0., 
    #                     "discrete": 0., 
    #                     "stepping stones": 0.,
    #                     "gaps": 0., 
    #                     "smooth flat": 0.,
    #                     "pit": 0.0,
    #                     "wall": 0.0,
    #                     "platform": 1.,
    #                     "large stairs up": 0.,
    #                     "large stairs down": 0.,
    #                     "parkour": 0.,
    #                     "parkour_hurdle": 0.,
    #                     "parkour_flat": 0.,
    #                     "parkour_step": 0.,
    #                     "parkour_gap": 0.,
    #                     "palne": 0,
    #                     "demo": 0.0,}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 10
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = True
    env_cfg.commands.heading_command = True

    depth_latent_buffer = []
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)

        
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)
    infos = {}
    count = 0

    ##-------------collect obs date for test real motor-------------
    time_length = 20
    obs_buffer = torch.zeros(time_length*50, env.num_actions, device=env.device, requires_grad=False)

    for i in range(10*int(env.max_episode_length)):

        actions = policy(obs.detach())

        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if i<time_length*50:
            obs_buffer[i,:] = obs[0,-36:-24]
        if i==time_length*50:
            obs_buffer

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        if i%10 == 0 :
            print("time:", env.episode_length_buf[env.lookat_id].item() / 500, 
                "cmd vx", env.commands[env.lookat_id, 0].item(),
                "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
            print(env.command_ranges["lin_vel_x"])
        id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

