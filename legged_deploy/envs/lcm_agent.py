import time
import torch
import lcm
import numpy as np
import torch
import cv2
import os
import json

from legged_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent():
    def __init__(self, cfg, se, command_profile):
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se = se
        self.command_profile = command_profile

        self.timestep =  0.
        # self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        # self.num_obs = self.cfg["env"]["num_observations"]
        # self.num_envs = 1
        # self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        # self.num_obs_scan =  self.cfg["env"]["n_scan"]
        # self.num_obs_priv = self.cfg["env"]["n_priv"]
        # self.num_obs_n_priv_latent = self.cfg["env"]["n_priv_latent"]
        # self.num_obs_proprio = self.cfg["env"]["n_proprio"]
        # self.obs_history_len = self.cfg["env"]["history_len"]

        # self.num_actions = self.cfg["env"]["num_actions"]
        # self.num_commands = self.cfg["commands"]["num_commands"]
        # if "obs_scales" in self.cfg.keys():
        #     self.obs_scales = self.cfg["obs_scales"]
        # else:
        #     self.obs_scales = self.cfg["normalization"]["obs_scales"]
        # self.commands_scale = np.array(
        #     [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"],
        #      self.obs_scales["ang_vel"]
        #      ])[:(self.num_commands-1)]
        
        self.dt = 0.02

        self.num_envs = 1
        self.num_obs_scan =  11*11
        self.num_obs_priv = 9
        self.num_obs_n_priv_latent = 4+1+12+12
        self.num_obs_proprio = 27
        self.obs_history_len = 10
        self.num_obs = 27+ 17*11 + (10*27) + 4+1+12+12 + 9 #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        self.num_actions = 12
        self.num_commands = 3
        self.device = 'cpu'

        self.commands_scale = np.array(
            [2.0, 
             2.0,
             0.25
             ])[:self.num_commands]


        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]
        
        # self.initial_dof_pos =np.array([ 0. ,0.,0,
        #                                  0.27,-0.05,0.2,
        #                                  -0.27,-0.3,0.4,
        #                                  -0.,0,0 ])

        # self.initial_dof_pos =np.array([ 0,0,0.27,
        #                                  -0.05,0.2,-0.27,
        #                                  -0.3,0.4,0,
        #                                  -0.,0,0, ])


        self.default_dof_pos = np.array([ 0. ,0.,0,
                                         -0.,0.,0,
                                         0.,0,0,
                                         -0.,0,0 ])
        # try:
        #     self.default_dof_pos_scale = np.array([self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
        #                                            self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
        #                                            self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
        #                                            self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"]])
        # except KeyError:
        self.default_dof_pos_scale = np.ones(12)         

        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            self.p_gains[i] = 40
            self.d_gains[i] = 1
            found = True
        print(f"p_gains: {self.p_gains}")

        self.commands = np.zeros((1, self.num_commands))
        self.imu_obs = np.zeros(3)
        self.actions = torch.zeros(12)
        self.last_actions = torch.zeros(12)
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)
        self.obs_history_buf = np.zeros((1,self.obs_history_len, self.num_obs_proprio))
        # self.action_history_buf = np.zeros(self.cfg.domain_rand.action_buf_len, self.num_dofs)
        self.actions = self.actions.reshape(1,-1)

        self.joint_idxs = self.se.joint_idxs

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)

        # if "obs_scales" in self.cfg.keys():
        #     self.obs_scales = self.cfg["obs_scales"]
        # else:
        #     self.obs_scales = self.cfg["normalization"]["obs_scales"]
  
        self.obs_scales_ang_vel = 0.25
        self.obs_scales_lin_vel = 2.0
        self.obs_scales_dof_pos = 1.0
        self.obs_scales_dof_vel = 0.05

        self.net_cat_yaw = torch.tensor([0])

        self.net_cat = torch.tensor([0,0,0])

        self.is_currently_probing = False
    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_obs(self):

        self.gravity_vector = self.se.get_gravity_vector()
        # print("no reshape",self.gravity_vector )
        # print("reshape",self.gravity_vector.reshape(1,-1) )

        cmds, reset_timer = self.command_profile.get_command(self.timestep * self.dt, probe=self.is_currently_probing)
        self.commands[:, :] = [cmds[:self.num_commands]]
        #else:
        #    self.commands[:, 0:3] = self.command_profile.get_command(self.timestep * self.dt)[0:3]
        self.imu_obs= self.se.get_rpy()
        self.dof_pos = self.se.get_dof_pos()
        print(self.imu_obs)
        self.dof_vel= self.se.get_dof_vel()
        self.body_linear_vel = self.se.get_body_linear_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()

        self.contact_state = self.se.get_contact_state()
        # ...............................................................................................................
        # obs_buf = np.concatenate((#skill_vector, 
        #                     self.imu_obs[:2].reshape(1,-1),
        #                     self.body_angular_vel.reshape(1,-1)  * self.obs_scales_ang_vel,   #[1,3]
        #                     self.gravity_vector.reshape(1,-1),
        #                     self.commands[:,:]*self.commands_scale,  #[1,3]
        #                     (self.dof_pos[:8] - self.default_dof_pos[:8]).reshape(1,-1) * self.obs_scales_dof_pos,
        #                     self.dof_vel[:8].reshape(1,-1) * self.obs_scales_dof_vel,
        #                     self.actions[:,:8].cpu().detach().numpy(),
        #                     ),axis=1)
        # # set priv_obs zeros
        

        # # # # 存儲dof_pos到文件
        # # print(4*self.dof_pos)

        # # self.dof_pos_list.append((4* self.self.dof_pos).detach().cpu().numpy().tolist())
        # # print(self.dof_pos_list)
        # # dof_pos_filename = os.path.join('/home/linqi/', 'dof_pos_real.json')
        # # os.makedirs(os.path.dirname(dof_pos_filename), exist_ok=True)
        # # with open(dof_pos_filename, 'w') as f:
        # #     json.dump(self.dof_pos_list, f)


        #  #right_yaw,right_foot,left_foot,left_roll,left_pitch,left_calf,right_roll,right_pitch,right_calf
        # priv_explicit = np.zeros(self.num_obs_priv)

        # priv_latent = np.zeros(self.num_obs_n_priv_latent)
        
        # heights = np.zeros(self.num_obs_scan)

        # self.obs_buf = np.concatenate((obs_buf,  priv_explicit.reshape(1,-1), heights.reshape(1,-1), priv_latent.reshape(1,-1), self.obs_history_buf.reshape(1,-1)), axis=1)
        # # set priv_obs zeros
        # self.obs_history_buf = np.where(
        #     (self.timestep <= 1),
        #     np.stack([obs_buf] * self.obs_history_len, axis=1),
        #     np.concatenate([
        #         self.obs_history_buf[:,1:],
        #         obs_buf[:,np.newaxis]
        #     ], axis=1)
        # )
        # self.commands = np.zeros((1, self.num_commands))
        # self.commands[0][0] =0.2
        # self.obs_buf_test = np.concatenate((#skill_vector, 
        #                     self.body_angular_vel.reshape(1,-1)  * self.obs_scales_ang_vel,   #[1,3]
        #                     self.gravity_vector.reshape(1,-1),
        #                     self.commands[:,:]*self.commands_scale,  #[1,3]
        #                     (self.dof_pos[3:9] - self.default_dof_pos[3:9]).reshape(1,-1) * self.obs_scales_dof_pos,
        #                     self.dof_vel[3:9].reshape(1,-1) * self.obs_scales_dof_vel,
        #                     self.actions[:,3:9].cpu().detach().numpy(),
        #                     ),axis=1)

        # clip_obs = 100
        # self.obs_buf = np.clip(self.obs_buf_test, -clip_obs, clip_obs)



        #------------------------AC_MD--------------------------------
        self.commands = np.zeros((1, self.num_commands))
        obs_buf = np.concatenate((#skill_vector, 
                            self.body_angular_vel.reshape(1,-1)  * self.obs_scales_ang_vel,   #[1,3]
                            self.gravity_vector.reshape(1,-1),
                            self.commands[:,:]*self.commands_scale,  #[1,3]
                            (self.dof_pos[3:9] - self.default_dof_pos[3:9]).reshape(1,-1) * self.obs_scales_dof_pos,
                            self.dof_vel[3:9].reshape(1,-1) * self.obs_scales_dof_vel,
                            self.actions[:,3:9].cpu().detach().numpy(),
                            ),axis=1)
        # set priv_obs zeros
        # print(self.dof_vel)
         #right_yaw,right_foot,left_foot,left_roll,left_pitch,left_calf,right_roll,right_pitch,right_calf
        
        self.obs_history_buf = np.where(
            (self.timestep <= 1),
            np.stack([obs_buf] * self.obs_history_len, axis=1),
            np.concatenate([
                self.obs_history_buf[:,1:],
                obs_buf[:,np.newaxis]
            ], axis=1)
        )
        
        priv_explicit = np.zeros(self.num_obs_priv)

        heights = np.zeros(self.num_obs_scan)

        self.obs_buf = np.concatenate((obs_buf,  priv_explicit.reshape(1,-1), heights.reshape(1,-1), self.obs_history_buf.reshape(1,-1)), axis=1)
        # set priv_obs zeros
        


        clip_obs = 100
        self.obs_buf = np.clip(self.obs_buf, -clip_obs, clip_obs)

        # print(self.obs_buf)
        # print(self.dof_pos)
        # 更新 self.contact_buf
        # self.contact_buf = np.where(
        #     (self.timestep <= 1)[:, None, None],
        #     np.stack([self.contact_filt.astype(float)] * self.cfg.env.contact_buf_len, axis=1),
        #     np.concatenate([
        #         self.contact_buf[1:],
        #         self.contact_filt.astype(float)[:]
        #     ], axis=1)
        # )

            # print(self.clock_inputs)

        # if self.cfg["env"]["observe_vel"]:
        #     ob = np.concatenate(
        #         (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
        #          self.body_angular_vel.reshape(1, -1) * self.obs_scales["ang_vel"],
        #          ob), axis=1)

        # if self.cfg["env"]["observe_only_lin_vel"]:
        #     ob = np.concatenate(
        #         (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
        #          ob), axis=1)

        # if self.cfg["env"]["observe_yaw"]:
        #     heading = self.se.get_yaw()
        #     ob = np.concatenate((ob, heading.reshape(1, -1)), axis=-1)

        # self.contact_state = self.se.get_contact_state()
        # if "observe_contact_states" in self.cfg["env"].keys() and self.cfg["env"]["observe_contact_states"]:
        #     ob = np.concatenate((ob, self.contact_state.reshape(1, -1)), axis=-1)

        # if "terrain" in self.cfg.keys() and self.cfg["terrain"]["measure_heights"]:
        #     robot_height = 0.25
        #     self.measured_heights = np.zeros
        #         (len(self.cfg["terrain"]["measured_points_x"]), len(self.cfg["terrain"]["measured_points_y"]))).reshape(
        #         1, -1)
        #     heights = np.clip(robot_height - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales["height_measurements"]
        #     ob = np.concatenate((ob, heights), axis=1)


        return torch.tensor(self.obs_buf, device=self.device).float()

    def get_privileged_observations(self):
        return None

    def publish_action(self, action, hard_reset=False):

        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[0, :12].detach().cpu().numpy() * 0.25).flatten()
        # self.joint_pos_target = np.insert(self.joint_pos_target, 0, 0)
        # print(self.joint_pos_target)
        # self.joint_pos_target[[0, 3, 6, 9]] *= -1
        # self.joint_pos_target = self.joint_pos_target
        self.joint_pos_target += self.default_dof_pos
        joint_pos_target = self.joint_pos_target
        self.joint_vel_target = np.zeros(12)
        # command_for_robot.q_des = joint_pos_target
        # command_for_robot.qd_des = self.joint_vel_target
        # command_for_robot.kp = self.p_gains
        # command_for_robot.kd = self.d_gains
        # command_for_robot.tau_ff = np.zeros(12)
        # command_for_robot.se_contactState = np.zeros(4)
        # command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        # command_for_robot.id = 0

        command_for_robot.q_des = [0,0,0,0,0,0,0,0,0,0,0,0]
        command_for_robot.qd_des = [0,0,0,0,0,0,0,0,0,0,0,0]
        command_for_robot.kp = [4.45,1,1,1,1,1,1,1,1,1,1,1]
        command_for_robot.kd = [1,1,1,1,1,1,1,1,1,1,1,1]
        command_for_robot.tau_ff = [0,0,0,0,0,0,0,0,0,0,0,0]
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains
        # print(command_for_robot.q_des)
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros(12).reshape(1,-1)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def reset_gait_indices(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step(self, actions, hard_reset=False):
        clip_actions = 4.8
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.actions = torch.cat((self.actions,self.net_cat.reshape(1,-1)),dim=1)
        self.actions = torch.cat((self.net_cat_yaw.reshape(1,-1),self.actions),dim=1)

        # self.actions = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,-1)
        #right_yaw,right_foot,left_foot,left_roll,left_pitch,left_calf,right_roll,right_pitch,right_calf
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz');
        self.time = time.time()
        obs = self.get_obs()

        # clock accounting
        # frequencies = self.commands[:, 4]
        # phases = self.commands[:, 5]
        # offsets = self.commands[:, 6]
        # if self.num_commands == 8:
        #     bounds = 0
        #     durations = self.commands[:, 7]
        # else:
        #     bounds = self.commands[:, 7]
        #     durations = self.commands[:, 8]
        # self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        # if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
        #     self.foot_indices = [self.gait_indices + phases + offsets + bounds,
        #                          self.gait_indices + bounds,
        #                          self.gait_indices + offsets,
        #                          self.gait_indices + phases]
        # else:
        #     self.foot_indices = [self.gait_indices + phases + offsets + bounds,
        #                          self.gait_indices + offsets,
        #                          self.gait_indices + bounds,
        #                          self.gait_indices + phases]
        self.clock_inputs[:, 0] = 0
        self.clock_inputs[:, 1] = 0
        self.clock_inputs[:, 2] = 0
        self.clock_inputs[:, 3] = 0


        images = {'front': self.se.get_camera_front(),
                  'bottom': self.se.get_camera_bottom(),
                  'rear': self.se.get_camera_rear(),
                  'left': self.se.get_camera_left(),
                  'right': self.se.get_camera_right()
                  }
        downscale_factor = 2
        temporal_downscale = 3

        for k, v in images.items():
            if images[k] is not None:
                images[k] = cv2.resize(images[k], dsize=(images[k].shape[0]//downscale_factor, images[k].shape[1]//downscale_factor), interpolation=cv2.INTER_CUBIC)
            if self.timestep % temporal_downscale != 0:
                images[k] = None
        #print(self.commands)

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": None,
                 "camera_image_front": images['front'],
                 "camera_image_bottom": images['bottom'],
                 "camera_image_rear": images['rear'],
                 "camera_image_left": images['left'],
                 "camera_image_right": images['right'],
                 }

        self.timestep += 1
        return obs, None, None, infos
