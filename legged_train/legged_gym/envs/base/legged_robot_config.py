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

from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn
class LeggedRobotCfg(BaseConfig):
    class play:
        load_student_config = False
        mask_priv_obs = False
    class env:
        num_envs = 4096
        scan_dim = 17*11
        priv_dim = 3
        priv_latent_dim = 4 + 1 + 12 +12
        proprio_dim = 2 + 3+ 3+ 3+ 36
        history_len = 10

        num_observations = proprio_dim 
        num_privileged_obs = proprio_dim + scan_dim + priv_dim + priv_latent_dim + history_len*proprio_dim# if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        episode_length_s = 20 # episode length in seconds
        obs_type = "og"
        
        history_encoding = False
        reorder_dofs = True
        send_timeouts = True # send time out information to the algorithm

        # additional visual inputs 
        include_foot_contacts = True
        
        randomize_start_x_pos = False
        randomize_start_pos = True
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 3.14
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class depth:
        use_camera = False      
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.343, 0, 0.07]  # front camera
        angle = [17, 23]  # positive pitch down

        update_interval = 5 # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        quantize_height = True
        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05

        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1

        vertical_scale = 0.005 # [m]
        border_size = 4 # [m]
        height = [0.01, 0.02]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        max_stair_height = 0.23
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = True  # select a unique terrain type and pass all arguments
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8#4
        terrain_width = 8 #3
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall ! make sure num_rows > 5
        num_cols = 20 # number of terrain cols (types)

        terrain_dict = {
                "smooth slope": 0.0, 
                "rough slope up": 0.0,
                "rough slope down":0.0,
                "stairs up": 0., 
                "stairs down": 0., 
                "discrete": 0., 
                "stepping stones": 0.,
                "gaps": 0., 
                "smooth flat": 0.0,
                "pit": 0.0,
                "wall": 0.0,
                "platform": 0,
                "large stairs up": 0.,
                "large stairs down": 0.,
                "parkour": 0.,
                "parkour_hurdle": 0.,
                "parkour_flat": 0.,
                "parkour_step": 0.,
                "parkour_gap": 0,
                "plane": 1,
                "demo": 0.0,
                "icra_qrc_a":0.,
                "icra_qrc_b":0.,
                "icra_qrc_e":0.,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False

        num_goals = 8

    class commands: 
        curriculum = False
        max_curriculum = 1.
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        forward_curriculum_threshold = 0.8
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        yaw_curriculum_threshold = 0.5
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.2

        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]

        heading = [-3.14, 3.14]
        # Easy ranges
        class ranges:
            lin_vel_x = [0.5, 1] # min max [m/s]
            lin_vel_y = [-0, 0]   # min max [m/s]
            ang_vel_yaw = [-0.1,0.1]    # min max [rad/s]
            heading = [-0.1, 0.1]

        # Easy ranges
        class max_ranges:
            lin_vel_x = [-0.4, 1.0] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3,0.3]    # min max [rad/s]
            heading = [-3.14, 3.14]

        class crclm_incremnt:
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1    # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        target_joint_angles= { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]

        randomize_restitution = True
        restitution_range = [0., 0.4]

        ##--------------------------

        randomize_base_mass = True
        added_mass_range = [-0.3, 0.3]

        randomize_base_com = True
        added_com_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]

        ##--------------------------

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]

        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]
        
        randomize_force_inject = True
        RFI_range = [-0.05, 0.05]

        randomize_actuate_offset = False # don't be ture
        RAO_range = [-0.1, 0.1]

        ##--------------------------
        
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.7

        disturbance = True
        disturbance_range = [-10.0, 10.0]
        disturbance_interval = 8

        ##--------------------------

        action_delay = True
        delay_inject_steps = 24 * 1

        
    class rewards:
        class scales:
            tracking_lin_vel = 1.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1
        soft_torque_limit = 1
        base_height_target = 1
        max_contact_force = 100. # forces above this value are penalized


    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        sigmoid_output = True
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
    
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt