from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class wowRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096

        scan_dim = 11*11
        priv_dim = 3
        priv_latent_dim = 15 + 1 + 10 +10
        proprio_dim = 2+ 3+ 3+ 3+ 30
        history_len = 10

        num_observations = proprio_dim
        num_privileged_obs = proprio_dim + scan_dim + priv_dim + priv_latent_dim + history_len*proprio_dim# if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 10


    class terrain(LeggedRobotCfg.terrain):
        measured_points_x = [-0.45, -0.3, -0.15, 0,    0.15,  0.3, 0.45, 0.6, 0.75, 0.9, 1.05] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0.,  0.15, 0.3, 0.45, 0.6, 0.75]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.9] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_roll_joint': 0,   # [rad]
            'left_yaw_joint': 0,   # [rad]
            'left_pitch_joint': -0.3,  # [rad]
            'left_knee_joint': 0.6,   # [rad]
            'left_foot_joint': -0.3,

            'right_roll_joint': 0,   # [rad]
            'right_yaw_joint': 0,   # [rad]
            'right_pitch_joint': 0.3 ,  # [rad]
            'right_knee_joint': -0.6,   # [rad]
            'right_foot_joint': 0.3,

        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wow/urdf/wow.urdf'
        name = "wow"
        foot_name = "foot"
        penalize_contacts_on = [" "]
        terminate_after_contacts_on = ["body"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.9
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        class scales( LeggedRobotCfg.rewards.scales ):
            # # termination = -200.
            # termination = -200.
            # tracking_ang_vel = 1.0
            # tracking_lin_vel = 3.
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            
            # torques = -1.e-5
            # dof_acc = -2.e-7
            # action_rate = -0.05 #-0.01
            
            # dof_pos_limits = -1.
            # no_fly = 0.2 #0.25print

            # dof_vel = -0.0
            # stand_still = -0.05
            # action_rate = -0.01 #-0.01
            # smoothness = -0.01

            # power_distribution = -1e-5
            # orientation = -0.2
            # joint_power = -2e-5
            # base_height = -1.0 

            #feet_air_time =  1.0
            #collision = -1.
            #feet_stumble = -0.0 
            # foot_clearance = -0.01
            #stand_still = -0.
            # feet_air_time = 6
            # dof_pos_limits = -1.
            # tracking_lin_vel=2 #参考速度
            # tracking_ang_vel=1.5 #参考角度
            # ang_vel_xy = -0 #旋转惩罚
            # feet_contact_forces = -0.1#惩罚高接触力
            # torques = -2e-5# 惩罚关节扭矩过大
            # dof_vel = -2e-5 # 惩罚关节速度过大
            # dof_acc = -5e-9# 惩罚关节加速度过大
            # # 惩罚机器人底座偏离水平
            # orientation = -1
            #termination = -200.
            #stand_still = -0.05

            #行走reward 
            # tracking_ang_vel = 1.0
            # tracking_lin_vel = 3.
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # delta_torques = -1.0e-7
            # torques = -0.00001 
            # dof_acc = -2.e-7
            # action_rate = -0.01 #-0.01
            # smoothness = -0.01
            # dof_pos_limits = -1.
            # no_fly = 0.1 #0.25print
            # dof_vel = -0.0

             #行走reward 
            # termination = -200.
            # tracking_ang_vel = 1.0
            # tracking_lin_vel = 1.
            # lin_vel_z = -0.5
            # ang_vel_xy = -0.05
            
            # torques = -1.e-5
            # dof_acc = -2.e-7
            # action_rate = -0.015 #-0.01
            
            # dof_pos_limits = -1.
            # # no_fly = 0.2 #0.25print

            # dof_vel=-2.e-5
            # stand_still = -0.05
            # orientation = -0.2
            # smoothness = -0.015
            # joint_power = -2e-5
        #added


            # # 站立reward
            # tracking_ang_vel = 0.0
            # tracking_lin_vel = 1.0
            # lin_vel_z = -1
            # lin_vel_x = -1
            # lin_vel_y = -1
            # ang_vel_xy = -0.00
            # torques = -1.e-5
            # dof_acc = -2.e-7
            # action_rate = -0.01 #-0.01
            # orientation = -5
            # dof_pos_limits = -5.
            # stand_still = 0.05
            # hip_pos = -1
            # base_height = 1.0 


            # termination = -10
            # base_height = -10
            # # feet_air = -1
            # # no_fly = 0. #0.25
            # # dof_vel = -0.0
            # # step_frequency = -0.1
            # # stand_still = 0.05


            #walk and stand on falt
            base_height=0.01
            foot_position_stand=0.09
            # arm_position=0.08
            # arm_position_stand=0.16
            base_acc=0.02
            action_difference=0.02
            torques=0.02
            tracking_x_vel = 2
            tracking_y_vel = 2
            tracking_ang_vel = 1.

            dof_vel=-4e-4
            dof_acc=-4e-6

            lin_vel_z = -0.8
            ang_vel_xy = -0.1

            dof_pos_limits = -2
            foot_height=-0.12

            orientation = -1

            feet_air_time=0.5





class wowCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        adapt_hidden_dims=[128,64,19]
        estimate_hidden_dims = [64,128,41],
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm():
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

    class runner():
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class estimator():
        #train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [19, 128]

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'wow'
  
