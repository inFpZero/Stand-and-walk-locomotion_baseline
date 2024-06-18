from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class wowRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096

        scan_dim = 11*11
        priv_dim = 3
        priv_latent_dim = 15 + 1 + 10 +10
        proprio_dim = 2+ 3+ 30 +6
        history_len = 10

        num_observations = proprio_dim
        num_privileged_obs = proprio_dim + scan_dim + priv_dim + priv_latent_dim # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 10


    class terrain(LeggedRobotCfg.terrain):
        measured_points_x = [-0.45, -0.3, -0.15, 0,    0.15,  0.3, 0.45, 0.6, 0.75, 0.9, 1.05] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0.,  0.15, 0.3, 0.45, 0.6, 0.75]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.912] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_roll_joint': 0.,   # [rad]
            'left_yaw_joint': 0,   # [rad]
            'left_pitch_joint': -0.3,  # [rad]
            'left_knee_joint': 0.6,   # [rad]
            'left_foot_joint': -0.3,

            'right_roll_joint': -0.,   # [rad]
            'right_yaw_joint': 0,   # [rad]
            'right_pitch_joint': 0.3 ,  # [rad]
            'right_knee_joint': -0.6,   # [rad]
            'right_foot_joint': 0.3,

        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {   'roll': 30.0, 'yaw': 30.0,
                        'pitch': 50., 'knee': 50., 'foot': 10.,
                        }  # [N*m/rad]
        damping = {   'roll': 0.75, 'yaw': 0.75,
                        'pitch': 1.25, 'knee': 1.25, 'foot': 0.25,
                        }   # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wow/urdf/wow.urdf'
        name = "wow"
        foot_name = "foot"
        hip_name = "roll"
        penalize_contacts_on = [" "]
        terminate_after_contacts_on = ["body"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        tracking_sigma = 0.2
        orient_tracking_sigma = 2
        soft_torque_limit = 0.9
        soft_dof_pos_limit = 0.90
        base_height_target = 0.86
        target_foot_pos_x = [0. , 0.  ]
        target_foot_pos_y = [0.9, -0.9]
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        class scales:
            # tracking_x_line_vel = 0.2
            # tracking_y_line_vel = 0.15
            # tracking_ang_vel = 0.2
            # roll_pitch_orient = 0.2
            # base_height = 0.05
            # feet_contact = 0.2
            # feet_air_time = 2.0
            # feet_orientation = 0.20
            # feet_position = 0.1
            # base_acc = 0.1
            # action_difference = 0.03
            # torques = 0.1
            # roll_yaw_position = 0.20
            # dof_pos_limits = -0.2
        #added
            # termination = -10
            # base_height = -10
            # # feet_air = -1
            # # no_fly = 0. #0.25
            # # dof_vel = -0.0
            # # step_frequency = -0.1
            # # stand_still = 0.05


            #walk and stand on falt
            # base_height=1
            # foot_position_stand=0.005
            # arm_position=0.08
            # arm_position_stand=0.16
            roll_yaw_position = -0.5
            base_acc=0.02
            torques=0.5
            tracking_x_line_vel = 4
            tracking_y_line_vel = 3
            tracking_ang_vel = 1.
            dof_vel=-6e-5
            dof_acc=-2e-7
            lin_vel_z = -1
            ang_vel_xy = -0.05
            dof_pos_limits = -10
            action_rate = -0.02 #-0.01

            orientation = -2
            feet_air_time = 4
            feet_contact = 3
            feet_orientation = 2

class wowCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        estimate_hidden_dims = [64,128,41],
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 2
        
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
        policy_class_name = 'ActorCriticRecurrent' #ActorCriticRecurrent
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'wow'
  
