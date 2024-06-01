import glob
import pickle as pkl
import lcm
import sys

from legged_deploy.utils.deployment_runner import DeploymentRunner
from legged_deploy.envs.lcm_agent import LCMAgent
from legged_deploy.envs.actor_critic import *
from legged_deploy.envs.actor_critic_test import *
from legged_deploy.envs.actor_critic_md import *
from legged_deploy.envs.estimator import *
from legged_deploy.utils.cheetah_state_estimator import StateEstimator
from legged_deploy.utils.command_profile import *
from legged_deploy.envs.depth_backbone import *
from copy import copy, deepcopy


import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

    # with open(logdir+"/parameters.pkl", 'rb') as file:
    #     pkl_cfg = pkl.load(file)
    #     cfg = pkl_cfg["Cfg"]

    cfg = None

    se = StateEstimator(lc)

    control_dt = 0.01
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=1, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from legged_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy(logdir,if_distill = False, if_test=True, if_md_ac=True):
    # body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    loaded_dict = torch.load(logdir + '/checkpoint/model_3000(1).pt')


    if not if_test:
        ac =  ActorCriticRMA(num_prop=38,
                            num_scan=81,
                            num_critic_obs=742,
                            num_priv_latent = 33,
                            num_priv_explicit = 9,
                            num_hist = 10,
                            num_actions = 9,
                            )
        
        estimator = Estimator(input_dim=38, output_dim=9, hidden_dims=[128, 64])
        
        if if_distill:
            distill_actor = deepcopy(ac.actor)
            depth_backbone = DepthOnlyFCBackbone58x87(38, 32, 512)
            encoder = RecurrentDepthBackbone(depth_backbone)

            distill_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])

            encoder.eval()
            distill_actor.eval()

        else:
            ac.load_state_dict(loaded_dict['model_state_dict'])
            ac.actor.eval()
                                
        estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        estimator.eval() 
            
        import os

        def policy(obs,if_distill = True):
            obs_priv = estimator(obs[:,:38].to('cpu'))
            obs[:,38:38+9] = obs_priv
            # if if_distill:
            latent = encoder(obs[:,:38].to('cpu'))
            actions = distill_actor(obs.to('cpu'), hist_encoding=True, scandots_latent = latent)
            # print(actions)

            # else:
            #     actions = ac.act_inference(obs.to('cpu'), hist_encoding=True)   
            return actions
        
    elif if_md_ac:
        ac = ActorCriticMD(num_prop=27,
                        num_priv_explicit=3,
                        num_hist=5,
                        num_critic_obs=151,
                        num_actions=8,
                        )
        estimator = Estimator(input_dim=27, output_dim=3, hidden_dims=[128, 64])
        
        ac.load_state_dict(loaded_dict['model_state_dict'])
        ac.actor.eval()

        estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        estimator.eval() 

        def policy(obs):
            obs_priv = estimator(obs[:,:27].to('cpu'))
            obs[:,27:27+3] = obs_priv
            actions = ac.act_inference(obs.to('cpu'))
            return actions
        
    else:
        # ac = ActorCritic(num_actor_obs=36,
        #                 num_critic_obs=36,
        #                 num_actions=9)
        ac = ActorCritic(num_actor_obs=27,
                        num_critic_obs=27,
                        num_actions=8)        
        ac.load_state_dict(loaded_dict['model_state_dict'])
        ac.actor.eval()

        def policy(obs):
            actions = ac.act_inference(obs.to('cpu')) 
            return actions

    return policy



if __name__ == '__main__':
    label = "diff_a1/train"

    experiment_name = "example_experiment"
    
    load_and_run_policy(label, experiment_name=experiment_name, max_vel=2, max_yaw_vel=2)
