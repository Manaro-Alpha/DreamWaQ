import glob
import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())


    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
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

def load_policy(logdir):
    actor=torch.jit.load(logdir + '/actor_1.pt')
    encoder=torch.jit.load(logdir + "/encoder_1.pt")
    fc_mu = torch.jit.load(logdir + '/encoder_mu_1.pt')
    fc_var = torch.jit.load(logdir + '/encoder_var_1.pt')

     
    def policy(obs, info):
        i = 0
      
        h = encoder(obs["obs_history"].to('cpu').float())
        mu = fc_mu(h)
        log_var = fc_var(h)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        action = actor(torch.cat((obs["obs"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretrain-v0/train"

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, max_vel=0.75, max_yaw_vel=1.0)
