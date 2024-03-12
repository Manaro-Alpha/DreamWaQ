from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic_DWAQ(nn.Module):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, cenet_in_dim, cenet_out_dim, activation="elu", init_noise_std=1.0,):
        super().__init__()

        self.activation = get_activation(activation)
        actor_input_dim = num_actor_obs
        critic_input_dim = num_critic_obs

        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim,128),
            self.activation,
            nn.Linear(128,64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_logvar_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_mean_vel = nn.Linear(64,3)
        self.encode_logvar_vel = nn.Linear(64,3)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim,64),
            self.activation,
            nn.Linear(64,128),
            self.activation,
            nn.Linear(128,45)
        )

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
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def cenet_forward(self,obs_history):
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        # var = torch.exp(logvar_latent*0.5)
        # code_temp = torch.randn_like(var)
        # code = mean_latent + var*code_temp
        # print("latent : ",code[0])
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_mean_vel(distribution)
        code_latent = self.reparameterise(mean_latent,logvar_latent)
        code_vel = self.reparameterise(mean_vel,logvar_vel)
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, obs_history, **kwargs):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations),dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations,obs_history):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value



        











def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
