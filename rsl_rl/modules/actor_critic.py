#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from networks.cnn import SimpleConv
from networks.image_attention import ImageAttention


matrix_size = 50,
patch_size = 10,
num_heads = 1,
embed_dim = 10,
in_channel = 2

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="tanh",
        init_noise_std=0.5,
        visual_latent="attention",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index-1]))   # Muye --> add layer norm for stability
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        # Muye --> add 2 activation as the last layer
        actor_layers.append(activation)
        actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_activation = get_activation("elu")
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(critic_activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index-1]))   # Muye --> add layer norm for stability
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(critic_activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

        # Muye --> added Conv2d module to process command map (50 x 50)
        if visual_latent == "cnn":
            self.cmd_map_latent = SimpleConv()
        if visual_latent == "attention":
            self.cmd_map_latent = ImageAttention(img_size=matrix_size, patch_size=patch_size,
                                                 n_heads=num_heads, d_embed=embed_dim, in_chans=in_channel)

        self.obs_latent = None

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
        cmd_latent = self.cmd_map_latent(observations)
        self.obs_latent = cmd_latent

        cmd_latent = cmd_latent.reshape((cmd_latent.shape[0], -1))
        mean = self.actor(cmd_latent)

        # if torch.min(mean) < 0:
        #     mean -= torch.min(mean)

        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        cmd_latent_critic = self.cmd_map_latent(critic_observations)
        cmd_latent_critic = (cmd_latent_critic-torch.min(cmd_latent_critic))/(torch.max(cmd_latent_critic)-torch.min(cmd_latent_critic))
        # print(f"actor_critic --> cmd_latent_critic shape: {cmd_latent_critic.shape}")
        batch_size = cmd_latent_critic.shape[0]
        cmd_latent_critic = cmd_latent_critic.reshape((batch_size,-1))
        value = self.critic(cmd_latent_critic)
        # print(f"actor_critic --> value: {value}")
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
