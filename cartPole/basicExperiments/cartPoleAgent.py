# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import numpy as np

import torch.nn as nn
import torch
from gymnasium.vector.vector_env import VectorEnv
from torch.distributions.categorical import Categorical

class Agent(nn.Module):
    def __init__(self, envs: VectorEnv):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class GRUAgent(nn.Module):
    def __init__(self, envs: VectorEnv, hidden_size: int = 128):
        # Init Actor-Critic model
        super().__init__()
        # GRU receives input size equal to observation space size
        # and outputs hidden state of specified size
        self.hidden_size = hidden_size
        self.gru_Critic = nn.GRU(
            input_size=np.array(envs.single_observation_space.shape).prod(),
            hidden_size=hidden_size,
            batch_first=True
        )

        self.mlp_Critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size + np.array(envs.single_observation_space.shape).prod(), 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 1), std=1.0),
        )

        self.gru_Actor = nn.GRU(
            input_size=np.array(envs.single_observation_space.shape).prod(),
            hidden_size=hidden_size,
            batch_first=True
        )
        self.mlp_Actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size + np.array(envs.single_observation_space.shape).prod(), 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, envs.single_action_space.n), std=0.01),
        )
    
    def get_value(self, x, hx=None):
        # Forward through GRU and MLP to get state value
        h_seq, hT = self.gru_Critic(x.unsqueeze(1), hx)

        hx = torch.cat([h_seq, x.unsqueeze(1)], dim=-1)
        z = self.mlp_Critic(hx)
        return z, hT

    def get_action_and_value(self, x, h_actor=None, h_critic=None, action=None) -> tuple:
        # Forward through GRU and MLP to get action logits
        h_seq, h_actor = self.gru_Actor(x.unsqueeze(1), h_actor)
        hx = torch.cat([h_seq, x.unsqueeze(1)], dim=-1)
        logits = self.mlp_Actor(hx)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        val, h_critic = self.get_value(x, h_critic)
        return action, h_actor, probs.log_prob(action), probs.entropy(), val, h_critic


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
