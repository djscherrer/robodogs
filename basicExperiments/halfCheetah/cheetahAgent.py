# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import numpy as np
import torch.nn as nn
import torch


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        # Critic (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor mean network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        # Log standard deviation (trainable parameter)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        # Small epsilon for numerical stability
        self._eps = 1e-6

    def get_value(self, x):
        return self.critic(x)

    def _gaussian(self, mu):
        std = self.actor_logstd.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def _squash(self, raw_action, dist):
        """
        Apply tanh squashing and correct log-probabilities.
        """
        a = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - a.pow(2) + self._eps)
        return a, log_prob.sum(-1)

    def get_action_and_value(self, x, action=None):
        mu = self.actor_mean(x)
        dist = self._gaussian(mu)

        if action is None:
            raw_action = dist.rsample()  # for reparameterization
            action, logprob = self._squash(raw_action, dist)
        else:
            # If external action given, invert tanh for correct logprob
            action = action.clamp(-1 + self._eps, 1 - self._eps)
            raw_action = 0.5 * torch.log((1 + action) / (1 - action))
            _, logprob = self._squash(raw_action, dist)

        entropy = dist.entropy().sum(-1)
        value = self.critic(x)
        return action, logprob, entropy, value
    

class GRUAgent(nn.Module):
    def __init__(self, envs, hidden_size: int = 128):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.hidden_size = hidden_size

        # Critic
        self.gru_Critic = nn.GRU(input_size=obs_dim, hidden_size=hidden_size, batch_first=True)
        self.mlp_Critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size + obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor (Gaussian)
        self.gru_Actor = nn.GRU(input_size=obs_dim, hidden_size=hidden_size, batch_first=True)
        self.mlp_mu = nn.Sequential(
            layer_init(nn.Linear(hidden_size + obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        # global log_std parameter (you can also predict it per-state if you want)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # for tanh squashing correction
        self._eps = 1e-6

    def get_value(self, x, hx=None):
        h_seq, hT = self.gru_Critic(x.unsqueeze(1), hx)           # [B,1,H], [1,B,H]
        z = self.mlp_Critic(torch.cat([h_seq.squeeze(1), x], -1)) # [B,1]
        return z, hT

    def _gaussian(self, mu):
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def _squash(self, raw_action, dist):
        # tanh squash into [-1,1] and apply change-of-variables correction
        a = torch.tanh(raw_action)
        # log_prob correction: log|det d_tanh|
        # log(1 - tanh(x)^2) = log(1 - a^2)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - a.pow(2) + self._eps)
        # sum over action dims
        return a, log_prob.sum(-1)

    def get_action_and_value(self, x, h_actor=None, h_critic=None, action=None):
        # Actor
        h_seq, h_actor = self.gru_Actor(x.unsqueeze(1), h_actor)
        mu = self.mlp_mu(torch.cat([h_seq.squeeze(1), x], -1))
        dist = self._gaussian(mu)

        if action is None:
            raw_action = dist.rsample()  # rsample for PPO w/ reparam
            action, logprob = self._squash(raw_action, dist)
        else:
            # inverse tanh to recover raw_action for correct logprob
            # clamp to avoid NaNs
            action = action.clamp(-1 + self._eps, 1 - self._eps)
            raw_action = 0.5 * torch.log((1 + action) / (1 - action))
            _, logprob = self._squash(raw_action, dist)

        # Entropy (of the squashed dist is tricky); typical PPO reports base entropy
        entropy = dist.entropy().sum(-1)

        # Critic
        val, h_critic = self.get_value(x, h_critic)
        return action, h_actor, logprob, entropy, val, h_critic


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
