"""
GRU(16) + MLP policy producing mean actions and value estimates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_state_size: int = 16, net_arch=(128,128)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_state_size = hidden_state_size

        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_state_size, batch_first=True)
        layers = []
        last = hidden_state_size + obs_dim  # concatenate h_t and x_t like paper
        for h in net_arch:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Parameter(torch.ones(1, act_dim) * -1.0)
        self.v = nn.Linear(last, 1)

    def forward(self, x, h0):
        # x: [B, T, obs_dim]
        h_seq, hT = self.gru(x, h0)  # h_seq: [B,T,H]
        # concat each step's h and x
        hx = torch.cat([h_seq, x], dim=-1)
        z = self.mlp(hx)
        mu = self.mu(z)
        v = self.v(z).squeeze(-1)
        return mu, v, hT

    def evaluate(self, x, actions, h0):
        # x: [B,T,obs_dim], actions: [B,T,act_dim]
        mu, v, _ = self.forward(x, h0)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return dist, v, logp, entropy
