"""
Very lightweight recurrent rollout buffer with fixed sequence extraction for PPO.
"""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class Step:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    logp: float
    value: float
    mask: float

class RecurrentRolloutBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.logp = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.masks = np.ones((capacity,), dtype=np.float32)
        self.ptr = 0

    def add(self, o,a,r,d,lp,v):
        self.obs[self.ptr] = o
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d
        self.logp[self.ptr] = lp
        self.values[self.ptr] = v
        self.masks[self.ptr] = 1.0 - float(d)
        self.ptr += 1

    def compute_returns_advantages(self, gamma=0.95, lam=0.95, last_value=0.0):
        T = self.ptr
        adv = np.zeros_like(self.rewards[:T])
        ret = np.zeros_like(self.rewards[:T])
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * last_value * nonterminal - self.values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
            last_value = self.values[t]
        ret = adv + self.values[:T]
        self.advantages = adv
        self.returns = ret

    def iter_minibatches(self, batch_size, seq_len):
        # Yield random chunks of length seq_len; in practice align with episodes and meta-episodes
        T = self.ptr
        idxs = np.arange(0, T - seq_len, seq_len)
        np.random.shuffle(idxs)
        for i in range(0, len(idxs), max(1, batch_size//seq_len)):
            sel = idxs[i:i+max(1, batch_size//seq_len)]
            obs = torch.tensor(self.obs[sel[0]:sel[0]+seq_len])[None, ...]  # [1, T, D]
            actions = torch.tensor(self.actions[sel[0]:sel[0]+seq_len])[None, ...]
            returns = torch.tensor(self.returns[sel[0]:sel[0]+seq_len])[None, ...]
            advantages = torch.tensor(self.advantages[sel[0]:sel[0]+seq_len])[None, ...]
            old_logp = torch.tensor(self.logp[sel[0]:sel[0]+seq_len])[None, ...]
            masks = torch.tensor(self.masks[sel[0]:sel[0]+seq_len])[None, ...]
            h0 = torch.zeros(1,1,16)  # hidden size 16
            yield obs, actions, returns, advantages, old_logp, masks, h0, returns
