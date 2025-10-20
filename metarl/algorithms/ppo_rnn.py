"""
A very small recurrent PPO skeleton with GRU.
For production, consider using an existing library (e.g., SB3 RecurrentPPO) and adapt sequence sampling + K-episode memory.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..utils.rollout_buffer import RecurrentRolloutBuffer

@dataclass
class PPOConfig:
    gamma: float = 0.95
    learning_rate: float = 5e-5
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    update_epochs: int = 10
    sequence_length: int = 16
    meta_episode_length: int = 5
    batch_size: int = 1024

class RecurrentPPO:
    def __init__(self, policy, cfg: PPOConfig):
        self.policy = policy
        self.cfg = cfg
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)

    def update(self, rollouts: RecurrentRolloutBuffer):
        """Perform PPO updates on collected rollouts."""
        device = next(self.policy.parameters()).device  # detect whether we're on cpu/mps/cuda

        for _ in range(self.cfg.update_epochs):
            for batch in rollouts.iter_minibatches(self.cfg.batch_size, self.cfg.sequence_length):
                # Unpack batch and move all to the same device as the policy
                obs, actions, returns, advantages, old_logp, masks, h0, vtargets = batch

                obs = obs.to(device)
                actions = actions.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                old_logp = old_logp.to(device)
                masks = masks.to(device)
                h0 = h0.to(device)
                vtargets = vtargets.to(device)

                # Forward through policy to get new predictions
                pi, v, logp, entropy = self.policy.evaluate(obs, actions, h0)

                # PPO ratio and losses
                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range)
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

                value_loss = ((v - vtargets) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss

                # Optimize
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()
