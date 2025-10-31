#!/usr/bin/env python3
"""
Simple GRU policy evaluation on Gymnasium MuJoCo HalfCheetah-v4
----------------------------------------------------------------

Matches the lean style of your other eval script: constants at top,
`pick_device()` for device selection, strict-ish config read, no extra
adapters or random policies.

Usage
-----
    python eval_halfcheetah_simple.py

Tweak the constants below for checkpoint/config/steps/render.
"""
from __future__ import annotations

import os
import json
import time
from typing import Optional

import numpy as np
import torch

from metarl.policies.gru_policy import GRUPolicy
from metarl.envs.mujoco_cheetah_env import HalfCheetahAdapter

import random

# =============================
# User-tunable constants
# =============================
CKPT = "checkpoints/cheetah/policy_20k.pt"
CFG  = "configs/metarl_default_cheetah.json"
STEPS = 1000
RENDER = True
SEED: int = 0

# Deterministic eval: use tanh(mu). If False and your policy returns log_std,
# we sample tanh(N(mu, std)). If policy returns only (mu, value, h), this flag
# has no effect (we use tanh(mu) to keep actions in [-1,1]).
DETERMINISTIC: bool = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# If your GRUPolicy already outputs squashed actions, set this False
APPLY_TANH_ON_MU: bool = True


# =============================
# Utilities
# =============================

def pick_device(prefer=None):
    prefer = (prefer or "").lower()
    if prefer == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# =============================
# Eval
# =============================

def main():
    assert os.path.exists(CKPT), f"Checkpoint not found: {CKPT}"
    assert os.path.exists(CFG), f"Config not found: {CFG}"

    with open(CFG, "r") as f:
        cfg = json.load(f)

    envp = cfg.get("environment_params", {})
    hyp = cfg.get("train_hyp_params", {})

    obs_dim_cfg = envp.get("obs_dim")
    render_mode = "human" if RENDER else envp.get("render_mode", None)

    # --- Env ---
    env = HalfCheetahAdapter(obs_dim=obs_dim_cfg, render_mode=render_mode)
    obs, _ = env.reset(seed=SEED)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    # --- Policy ---
    hidden = hyp.get("hidden_state_size") # if not in config, 64 is a reasonable eval default
    device = pick_device()
    print("Using device:", device)

    policy = GRUPolicy(obs_dim, act_dim, hidden_state_size=hidden).to(device)
    policy.load_state_dict(torch.load(CKPT, map_location=device))
    policy.eval()

    h = torch.zeros(1, 1, hidden, device=device)

    total_reward, steps = 0.0, 0
    with torch.no_grad():
        for t in range(STEPS):
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)

            out = policy(x, h)
            if len(out) == 3:
                mu, value, h = out
                # No log_std provided; use deterministic tanh(mu) to stay within bounds
                action_t = torch.tanh(mu) if APPLY_TANH_ON_MU else mu
            elif len(out) == 4:
                mu, log_std, value, h = out
                if DETERMINISTIC:
                    action_t = torch.tanh(mu) if APPLY_TANH_ON_MU else mu
                else:
                    # Sample tanh-Gaussian
                    log_std = torch.clamp(log_std, -5.0, 2.0)
                    std = log_std.exp()
                    z = mu + std * torch.randn_like(std)
                    action_t = torch.tanh(z) if APPLY_TANH_ON_MU else z
            else:
                raise RuntimeError("GRUPolicy forward must return (mu, value, h) or (mu, log_std, value, h)")

            action = action_t.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

            # Defensive clip to env bounds
            lo, hi = env.action_space.low, env.action_space.high
            action = np.clip(action, lo, hi)

            obs, r, term, trunc, info = env.step(action)
            total_reward += float(r)
            steps += 1

            if RENDER:
                env.render()
                # time.sleep(0.0)  # uncomment to slow down

            if term or trunc:
                break

    env.close()
    avg = total_reward / max(1, steps)
    print(f"Episode steps: {steps} | Total reward: {total_reward:.3f} | Avg/step: {avg:.3f}")


if __name__ == "__main__":
    main()
