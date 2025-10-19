import os, torch, numpy as np
from metarl.policies.gru_policy import GRUPolicy
from metarl.envs.mujoco_quadruped_env import QuadrupedMujocoEnv
import json

XML = "models/a1/a1.xml"              # adjust if needed
CKPT = "checkpoints/policy_old.pt"  # make sure this exists
CFG = "configs/metarl_default.json"
STEPS = 500
RENDER = False                        # set True if you wired env.render()

def pick_device(prefer=None):
    prefer = (prefer or "").lower()
    if prefer == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


with open(CFG, "r") as f:
    _cfg = json.load(f)
REWARD_PARAMS = _cfg["reward_params"]

env = QuadrupedMujocoEnv(model_xml=XML, reward_params=REWARD_PARAMS)
obs, _ = env.reset(seed=0)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(f"obs_dim={obs_dim}, act_dim={act_dim}")

device = pick_device()   # "mps" on M1, "cuda" on cluster, else cpu
print("Using device:", device)

policy = GRUPolicy(obs_dim, act_dim, hidden_state_size=16).to(device)
policy.load_state_dict(torch.load(CKPT, map_location=device))
policy.eval()

h = torch.zeros(1, 1, 16, device=device)
total_reward, steps = 0.0, 0

with torch.no_grad():
    for t in range(STEPS):
        x = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
        mu, _, h = policy(x, h)
        action = mu.squeeze(0).squeeze(0).cpu().numpy()
        obs, r, done, trunc, info = env.step(action)
        total_reward += r
        steps += 1
        if RENDER:
            env.render()
        if done or trunc:
            break

print(f"Episode steps: {steps}  |  Total reward: {total_reward:.3f}  |  Avg/step: {total_reward/steps:.3f}")