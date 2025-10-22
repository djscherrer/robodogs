"""
Train Meta-RL with recurrent PPO on MuJoCo env, mapping from your PyLoCo training flow.
"""
import json
import argparse
import numpy as np
import torch
from metarl.envs.mujoco_quadruped_env import QuadrupedMujocoEnv
from metarl.policies.gru_policy import GRUPolicy
from metarl.algorithms.ppo_rnn import RecurrentPPO, PPOConfig
from metarl.utils.rollout_buffer import RecurrentRolloutBuffer

def pick_device(prefer=None):
    prefer = (prefer or "").lower()
    if prefer == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def main(args):
    with open(args.config, "r") as f:
        params = json.load(f)
    envp = params["environment_params"]
    rew = params["reward_params"]
    hyp = params["train_hyp_params"]

    device = pick_device()
    print("Using device:", device)

    env = QuadrupedMujocoEnv(
        model_xml=envp["model_xml"],
        control_frequency_hz=envp["control_frequency_hz"],
        pd_frequency_hz=envp["pd_frequency_hz"],
        episode_steps=envp["episode_steps"],
        terminate_on_fall=envp["terminate_on_fall"],
        rsi=envp["rsi"],
        reward_params=rew,
        obs_include=envp["obs_include"]
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GRUPolicy(obs_dim, act_dim, hidden_state_size=16).to(device)
    algo = RecurrentPPO(policy, PPOConfig(
        gamma=hyp["gamma"],
        learning_rate=hyp["learning_rate"],
        clip_range=hyp["clip_range"],
        entropy_coef=hyp["entropy_coef"],
        value_coef=hyp["value_coef"],
        update_epochs=hyp["update_epochs"],
        sequence_length=hyp["sequence_length"],
        meta_episode_length=hyp["meta_episode_length"],
        batch_size=hyp["batch_size"]
    ))

    steps_per_update = hyp["n_steps"]
    total_steps = hyp["time_steps"]
    buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim)
    obs, _ = env.reset(seed=hyp.get("random_seed",42))
    hidden_size = policy.hidden_state_size 
    h_state = torch.zeros(1, 1, hidden_size, device=device)
    last_done = False

    for t in range(total_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mu, value_tensor, next_h_state = policy(obs_tensor, h_state)
            std = policy.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action_tensor = dist.sample()
            logp_tensor = dist.log_prob(action_tensor).sum(-1)
        action = action_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        value = value_tensor.squeeze().detach().cpu().item()
        logp = logp_tensor.squeeze().detach().cpu().item()
        next_obs, rew, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        buf.add(obs, action, rew, done, logp, value)
        h_state = next_h_state.detach()
        if done:
            obs, _ = env.reset()
            h_state = torch.zeros_like(h_state)
        else:
            obs = next_obs
        last_done = done
        if (t+1) % steps_per_update == 0:
            with torch.no_grad():
                if last_done:
                    bootstrap_value = 0.0
                else:
                    bootstrap_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    _, value_bootstrap, _ = policy(bootstrap_obs, h_state)
                    bootstrap_value = value_bootstrap.squeeze().detach().cpu().item()
            buf.compute_returns_advantages(gamma=hyp["gamma"], last_value=bootstrap_value)
            algo.update(buf)
            buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim)
            if (t+1) % hyp.get("n_steps", steps_per_update) == 0:
                print(f"Trained up to step {t+1}")

    torch.save(policy.state_dict(), "checkpoints/policy_final.pt")
    print("✅ Saved trained policy to checkpoints/policy_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/metarl_default.json")
    args = parser.parse_args()
    main(args)
