"""
Train Meta-RL with recurrent PPO on MuJoCo env, mapping from your PyLoCo training flow.
"""
import json
import argparse
import math
import numpy as np
import torch
from metarl.envs.mujoco_quadruped_env import QuadrupedMujocoEnv
from metarl.policies.gru_policy import GRUPolicy
from metarl.algorithms.ppo_rnn import RecurrentPPO, PPOConfig
from metarl.utils.rollout_buffer import RecurrentRolloutBuffer

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class DiagGaussian:
    """Minimal diagonal Gaussian with tanh squashing + logprob."""
    def __init__(self, mean, log_std):
        # mean/log_std: (..., act_dim)
        self.mean = mean
        self.log_std = log_std.clamp(-5, 2)  # keep std sane
        self.std = torch.exp(self.log_std)

    def sample(self):
        eps = torch.randn_like(self.mean)
        u = self.mean + self.std * eps
        a = torch.tanh(u)                    # squash to [-1, 1]
        # logprob with tanh correction (Sum over action dims)
        # logN(u; mean, std) - sum(log(1 - tanh(u)^2))
        logp_u = -0.5 * (((u - self.mean) / (self.std + 1e-8))**2 + 2*self.log_std + math.log(2*math.pi)).sum(dim=-1)
        logp = logp_u - torch.log(1 - a.pow(2) + 1e-8).sum(dim=-1)
        return a, logp

    def deterministic(self):
        a = torch.tanh(self.mean)
        logp = None
        return a, logp

def main(args):
    with open(args.config, "r") as f:
        params = json.load(f)
    envp = params["environment_params"]
    rew = params["reward_params"]
    hyp = params["train_hyp_params"]

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

    device = pick_device()
    policy = GRUPolicy(obs_dim, act_dim, hidden_state_size=16).to(device)
    policy.train()

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

    # Action std can live in config; fallback to 0.3
    action_log_std = torch.ones(act_dim, device=device) * float(hyp.get("action_log_std", -1.2040))  # exp(-1.204)=0.3


    steps_per_update = hyp["n_steps"]
    total_steps = hyp["time_steps"]
    buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim)


    # init env + hidden
    obs, _ = env.reset(seed=hyp.get("random_seed", 42))
    h = torch.zeros(1, 1, 16, device=device)  # (num_layers=1, batch=1, hidden)

    ep_step = 0
    for t in range(total_steps):
        x = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)

        with torch.no_grad():
            # --- Forward policy ---
            # Expecting GRUPolicy to return either:
            #   (mu, value, next_h)   OR   (mu, log_std, value, next_h)
            out = policy(x, h)
            if len(out) == 3:
                mu, value, h = out
                log_std = action_log_std.expand_as(mu)
            elif len(out) == 4:
                mu, log_std, value, h = out
            else:
                raise RuntimeError("GRUPolicy forward must return (mu, value, h) or (mu, log_std, value, h)")

            # --- Build distribution & sample action ---
            dist = DiagGaussian(mu.squeeze(0).squeeze(0), log_std)  # (act_dim,)
            action_t, logp_t = dist.sample()  # stochastic; use .deterministic() for eval

            # clamp just in case (tanh already in [-1,1])
            action_np = action_t.squeeze().cpu().numpy().astype(np.float32)

            v_t = value.squeeze().cpu().item()
            logp_item = logp_t.cpu().item()

        # --- Step environment ---
        next_obs, r, term, trunc, info = env.step(action_np)
        done = bool(term or trunc)

        # --- Store transition ---
        buf.add(obs, action_np, float(r), done, float(logp_item), float(v_t))

        obs = next_obs
        ep_step += 1

        # --- Episode reset handling (very important for RNN) ---
        if done:
            obs, _ = env.reset()
            h.zero_()       # reset recurrent state at episode boundary
            ep_step = 0

        # --- PPO update ---
        if (t + 1) % steps_per_update == 0:
            # If you have GAE(lambda), also pass lambda from config; here minimal
            buf.compute_returns_advantages(gamma=hyp["gamma"])
            algo.update(buf)  # uses policy & optimizer internally
            buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim)

            if (t + 1) % steps_per_update == 0:
                print(f"Trained up to step {t+1}")

    torch.save(policy.state_dict(), "checkpoints/policy_final.pt")
    print("✅ Saved trained policy to checkpoints/policy_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/metarl_default.json")
    args = parser.parse_args()
    main(args)
