"""
Train Meta-RL with recurrent PPO on MuJoCo env, mapping from your PyLoCo training flow.
"""
import json
import argparse
import os
import torch
import numpy as np
import torch.distributions as D
from metarl.envs.mujoco_cheetah_env import HalfCheetahAdapter
from metarl.policies.gru_policy import GRUPolicy
from metarl.algorithms.ppo_rnn import RecurrentPPO, PPOConfig
from metarl.utils.rollout_buffer import RecurrentRolloutBuffer

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class TanhDiagGaussian:
    """Tanh‑squashed diagonal Gaussian with change‑of‑variables log‑prob.

    Given *unsquashed* parameters (mean, log_std), we sample z ~ N(mean, std),
    then a = tanh(z) ∈ (−1, 1). To compute log π(a), we correct with
    log|det J| of the tanh transform: 

        log π(a) = log N(z; mean, std) − Σ log(1 − tanh(z)^2)

    evaluated at the sampled z that maps to a via tanh.
    """
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = torch.clamp(log_std, -5.0, 2.0)
        self.std = self.log_std.exp()
        self.base = D.Normal(self.mean, self.std)

    def rsample(self):
        z = self.base.rsample()            # reparameterized sample
        a = torch.tanh(z)
        logp = self.base.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, z, logp.sum(-1)
    

def main(args):
    with open(args.config, "r") as f:
        params = json.load(f)
    envp = params["environment_params"]
    hyp = params["train_hyp_params"]

    device = pick_device()
    print(f"Using device: {device}")

    # --- Env ---
    obs_dim_opt = envp.get("obs_dim")
    render_mode = envp.get("render_mode")
    env = HalfCheetahAdapter(obs_dim=obs_dim_opt, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # --- Policy ---
    hidden_size = hyp.get("hidden_state_size")
    policy = GRUPolicy(obs_dim, act_dim, hidden_state_size=hidden_size).to(device)
    policy.train()

    # If your GRUPolicy does not own its optimizer, RecurrentPPO should; otherwise pass one in.
    algo = RecurrentPPO(policy, PPOConfig(
        gamma=float(hyp["gamma"]),
        learning_rate=float(hyp["learning_rate"]),
        clip_range=float(hyp["clip_range"]),
        entropy_coef=float(hyp["entropy_coef"]),
        value_coef=float(hyp["value_coef"]),
        update_epochs=int(hyp["update_epochs"]),
        sequence_length=int(hyp["sequence_length"]),
        meta_episode_length=int(hyp.get("meta_episode_length", 0)),
        batch_size=int(hyp["batch_size"]),
    ))

    # Global/default log‑std (policy may override by returning its own log_std)
    action_log_std = torch.ones(act_dim, device=device) * float(hyp.get("action_log_std", -1.2040))

    # --- Rollout storage ---
    steps_per_update = int(hyp["n_steps"])  # transitions collected per update
    total_steps = int(hyp["time_steps"])    # total env steps
    buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim, hidden_size)

    # --- Seeding & init ---
    torch.manual_seed(int(hyp.get("random_seed", 42)))
    np.random.seed(int(hyp.get("random_seed", 42)))
    obs, _ = env.reset(seed=int(hyp.get("random_seed", 42)))

    # RNN hidden state: (num_layers=1, batch=1, hidden)
    h = torch.zeros(1, 1, hidden_size, device=device)

    ep_return, ep_len, ep_count = 0.0, 0, 0

    # --------------
    # Training loop
    # --------------
    for t in range(total_steps):
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)

        with torch.no_grad():
            out = policy(x, h)
            if len(out) == 3:
                mu, value, h = out
                log_std = action_log_std.expand_as(mu)
            elif len(out) == 4:
                mu, log_std, value, h = out
            else:
                raise RuntimeError("GRUPolicy forward must return (mu, value, h) or (mu, log_std, value, h)")

            # Build tanh‑squashed Gaussian in action space
            # mu/log_std expected shaped [1, 1, act_dim]
            mu_flat = mu.squeeze(0).squeeze(0)
            log_std_flat = log_std.squeeze(0).squeeze(0)
            dist = TanhDiagGaussian(mu_flat, log_std_flat)
            a_t, pre_tanh, logp_t = dist.rsample()
            action_np = a_t.detach().cpu().numpy().astype(np.float32)

            v_t = value.squeeze().detach().cpu().item()
            logp_item = logp_t.detach().cpu().item()

        # Step env
        next_obs, r, term, trunc, info = env.step(action_np)
        done = bool(term or trunc)

        # Store
        buf.add(obs, action_np, float(r), done, float(logp_item), float(v_t))

        # Book‑keeping
        ep_return += float(r)
        ep_len += 1

        obs = next_obs

        # Episode handling (reset env & hidden)
        if done:
            ep_count += 1
            if ep_count % max(1, hyp.get("log_interval")) == 0:
                print(f"Episode {ep_count}: return={ep_return:.2f} length={ep_len}")
            obs, _ = env.reset()
            h.zero_()
            ep_return, ep_len = 0.0, 0

        # PPO update
        if (t + 1) % steps_per_update == 0:
            buf.compute_returns_advantages(gamma=float(hyp["gamma"]))
            algo.update(buf)
            buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim, hidden_size)
            print(f"Trained up to step {t + 1}")


    # ---- Save final policy ----
    torch.save(policy.state_dict(), "checkpoints/cheetah/policy_final.pt")
    print(f"✅ Saved trained policy to checkpoints/cheetah/policy_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/metarl_default.json")
    args = parser.parse_args()
    main(args)
