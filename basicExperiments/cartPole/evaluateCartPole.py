from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import os
import numpy as np
import torch
import gymnasium as gym


from cartPole.cartPoleEnv import make_evaluate_env
from cartPole.cartPoleAgent import Agent

@dataclass(frozen=True)
class EnvConfig:
    length: float
    masspole: float
    masscart: float
    gravity: float = 9.8
    force_mag: float = 10.0
    tau: float = 0.02

def apply_config(env: gym.Env, cfg: EnvConfig) -> None:
    base = env.unwrapped
    base.length = float(cfg.length)
    base.masspole = float(cfg.masspole)
    base.masscart = float(cfg.masscart)
    base.gravity = float(cfg.gravity)
    base.force_mag = float(cfg.force_mag)
    base.tau = float(cfg.tau)
    # derived terms
    base.polemass_length = base.masspole * base.length
    base.total_mass = base.masspole + base.masscart

def return_config(env: gym.Env) -> Dict[str, float]:
    base = env.unwrapped
    return {
        "length": base.length,
        "gravity": base.gravity,
        "masscart": base.masscart,
        "masspole": base.masspole,
        "force_mag": base.force_mag,
        "tau": base.tau,
        "max_episode_steps": base.spec.max_episode_steps,
    }

def make_vector_eval_env(env_id: str, num_envs: int, seed: int, video_dir: Optional[str] = None):
    # Build N single envs via your factory, each with its own seed offset
    def thunk(i):
        def _make():
            return make_evaluate_env(env_id, video_dir=(f"{video_dir}/env{i}" if video_dir else None), seed=seed + i)
        return _make
    return gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])

@torch.no_grad()
def eval_one_config(
    agent,
    env_id: str,
    device: torch.device | str,
    episodes: int,
    cfg: Optional[EnvConfig] = None,
    video_dir: Optional[str] = None,
    seed: int = 0,
    verbose: bool = False,
    greedy: bool = True,           # True = argmax, False = sample
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate on a single (optionally provided) env config for `episodes` episodes.
    Supports both feed-forward Agent and GRUAgent (recurrent).
    """
    env = make_evaluate_env(env_id, video_dir=video_dir, seed=seed)
    ep_returns, ep_lengths = [], []
    rs = np.random.RandomState(seed)

    # detect recurrent agent (has GRU modules or get_action_and_value with h_* args)
    is_recurrent = any(
        hasattr(agent, attr) for attr in ("gru_Actor", "gru_Critic")
    )

    # pick device for internal tensors
    agent_device = next(agent.parameters()).device if isinstance(agent, torch.nn.Module) else torch.device(device)

    # get hidden size if recurrent (fallback 128)
    hidden_size = getattr(agent, "hidden_size", 128)

    for ep in range(episodes):
        # reset env, THEN apply custom physics (safer if reset re-derives params)
        obs, _ = env.reset(seed=int(rs.randint(0, 2**31 - 1)))
        if cfg is not None:
            apply_config(env, cfg)

        done = False
        ret, length = 0.0, 0

        # init recurrent state per episode
        if is_recurrent:
            h_a = torch.zeros(1, 1, hidden_size, device=agent_device)
            h_c = torch.zeros(1, 1, hidden_size, device=agent_device)

        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=agent_device).unsqueeze(0)  # [1, obs]

            if is_recurrent:
                # --- Recurrent forward (greedy or stochastic) ---
                # Actor pass
                h_seq_a, h_a = Agent.gru_Actor(x.unsqueeze(1), h_a)                    # [1,1,H], [1,1,H]
                logits = Agent.mlp_Actor(torch.cat([h_seq_a.squeeze(1), x], dim=-1))   # [1, A]
                if greedy:
                    action = torch.argmax(logits, dim=-1)                               # [1]
                else:
                    action = Categorical(logits=logits).sample()                        # type: ignore # [1]

                # Critic pass (optional here, but cheap + consistent)
                h_seq_c, h_c = Agent.gru_Critic(x.unsqueeze(1), h_c)
                _ = Agent.mlp_Critic(torch.cat([h_seq_c.squeeze(1), x], dim=-1))       # [1, 1] (unused)

                a = int(action.item())

            else:
                # --- Feed-forward policy ---
                if hasattr(agent, "actor"):
                    logits = Agent.actor(x)                                             # [1, A]
                    a = int(torch.argmax(logits, dim=-1).item()) if greedy else int(Categorical(logits=logits).sample().item())
                else:
                    # fallback: use the generic API if present
                    a, *_ = Agent.get_action_and_value(x)
                    a = int(a.item())

            obs, reward, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            ret += reward
            length += 1

            if is_recurrent and done:
                # reset hidden states between episodes
                h_a.zero_()
                h_c.zero_()

            if verbose:
                print(f"\r[eval] ep={ep} step={length} return={ret:.1f}", end="")

        ep_returns.append(ret)
        ep_lengths.append(length)
        if verbose:
            print(f"\n[eval] ep={ep} cfg={return_config(env)} return={ret:.1f} length={length}")

    env.close()
    return np.array(ep_returns, dtype=np.float32), np.array(ep_lengths, dtype=np.int32)


@torch.no_grad()
def eval_one_config_vector(
    agent,
    env_id: str,
    device: torch.device | str,
    episodes: int,                    # total episodes to collect across all envs
    cfg: Optional[EnvConfig] = None,
    video_dir: Optional[str] = None,
    seed: int = 0,
    num_envs: int = 8,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation that supports both feed-forward and recurrent agents.
    Uses agent.get_action_and_value(...) only, handling both signatures:
      - MLP:   (x, action=None) -> action, logp, entropy, value
      - GRU:   (x, h_actor=None, h_critic=None, action=None) -> action, h_a, logp, ent, v, h_c
    """
    envs = make_vector_eval_env(env_id, num_envs, seed, video_dir)
    agent_device = next(agent.parameters()).device if isinstance(agent, torch.nn.Module) else torch.device(device)

    # Apply config to each sub-env after first reset (safer with some envs)
    obs, _ = envs.reset(seed=seed)
    if cfg is not None:
        for e in envs.envs:
            apply_config(e, cfg)

    # Recurrent state buffers (allocated lazily if agent is recurrent)
    h_a = h_c = None
    hidden_size = getattr(agent, "hidden_size", 128)

    # Per-env episode accumulators
    ret = np.zeros(num_envs, dtype=np.float32)
    length = np.zeros(num_envs, dtype=np.int32)

    done_ep_returns: List[float] = []
    done_ep_lengths: List[int] = []

    # Keep stepping until we collected requested number of episodes
    while len(done_ep_returns) < episodes:
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent_device)   # [B, obs]

        # Try recurrent signature first; fall back to MLP signature.
        try:
            if h_a is None:
                h_a = torch.zeros(1, num_envs, hidden_size, device=agent_device)
                h_c = torch.zeros(1, num_envs, hidden_size, device=agent_device)
            action, h_a, _, _, _, h_c = agent.get_action_and_value(x, h_actor=h_a, h_critic=h_c, action=None)
        except TypeError:
            # MLP agent
            action, _, _, _ = agent.get_action_and_value(x, action=None)

        a_np = action.detach().cpu().numpy()
        obs, reward, term, trunc, info = envs.step(a_np)
        done = np.logical_or(term, trunc)  # [B]

        # Update per-env accumulators
        ret += reward.astype(np.float32)
        length += 1

        # For each env that finished, record and reset counters
        if done.any():
            # Mask recurrent states to zero at boundaries
            if h_a is not None:
                done_mask = torch.from_numpy(done).to(agent_device, dtype=torch.bool)
                h_a[:, done_mask] = 0
                h_c[:, done_mask] = 0

            for i in np.where(done)[0]:
                done_ep_returns.append(float(ret[i]))
                done_ep_lengths.append(int(length[i]))
                ret[i] = 0.0
                length[i] = 0

        if verbose and len(done_ep_returns) % max(1, (episodes // 10)) == 0 and len(done_ep_returns) > 0:
            print(f"\r[eval] collected {len(done_ep_returns)}/{episodes} episodes. "
                  f"mean_return={np.mean(done_ep_returns):.1f}", end="")

    envs.close()
    return np.array(done_ep_returns[:episodes], dtype=np.float32), np.array(done_ep_lengths[:episodes], dtype=np.int32)

# Config Generators

def fixed_scenarios() -> List[tuple[str, EnvConfig]]:
    return [
        ("baseline",     EnvConfig(length=0.5, masspole=0.10, masscart=1.0)),
        ("long_pole",    EnvConfig(length=0.9, masspole=0.10, masscart=1.0)),
        ("heavy_pole",   EnvConfig(length=0.5, masspole=0.25, masscart=1.0)),
        ("heavy_cart",   EnvConfig(length=0.5, masspole=0.10, masscart=1.8)),
        ("light_setup",  EnvConfig(length=0.3, masspole=0.05, masscart=0.3)),
    ]

def sample_config(rng: np.random.RandomState) -> EnvConfig:
    length   = float(np.clip(rng.normal(0.5, 0.4), 0.10, 1.00))
    masspole = float(np.clip(rng.normal(0.10, 0.05), 0.01, 0.30))
    masscart = float(np.clip(rng.normal(1.00, 0.50), 0.10, 2.00))
    return EnvConfig(length=length, masspole=masspole, masscart=masscart)

def sample_configs(n: int, seed: int) -> list[EnvConfig]:
    rng = np.random.RandomState(seed)
    return [sample_config(rng) for _ in range(n)]


# Wrappers

def evaluate_on_fixed_scenarios(
    agent,
    env_id: str,
    device: torch.device | str,
    episodes_per_scenario: int = 8,
    video_root: str | None = None,
    seed: int = 0,
    num_envs: int = 8,
) -> list[dict]:
    rows = []
    for name, cfg in fixed_scenarios():
        vdir = f"{video_root}/{name}" if video_root else None
        rets, lens = eval_one_config_vector(
            agent, env_id, device,
            episodes=episodes_per_scenario,
            cfg=cfg, video_dir=vdir, seed=seed, num_envs=num_envs
        )
        rows.append({
            "scenario": name,
            "length": cfg.length, "masspole": cfg.masspole, "masscart": cfg.masscart,
            "return_mean": float(rets.mean()), "return_std": float(rets.std(ddof=1) if len(rets)>1 else 0.0),
            "len_mean": float(lens.mean()),   "len_std": float(lens.std(ddof=1) if len(lens)>1 else 0.0),
        })
    return rows

def evaluate_on_random_configs(
    agent,
    env_id: str,
    device: torch.device | str,
    n_configs: int,
    episodes_per_config: int = 8,
    video_root: str | None = None,
    seed: int = 0,
    num_envs: int = 8,
) -> list[dict]:
    rows = []
    cfgs = sample_configs(n_configs, seed)
    for i, cfg in enumerate(cfgs):
        vdir = f"{video_root}/rand_{i:03d}" if video_root else None
        rets, lens = eval_one_config_vector(
            agent, env_id, device,
            episodes=episodes_per_config,
            cfg=cfg, video_dir=vdir, seed=seed + i + 1, num_envs=num_envs
        )
        rows.append({
            "scenario": f"rand_{i:03d}",
            "length": cfg.length, "masspole": cfg.masspole, "masscart": cfg.masscart,
            "return_mean": float(rets.mean()), "return_std": float(rets.std(ddof=1) if len(rets)>1 else 0.0),
            "len_mean": float(lens.mean()),   "len_std": float(lens.std(ddof=1) if len(lens)>1 else 0.0),
        })
    return rows


# main
if __name__ == "__main__":
    import argparse
    from cartPole.basicExperiments.cartPoleAgent import load_agent_from_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to agent checkpoint (.pt) to load.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device to use (e.g., cpu, cuda:0).")
    parser.add_argument("--episodes-per-scenario", type=int, default=5,
                        help="Number of evaluation episodes per scenario.")
    parser.add_argument("--video-root", type=str, default=None,
                        help="If provided, root directory to save evaluation videos.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for evaluation.")
    args = parser.parse_args()

    device = torch.device(args.device)
    agent = load_agent_from_checkpoint(args.ckpt, device)

    print("Evaluating on fixed scenarios...")
    fixed_results = evaluate_on_fixed_scenarios(
        agent, args.env_id, device,
        episodes_per_scenario=args.episodes_per_scenario,
        video_root=args.video_root,
        seed=args.seed,
    )
    for row in fixed_results:
        print(row)

    print("\nEvaluating on random scenarios...")
    random_results = evaluate_on_random_configs(
        agent, args.env_id, device,
        n_configs=10,
        episodes_per_config=args.episodes_per_scenario,
        video_root=args.video_root,
        seed=args.seed + 1000,
    )
    for row in random_results:
        print(row)