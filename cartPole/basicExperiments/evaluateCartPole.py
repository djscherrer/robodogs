from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import os
import numpy as np
import torch
import gymnasium as gym

from cartPole.basicExperiments.cartPoleEnv import make_evaluate_env
from cartPole.basicExperiments.cartPoleAgent import Agent

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

@torch.no_grad()
def eval_one_config(
    agent: Agent,
    env_id: str,
    device: torch.device | str,
    episodes: int,
    cfg: Optional[EnvConfig] = None,
    video_dir: Optional[str] = None,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate on a single (optionally provided) env config for `episodes` episodes.
    If `cfg` is None, uses the env's default physics.
    """
    env = make_evaluate_env(env_id, video_dir=video_dir, seed=seed)
    ep_returns, ep_lengths = [], []
    rs = np.random.RandomState(seed)

    for ep in range(episodes):
        # Apply config (fixed over the whole episode)
        if cfg is not None:
            apply_config(env, cfg)

        obs, _ = env.reset(seed=int(rs.randint(0, 2**31 - 1)))
        done = False
        ret, length = 0.0, 0

        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = agent.actor(x)
            action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ret += reward
            length += 1
            if verbose:
                print(f"\r[eval] ep={ep} step={length} return={ret:.1f}", end="")

        ep_returns.append(ret)
        ep_lengths.append(length)
        if verbose:
            print(f"\n[eval] ep={ep} cfg={return_config(env)} return={ret:.1f} length={length}")

    env.close()
    return np.array(ep_returns, dtype=np.float32), np.array(ep_lengths, dtype=np.int32)


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
    agent: Agent,
    env_id: str,
    device: torch.device | str,
    episodes_per_scenario: int = 5,
    video_root: str | None = None,
    seed: int = 0,
) -> list[dict]:
    rows = []
    for name, cfg in fixed_scenarios():
        vdir = f"{video_root}/{name}" if video_root else None
        rets, lens = eval_one_config(agent, env_id, device, episodes_per_scenario, cfg, vdir, seed)
        rows.append({
            "scenario": name,
            "length": cfg.length, "masspole": cfg.masspole, "masscart": cfg.masscart,
            "return_mean": float(rets.mean()), "return_std": float(rets.std(ddof=1) if len(rets)>1 else 0.0),
            "len_mean": float(lens.mean()),   "len_std": float(lens.std(ddof=1) if len(lens)>1 else 0.0),
        })
    return rows

def evaluate_on_random_configs(
    agent: Agent,
    env_id: str,
    device: torch.device | str,
    n_configs: int,
    episodes_per_config: int = 5,
    video_root: str | None = None,
    seed: int = 0,
) -> list[dict]:
    rows = []
    cfgs = sample_configs(n_configs, seed)
    for i, cfg in enumerate(cfgs):
        vdir = f"{video_root}/rand_{i:03d}" if video_root else None
        rets, lens = eval_one_config(agent, env_id, device, episodes_per_config, cfg, vdir, seed + i + 1)
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