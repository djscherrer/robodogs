# evaluate_jita.py
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import gymnasium as gym
from . import cheetahEnv  # expects CheetahCustom + make_evaluate_env


# -------------------------
# Embodiment config
# -------------------------
@dataclass(frozen=True)
class CheetahEmbodiment:
    # Torso
    torso_len_scale: Optional[float] = None
    torso_rad_scale: Optional[float] = None
    torso_mass_scale: Optional[float] = None
    torso_scale: Optional[float] = None  # uniform size if *_len/_rad not set
    # Front leg segments
    fthigh_len_scale: Optional[float] = None
    fthigh_rad_scale: Optional[float] = None
    fthigh_mass_scale: Optional[float] = None
    fshin_len_scale: Optional[float] = None
    fshin_rad_scale: Optional[float] = None
    fshin_mass_scale: Optional[float] = None
    ffoot_len_scale: Optional[float] = None
    ffoot_rad_scale: Optional[float] = None

    ffoot_mass_scale: Optional[float] = None
    # Back leg segments
    bthigh_len_scale: Optional[float] = None
    bthigh_rad_scale: Optional[float] = None
    bthigh_mass_scale: Optional[float] = None
    bshin_len_scale: Optional[float] = None
    bshin_rad_scale: Optional[float] = None
    bshin_mass_scale: Optional[float] = None
    bfoot_len_scale: Optional[float] = None
    bfoot_rad_scale: Optional[float] = None
    bfoot_mass_scale: Optional[float] = None
    # Leg-level shortcuts (applied if seg-specific is None)
    fleg_len_scale: Optional[float] = None
    fleg_rad_scale: Optional[float] = None
    fleg_mass_scale: Optional[float] = None
    bleg_len_scale: Optional[float] = None
    bleg_rad_scale: Optional[float] = None
    bleg_mass_scale: Optional[float] = None
    # Legacy fallbacks (lowest precedence)
    thigh_scale: Optional[float] = None
    shin_scale: Optional[float] = None
    foot_scale: Optional[float] = None
    front_leg_scale: Optional[float] = None
    back_leg_scale: Optional[float] = None


def _apply_embodiment(env: gym.Env, cfg: CheetahEmbodiment) -> None:
    """Forward non-None fields to CheetahCustom.set_morphology and apply once."""
    base = env.unwrapped
    kv = {k: v for k, v in vars(cfg).items() if v is not None}
    if kv:
        base.set_morphology(**kv)
        base._apply_morphology()


def make_vector_eval_env(
    env_id: str,
    num_envs: int,
    seed: int,
    video_dir: Optional[str] = None,
    cfg: Optional[CheetahEmbodiment] = None,
    proxy_period_steps: int = 32,
    proxy_training_steps: int = 128,
    proxy_amplitude: float = 0.10,
):    
    def thunk(i):
        def _make():
            env = cheetahEnv.make_evaluate_env(
                env_id,
                video_dir=(f"{video_dir}/env{i}" if (video_dir is not None and i==0)else None), # NOTE: Only capture video on env0
                seed=seed + i,
                proxy_period_steps=proxy_period_steps,
                proxy_training_steps=proxy_training_steps,
                proxy_amplitude=proxy_amplitude,
            )
            if cfg is not None:
                _apply_embodiment(env, cfg)   # apply scenario morphology right here
            return env
        return _make
    return gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])


# -------------------------
# Vectorized evaluation
# -------------------------
@torch.no_grad()
def eval_one_config_vector(
    agent,
    env_id: str,
    device: torch.device | str,
    episodes: int,                         # total finished episodes to collect
    cfg: Optional[CheetahEmbodiment] = None,
    reset_after_proxy: bool = False,
    video_dir: Optional[str] = None,
    seed: int = 0,
    num_envs: int = 8,
    verbose: bool = False,
    proxy_period_steps: int = 32,
    proxy_training_steps: int = 128,
    proxy_amplitude: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    envs = make_vector_eval_env(env_id, num_envs, seed, video_dir, cfg, proxy_period_steps, proxy_training_steps, proxy_amplitude, reset_after_proxy=reset_after_proxy)
    agent_device = next(agent.parameters()).device if isinstance(agent, torch.nn.Module) else torch.device(device)

    obs, _ = envs.reset(seed=seed)

    # recurrent-safe
    h_a = h_c = None
    hidden = getattr(agent, "hidden_size", 128)

    ret = np.zeros(num_envs, dtype=np.float32)
    ret_proxy = np.zeros(num_envs, dtype=np.float32)
    ret_real = np.zeros(num_envs, dtype=np.float32)
    length = np.zeros(num_envs, dtype=np.int32)
    done_R: List[float] = []
    done_R_proxy : List[float] = []
    done_R_real : List[float] = []
    done_L: List[int] = []

    while len(done_R) < episodes:
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent_device)
        try:
            if h_a is None:
                h_a = torch.zeros(1, num_envs, hidden, device=agent_device)
                h_c = torch.zeros(1, num_envs, hidden, device=agent_device)
            action, h_a, _, _, _, h_c = agent.get_action_and_value(x, h_actor=h_a, h_critic=h_c, action=None)
        except TypeError:
            action, _, _, _ = agent.get_action_and_value(x, action=None)

        a = action.detach().cpu().numpy()
        obs, reward, term, trunc, info = envs.step(a)
        done = np.logical_or(term, trunc)

        ret += reward.astype(np.float32)
        if "current_proxy_reward" in info:
            ret_proxy += info["current_proxy_reward"].astype(np.float32)
        if "current_real_reward" in info:
            ret_real += info["current_real_reward"].astype(np.float32)
        length += 1

        if done.any():
            if h_a is not None:
                mask = torch.from_numpy(done).to(agent_device, dtype=torch.bool)
                h_a[:, mask] = 0
                h_c[:, mask] = 0
            for i in np.where(done)[0]:
                done_R.append(float(ret[i]))
                done_L.append(int(length[i]))
                done_R_proxy.append(float(ret_proxy[i]))
                done_R_real.append(float(ret_real[i]))
                ret[i] = 0.0
                ret_proxy[i] = 0.0
                ret_real[i] = 0.0
                length[i] = 0

        if verbose and len(done_R) % max(1, (episodes // 10)) == 0 and len(done_R) > 0:
            print(f"\r[eval] {len(done_R)}/{episodes} episodes, mean_return={np.mean(done_R):.1f}", end="")

    envs.close()
    return np.array(done_R[:episodes], dtype=np.float32), np.array(done_L[:episodes], dtype=np.int32), np.array(done_R_proxy[:episodes], dtype=np.float32), np.array(done_R_real[:episodes], dtype=np.float32)


# -------------------------
# Fixed embodiments (suite)
# -------------------------
def fixed_scenarios() -> List[tuple[str, CheetahEmbodiment]]:
    return [
        ("baseline",            CheetahEmbodiment()),
        ("torso_uniform_big",   CheetahEmbodiment(torso_scale=1.3, torso_mass_scale=1.3)),
        ("torso_longer",        CheetahEmbodiment(torso_len_scale=1.3, torso_rad_scale=1.20)),
        ("torso_shorter",       CheetahEmbodiment(torso_len_scale=0.7, torso_rad_scale=0.8)),
        ("torso_heavier",       CheetahEmbodiment(torso_mass_scale=1.3)),
        ("torso_lighter",       CheetahEmbodiment(torso_mass_scale=0.7)),

        ("front_legs_long",     CheetahEmbodiment(fleg_len_scale=1.3)),
        ("back_legs_long",      CheetahEmbodiment(bleg_len_scale=1.3)),
        ("front_legs_heavy",    CheetahEmbodiment(fleg_mass_scale=1.3)),
        ("back_legs_heavy",     CheetahEmbodiment(bleg_mass_scale=1.3)),
        ("thin_legs",           CheetahEmbodiment(fleg_rad_scale=0.75, bleg_rad_scale=0.75)),
        ("chunky_feet",         CheetahEmbodiment(ffoot_rad_scale=1.3, bfoot_rad_scale=1.3)),
        ("long_shins_only",     CheetahEmbodiment(fshin_len_scale=1.3, bshin_len_scale=1.3)),
        
        ("asym_longF_shortB",   CheetahEmbodiment(fleg_len_scale=1.2, bleg_len_scale=0.8)),
        ("asym_heavyF_lightB",  CheetahEmbodiment(fleg_mass_scale=1.3, bleg_mass_scale=0.8)),
        ("front_long_heavy",    CheetahEmbodiment(fleg_len_scale=1.3, fleg_mass_scale=1.3)),
        ("back_long_heavy",     CheetahEmbodiment(bleg_len_scale=1.3, bleg_mass_scale=1.3)),
    ]


def evaluate_on_fixed_scenarios(
    agent,
    env_id: str,
    device: torch.device | str,
    episodes_per_scenario: int = 8,
    video_root: Optional[str] = None,
    seed: int = 0,
    num_envs: int = 8,
    eval_tag: Optional[str] = None,
    isTargetTask: bool = True,
    # proxy task params
    proxy_period_steps: int = 32,
    proxy_training_steps: int = 128,
    proxy_amplitude: float = 0.10,
    reset_after_proxy: bool = False,
) -> List[dict]:
    rows = []
    for name, cfg in fixed_scenarios():
        print("Evaluating scenario:", name)
        vdir = f"{video_root}/{name}/{eval_tag}" if video_root else None
        rets, lens, rets_proxy, rets_real = eval_one_config_vector(
            agent, env_id, device,
            episodes=episodes_per_scenario,
            cfg=cfg, video_dir=vdir, seed=seed, num_envs=num_envs, 
            proxy_period_steps=proxy_period_steps,
            proxy_training_steps=proxy_training_steps,
            proxy_amplitude=proxy_amplitude,
            reset_after_proxy=reset_after_proxy,
        )
        row = {
            "scenario": name,
            "return_mean": float(rets.mean()),
            "return_std": float(rets.std(ddof=1) if len(rets) > 1 else 0.0),
            "len_mean": float(lens.mean()),
            "len_std": float(lens.std(ddof=1) if len(lens) > 1 else 0.0),
            "proxy_return_mean": float(rets_proxy.mean()),
            "proxy_return_std": float(rets_proxy.std(ddof=1) if len(rets_proxy) > 1 else 0.0),
            "real_return_mean": float(rets_real.mean()),
            "real_return_std": float(rets_real.std(ddof=1) if len(rets_real) > 1 else 0.0),
        }
        # store exact embodiment used (handy for CSV/W&B)
        row.update({k: (getattr(cfg, k) if getattr(cfg, k) is not None else None) for k in vars(cfg)})
        rows.append(row)
    return rows


# -------------------------
# Random embodiments
# -------------------------
def _nrm(rng, mu=1.0, sd=0.15, lo=0.7, hi=1.4) -> float:
    return float(np.clip(rng.normal(mu, sd), lo, hi))

def sample_embodiment(rng: np.random.RandomState) -> CheetahEmbodiment:
    # Conservative, embodiment-only jitter around 1.0
    return CheetahEmbodiment(
        torso_len_scale=_nrm(rng, sd=0.12, lo=0.8, hi=1.25),
        torso_rad_scale=_nrm(rng, sd=0.10, lo=0.85, hi=1.20),
        torso_mass_scale=_nrm(rng, sd=0.20, lo=0.7, hi=1.4),
        fleg_len_scale=_nrm(rng, sd=0.18, lo=0.75, hi=1.3),
        bleg_len_scale=_nrm(rng, sd=0.18, lo=0.75, hi=1.3),
        fleg_rad_scale=_nrm(rng, sd=0.12, lo=0.8, hi=1.2),
        bleg_rad_scale=_nrm(rng, sd=0.12, lo=0.8, hi=1.2),
        fleg_mass_scale=_nrm(rng, sd=0.20, lo=0.7, hi=1.4),
        bleg_mass_scale=_nrm(rng, sd=0.20, lo=0.7, hi=1.4),
    )

def sample_embodiments(n: int, seed: int) -> List[CheetahEmbodiment]:
    rng = np.random.RandomState(seed)
    return [sample_embodiment(rng) for _ in range(n)]

def evaluate_on_random_configs(
    agent,
    env_id: str,
    device: torch.device | str,
    n_configs: int,
    episodes_per_config: int = 8,
    video_root: Optional[str] = None,
    seed: int = 0,
    num_envs: int = 8,
    # proxy task params
    proxy_period_steps: int = 32,
    proxy_training_steps: int = 128,
    proxy_amplitude: float = 0.10,
) -> List[dict]:
    rows = []
    cfgs = sample_embodiments(n_configs, seed)
    for i, cfg in enumerate(cfgs):
        name = f"rand_{i:03d}"
        vdir = f"{video_root}/{name}" if video_root else None
        rets, lens, rets_proxy, rets_real = eval_one_config_vector(
            agent, env_id, device,
            episodes=episodes_per_config,
            cfg=cfg, video_dir=vdir, seed=seed + i + 1, num_envs=num_envs,
            proxy_period_steps=proxy_period_steps,
            proxy_training_steps=proxy_training_steps,
            proxy_amplitude=proxy_amplitude,
        )
        row = {
            "scenario": name,
            "return_mean": float(rets.mean()),
            "return_std": float(rets.std(ddof=1) if len(rets) > 1 else 0.0),
            "len_mean": float(lens.mean()),
            "len_std": float(lens.std(ddof=1) if len(lens) > 1 else 0.0),
        }
        row.update({k: (getattr(cfg, k) if getattr(cfg, k) is not None else None) for k in vars(cfg)})
        rows.append(row)
    return rows


# -------------------------
# CLI (optional)
# -------------------------
if __name__ == "__main__":
    import argparse
    from basicExperiments.halfCheetah.cheetahAgent import load_agent_from_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes-per-scenario", type=int, default=8)
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-random", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    agent = load_agent_from_checkpoint(args.ckpt, device)

    print("Evaluating on fixed scenarios...")
    fixed_rows = evaluate_on_fixed_scenarios(
        agent, args.env_id, device,
        episodes_per_scenario=args.episodes_per_scenario,
        video_root=args.video_root,
        seed=args.seed,
        eval_tag="manual_eval"
    )
    for r in fixed_rows:
        print(r)

    print("\nEvaluating on random scenarios...")
    rand_rows = evaluate_on_random_configs(
        agent, args.env_id, device,
        n_configs=args.n_random,
        episodes_per_config=args.episodes_per_scenario,
        video_root=args.video_root,
        seed=args.seed + 1000,
    )
    for r in rand_rows:
        print(r)