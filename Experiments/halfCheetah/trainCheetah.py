# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import wandb

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from gymnasium.vector import AsyncVectorEnv
from gymnasium.envs.registration import register
from Experiments.halfCheetah import cheetahAgent, cheetahEnv, evaluateCheetah, trainingLogic


@dataclass
class Args:
    exp_name: str = "Recurrent Proxy + Main: 256s_p + 256s_m"
    """the name of this experiment"""
    agent_type: str = "gru" # gru or mlp
    """the type of agent to use"""
    seed: int = 2
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "robodogs-cheetah"
    """the wandb's project name"""
    wandb_entity: str = "robodogs"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Environment Randomization
    randomize_morphology_every: int = 5
    """If >0, randomize morphology every N episodes"""
    morphology_jitter: float = 0.5
    """The amount of jitter to apply when randomizing morphology"""

    # Evaluation settings
    eval_every: int = 8
    """If >0, evaluate the agent every N updates (default: no evaluation)"""
    eval_episodes: int = 2
    """The number of episodes to run during each evaluation phase"""
    eval_num_envs: int = 8  
    """The number of parallel envs to use during evaluation"""
    eval_capture_video: bool = True
    """Whether to capture videos during evaluation"""

    # Proxy Episode Switch
    proxy_only_timesteps: int = 0
    """Train only on proxy reward until this many env steps are collected; afterwards, pure main-task training."""
    
    # Proxy task settings training
    proxy_training_steps: int = 64 #!TODO: change for runs
    """Number of steps over which to train on proxy task before switching to main task"""
    proxy_steps_per_period: int = 128
    """Number of steps for one full sine wave period"""
    proxy_amplitude: float = 0.2
    """Amplitude of the proxy sine wave relative to inital torso height"""
    proxy_track_weight: float = 2.0
    """Weight of the proxy tracking reward"""
    proxy_vel_penalty_weight: float = 0.2
    """Weight of the proxy velocity penalty"""
    reset_after_proxy: bool = False
    """Whether to reset the environment after the proxy task phase, this will also allow to continue if proxy task fails"""

    # Proxy task settings evaluation
    eval_proxy_training_steps: int = 64 #!TODO: change for runs
    """Number of steps over which to train on proxy task before switching to main task during evaluation"""
    eval_proxy_steps_per_period: int = 128
    """Number of steps for one full sine wave period during evaluation"""
    eval_proxy_amplitude: float = 0.2
    """Amplitude of the proxy sine wave during evaluation"""
    eval_reset_after_proxy: bool = True
    """Whether to reset the environment after the proxy task phase during evaluation"""
    # Algorithm specific arguments
    env_id: str = "Cheetah"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512 #!TODO: change for runs
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.01
    """the target KL divergence threshold"""
    gru_hidden_size: int = 128
    """the hidden size of the GRU"""
    mlp_hidden_size: int = 128
    """the hidden size of the MLP"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Checkpointing
    resume: Optional[str] = None
    """Path to a checkpoint (.pt) to resume from (e.g., checkpoints/NAME/last.pt)."""
    save_every_episodes: int = 0
    """If >0, also save 'last.pt' every N completed episodes (in addition to best)."""
    global_ckpt_dir = f"checkpoints/halfCheetah"


def return_config(env: gym.Env):
    sp = env.unwrapped.observation_space
    ac = env.unwrapped.action_space
    #TODO: replace with cheetahEnv embodiment variables (leg lengths, etc.)
    return {
        "obs_shape": tuple(sp.shape),
        "act_shape": tuple(ac.shape),
        "act_low": float(np.min(ac.low)),
        "act_high": float(np.max(ac.high)),
        "max_episode_steps": env.spec.max_episode_steps if env.spec else None,
    }

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("cpu") #NOTE: change to mps if RL stuff gets large
    return torch.device("cpu")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.agent_type}__{args.exp_name}__{args.seed}__{int(time.time())}"

    register(
        id=args.env_id,
        entry_point=cheetahEnv.CheetahCustom,
        max_episode_steps=500,
    )

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,   # can be None on personal account
            config=vars(args),
            name=run_name,
            save_code=True,             # snapshot of your code
            settings=wandb.Settings(start_method="thread"),  # robust on macOS
            mode="online",
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = pick_device()
    print(f"Using device: {device}")

    # env setup
    env_fns = [
        cheetahEnv.make_env(
            args.env_id, i, args.capture_video, run_name,
            args.proxy_steps_per_period, args.proxy_training_steps,
            args.proxy_amplitude,
            args.randomize_morphology_every, args.morphology_jitter,
            reset_after_proxy=args.reset_after_proxy,
            proxy_track_weight=args.proxy_track_weight,
            proxy_vel_penalty_weight=args.proxy_vel_penalty_weight,
        )
        for i in range(args.num_envs)
    ]

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    # Default: mac -> sync; linux+cuda -> async; otherwise sync
    prefer = "sync" if has_mps else ("async" if has_cuda else "sync")
    if prefer == "async":
        envs = AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)

    # Query attributes from worker 0
    sp = envs.single_observation_space
    ac = envs.single_action_space

    # cfg logging
    specs       = envs.get_attr("spec")
    frame_skips = envs.get_attr("frame_skip")
    dts         = envs.get_attr("dt")

    spec0       = specs[0]
    frame_skip0 = frame_skips[0] if frame_skips else None
    dt0         = dts[0] if dts else None

    cfg = {
        "env_id": args.env_id,
        "dt": dt0,
        "frame_skip": frame_skip0,
        "render_fps": int(round(1.0 / dt0)) if dt0 else None,
        "max_episode_steps": spec0.max_episode_steps if spec0 else None,
    }
    if args.track:
        wandb.config.update({"env_cfg": cfg}, allow_val_change=True)



    # === Training dispatch ===
    if args.agent_type == "gru":
        agent, global_step, global_update_idx = trainingLogic.train_gru(
            args, envs, device, run_name
        )
    elif args.agent_type == "mlp":
        agent, global_step, global_update_idx = trainingLogic.train_mlp(
            args, envs, device, run_name
        )
    else:
        raise ValueError(f"Unknown agent_type: {args.agent_type}")
    

    # === Final evaluation (common for both agent types) ===

    final_eval_tag = "final"
    final_eval_video_root = f"videos/{run_name}-eval-final" if args.eval_capture_video else None
    print("\n=== Running final evaluation ===")

    rows, height_logs = evaluateCheetah.evaluate_on_fixed_scenarios(
        agent,
        args.env_id,
        device,
        episodes_per_scenario=args.eval_episodes,
        video_root=final_eval_video_root,
        seed=args.seed + 100 + global_update_idx,
        num_envs=args.eval_num_envs,
        eval_tag=final_eval_tag,
        proxy_steps_per_period=args.eval_proxy_steps_per_period,
        proxy_training_steps=args.eval_proxy_training_steps,
        proxy_amplitude=args.eval_proxy_amplitude,
        reset_after_proxy=args.eval_reset_after_proxy,
    )

    # ---- Summaries over scenarios ----
    ret_means = np.array([r["return_mean"] for r in rows], dtype=np.float64)
    len_means = np.array([r["len_mean"] for r in rows], dtype=np.float64)
    ret_mean_proxy = np.array([r.get("proxy_return_mean", 0.0) for r in rows], dtype=np.float64)
    ret_mean_real = np.array([r.get("real_return_mean", 0.0) for r in rows], dtype=np.float64)

    summary = {
        "eval/return_mean_over_scenarios": float(ret_means.mean()),
        "eval/return_std_over_scenarios":  float(ret_means.std(ddof=1) if len(ret_means) > 1 else 0.0),
        "eval/len_mean_over_scenarios":    float(len_means.mean()),
        "eval/len_std_over_scenarios":     float(len_means.std(ddof=1) if len(len_means) > 1 else 0.0),
        "eval/return_mean_over_scenarios_proxy": float(ret_mean_proxy.mean()),
        "eval/return_std_over_scenarios_proxy":  float(ret_mean_proxy.std(ddof=1) if len(ret_mean_proxy) > 1 else 0.0),
        "eval/return_mean_over_scenarios_real": float(ret_mean_real.mean()),
        "eval/return_std_over_scenarios_real":  float(ret_mean_real.std(ddof=1) if len(ret_mean_real) > 1 else 0.0),
    }

    print("\n=== Eval summary (over scenarios) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}")

    print("\n=== Per-scenario results ===")
    for r in rows:
        print(
            f"{r['scenario']:>16s} | "
            f"ret_mean={r['return_mean']:+8.2f} (±{r['return_std']:.2f}) | "
            f"len_mean={r['len_mean']:7.1f} (±{r['len_std']:.1f})"
        )

    # ---- Save CSV with per-scenario rows ----
    eval_dir = f"videos/{run_name}-eval"
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, "eval_summary.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario", "length", "masspole", "masscart",
                "return_mean", "return_std", "len_mean", "len_std",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] wrote {csv_path}")

    # ---- W&B logging (table + summary + one video if present) ----
    if args.track:
        # Log scalar summary
        wandb.log({**summary, "global_step": global_step})

        # Log table
        table_cols = [
            "scenario", "return_mean", "return_std", "len_mean", "len_std",
            "proxy_return_mean", "proxy_return_std", "real_return_mean", "real_return_std"
        ]
        wb_table = wandb.Table(columns=table_cols)
        for r in rows:
            wb_table.add_data(
                r.get("scenario"),
                float(r.get("return_mean", 0.0)),
                float(r.get("return_std", 0.0)),
                float(r.get("len_mean", 0.0)),
                float(r.get("len_std", 0.0)),
                float(r.get("proxy_return_mean", 0.0)),
                float(r.get("proxy_return_std", 0.0)),
                float(r.get("real_return_mean", 0.0)),
                float(r.get("real_return_std", 0.0)),
            )
        wandb.log({"eval/table": wb_table})

        # Log one short eval clip (if any video produced)
        if os.path.isdir(eval_dir):
            mp4s = sorted([f for f in os.listdir(eval_dir) if f.endswith(".mp4")])
            if mp4s:
                wandb.log({"eval/video": wandb.Video(os.path.join(eval_dir, mp4s[0]), fps=24, format="mp4")})

        wandb.finish()

    envs.close()
    del envs
    import gc; gc.collect()
            