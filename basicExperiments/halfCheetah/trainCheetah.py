# docs and experiment results can be found at
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

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
from . import cheetahAgent, cheetahEnv, evaluateCheetah


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
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

    # Environment Randomization (passed to CheetahCustom/make_env)
    randomize_morphology_every: int = 5
    """If >0, randomize morphology every N episodes"""
    morphology_jitter: float = 0.2
    """The amount of jitter to apply when randomizing morphology"""

    # Evaluation settings (mirrors recurrent script)
    eval_every: int = 8
    """If >0, evaluate the agent every N updates"""
    eval_episodes: int = 8
    """Episodes per scenario during evaluation"""
    eval_num_envs: int = 8
    """Parallel envs for eval"""
    eval_capture_video: bool = False
    """Capture videos during eval"""

    # Algorithm specific arguments
    env_id: str = "Cheetah_MLP_Rand5"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel environments"""
    num_steps: int = 256
    """steps per environment per rollout"""
    anneal_lr: bool = True
    """LR annealing"""
    gamma: float = 0.99
    """discount"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    num_minibatches: int = 4
    """number of SGD minibatches"""
    update_epochs: int = 10
    """PPO epochs"""
    norm_adv: bool = True
    """normalize advantages"""
    clip_coef: float = 0.2
    """policy clip coefficient"""
    clip_vloss: bool = True
    """clip value loss"""
    ent_coef: float = 0.001
    """entropy coefficient"""
    vf_coef: float = 0.5
    """value loss coefficient"""
    max_grad_norm: float = 0.5
    """grad clip norm"""
    target_kl: float = 0.01
    """early stop if KL > target"""

    # computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # Checkpointing
    resume: Optional[str] = None
    """Path to a checkpoint (.pt) to resume from."""
    save_every_episodes: int = 0
    """If >0, also save 'last.pt' every N completed episodes."""
    global_ckpt_dir: str = "checkpoints/halfCheetah"


def return_config(env: gym.Env):
    sp = env.unwrapped.observation_space
    ac = env.unwrapped.action_space
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
        # MPS can be slower for small models; defaulting to CPU unless you want it
        return torch.device("cpu")
    return torch.device("cpu")


if __name__ == "__main__":
    # ---- Args & derived sizes ----
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # ---- Register env ----
    register(
        id=args.env_id,
        entry_point=cheetahEnv.CheetahCustom,
        max_episode_steps=500,
    )

    # ---- W&B init ----
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(start_method="thread"),
            mode="online",
        )
        # Make global_step the step metric for all logs
        try:
            wandb.define_metric("global_step")
            wandb.define_metric("*/**", step_metric="global_step")
        except Exception:
            pass

    # ---- Seeding ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = pick_device()
    print(f"Using device: {device}")

    # ---- Env setup (match recurrent async/sync choice) ----
    env_fns = [
        cheetahEnv.make_env(
            args.env_id,
            i,
            args.capture_video,
            run_name,
            args.randomize_morphology_every,
            args.morphology_jitter,
        )
        for i in range(args.num_envs)
    ]
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    prefer = "sync" if has_mps else ("async" if has_cuda else "sync")
    if prefer == "async":
        envs = AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)

    # ---- Config snapshot ----
    base_cfg = return_config(envs.envs[0])
    cfg = {
        "env_id": args.env_id,
        "dt": getattr(envs.envs[0].unwrapped, "dt", None),
        "frame_skip": getattr(envs.envs[0].unwrapped, "frame_skip", None),
        "render_fps": (
            int(round(1.0 / getattr(envs.envs[0].unwrapped, "dt")))
            if hasattr(envs.envs[0].unwrapped, "dt") else None
        ),
        "max_episode_steps": base_cfg["max_episode_steps"],
        "obs_shape": base_cfg["obs_shape"],
        "act_shape": base_cfg["act_shape"],
        "act_low": base_cfg["act_low"],
        "act_high": base_cfg["act_high"],
    }
    if args.track:
        wandb.config.update({"env_cfg": cfg}, allow_val_change=True)

    # ---- Agent (MLP) + optimizer ----
    agent = cheetahAgent.Agent(envs).to(device)  # MLP actor-critic with tanh-squashed Gaussian
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ---- Checkpointing ----
    ckpt_dir = f"{args.global_ckpt_dir}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_return = -float("inf")
    episodes_done = 0
    global_update_idx = 0

    def _checkpoint_payload():
        return {
            "state_dict": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_return": best_return,
            "episodes_done": episodes_done,
            "global_step": global_step,
            "global_update_idx": global_update_idx,
            "args": vars(args),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
            },
        }

    def save_last():
        torch.save(_checkpoint_payload(), os.path.join(ckpt_dir, "last.pt"))

    def save_best():
        torch.save(_checkpoint_payload(), os.path.join(ckpt_dir, "best.pt"))

    # ---- Resume ----
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        best_return = ckpt.get("best_return", best_return)
        episodes_done = ckpt.get("episodes_done", episodes_done)
        global_step = ckpt.get("global_step", 0)
        global_update_idx = ckpt.get("global_update_idx", 0)
        if "rng" in ckpt:
            random.setstate(ckpt["rng"]["python"])
            np.random.set_state(ckpt["rng"]["numpy"])
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
        print(f"[resume] loaded checkpoint from: {args.resume}")
    else:
        global_step = 0

    # ---- Buffers (no GRU) ----
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape,
                      dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape,
                          dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # ---- Rollout ----
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)

    b_inds = np.arange(args.batch_size)

    for iteration in range(1, args.num_iterations + 1):
        # LR anneal
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Collect
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.view(-1)

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done, device=device, dtype=torch.float32)

            if "episode" in infos:
                ep = infos["episode"]
                ep_r = float(ep["r"].max())
                ep_l = int(ep["l"].max())
                print(f"global_step={global_step}, episodic_return={ep_r}")
                if args.track:
                    wandb.log({
                        "charts/episodic_return": ep_r,
                        "charts/episodic_length": ep_l,
                        "global_step": global_step,
                    })
                episodes_done += 1
                save_last()
                if args.save_every_episodes and (episodes_done % args.save_every_episodes == 0):
                    save_last()
                if ep_r > best_return:
                    best_return = ep_r
                    save_best()
                    if args.track:
                        wandb.log({"perf/best_return": best_return, "global_step": global_step})

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_old_logprobs = b_logprobs[mb_inds]
                mb_adv = b_advantages[mb_inds]
                mb_ret = b_returns[mb_inds]
                mb_val = b_values[mb_inds]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(mb_obs, mb_actions)

                logratio = new_logprob - mb_old_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                adv = mb_adv
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                if args.clip_vloss:
                    v_unclipped = (new_value - mb_ret).pow(2)
                    v_clipped = mb_val + torch.clamp(new_value - mb_val, -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - mb_ret).pow(2)).mean()
                else:
                    v_loss = 0.5 * (new_value - mb_ret).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logs
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        if args.track:
            wandb.log({
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": float(v_loss.item()),
                "losses/policy_loss": float(pg_loss.item()),
                "losses/entropy": float(entropy_loss.item()),
                "losses/old_approx_kl": float(old_approx_kl.item()),
                "losses/approx_kl": float(approx_kl.item()),
                "losses/clipfrac": float(np.mean(clipfracs)),
                "losses/explained_variance": float(explained_var),
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "global_step": global_step,
            })
        print("SPS:", int(global_step / (time.time() - start_time)))

        # --- lightweight eval every N updates (same as recurrent) ---
        if args.eval_every > 0 and (global_update_idx % args.eval_every) == 0:
            agent.eval()
            try:
                print("\n=== Running evaluation ===")
                eval_tag = f"u{global_update_idx:05d}"
                eval_video_root: Optional[str] = (
                    f"videos/{run_name}-eval/{eval_tag}" if args.eval_capture_video else None
                )

                rows = evaluateCheetah.evaluate_on_fixed_scenarios(
                    agent,
                    args.env_id,
                    device,
                    episodes_per_scenario=args.eval_episodes,
                    video_root=eval_video_root,
                    seed=args.seed + 100 + global_update_idx,
                    num_envs=args.eval_num_envs,
                )

                ret_means = np.array([r["return_mean"] for r in rows], dtype=np.float64)
                len_means = np.array([r["len_mean"] for r in rows], dtype=np.float64)
                eval_summary = {
                    "eval/return_mean_over_scenarios": float(ret_means.mean()),
                    "eval/return_std_over_scenarios":  float(ret_means.std(ddof=1) if len(ret_means) > 1 else 0.0),
                    "eval/len_mean_over_scenarios":    float(len_means.mean()),
                    "eval/len_std_over_scenarios":     float(len_means.std(ddof=1) if len(len_means) > 1 else 0.0),
                }

                print(f"[eval @ update {global_update_idx}] "
                      f"mean_ret={eval_summary['eval/return_mean_over_scenarios']:.2f} "
                      f"std_ret={eval_summary['eval/return_std_over_scenarios']:.2f}")

                if eval_summary["eval/return_mean_over_scenarios"] > best_return:
                    best_return = eval_summary["eval/return_mean_over_scenarios"]
                    save_best()

                if args.track:
                    wandb.log({**eval_summary, "global_step": global_step,
                               "eval/update_idx": global_update_idx})

                    for r in rows:
                        scen = r["scenario"]
                        wandb.log({
                            f"eval_meta/return_mean/{scen}": float(r["return_mean"]),
                            f"eval_meta/len_mean/{scen}":    float(r["len_mean"]),
                            f"eval_meta/return_std/{scen}":  float(r["return_std"]),
                            f"eval_meta/len_std/{scen}":     float(r["len_std"]),
                            "global_step": global_step,
                        })

                    if eval_video_root and os.path.isdir(eval_video_root):
                        mp4s = sorted(f for f in os.listdir(eval_video_root) if f.endswith(".mp4"))
                        if mp4s:
                            wandb.log({"eval/video": wandb.Video(os.path.join(eval_video_root, mp4s[0]), fps=24, format="mp4")})
            except Exception as e:
                print(f"[eval] skipped due to error: {e}")
            finally:
                agent.train()
                print("=== Finished evaluation ===\n")

        global_update_idx += 1

    # ---- Final eval (same reporting as recurrent) ----
    rows = evaluateCheetah.evaluate_on_fixed_scenarios(
        agent, args.env_id, device, 6, video_root=f"videos/{run_name}-eval", seed=args.seed + 100
    )
    ret_means = np.array([r["return_mean"] for r in rows], dtype=np.float64)
    len_means = np.array([r["len_mean"] for r in rows], dtype=np.float64)
    summary = {
        "eval/return_mean_over_scenarios": float(ret_means.mean()),
        "eval/return_std_over_scenarios":  float(ret_means.std(ddof=1) if len(ret_means) > 1 else 0.0),
        "eval/len_mean_over_scenarios":    float(len_means.mean()),
        "eval/len_std_over_scenarios":     float(len_means.std(ddof=1) if len(len_means) > 1 else 0.0),
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

    # ---- Save CSV + W&B table & artifact ----
    eval_dir = f"videos/{run_name}-eval"
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, "eval_summary.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario","length","masspole","masscart","return_mean","return_std","len_mean","len_std"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] wrote {csv_path}")

    if args.track:
        wandb.log({**summary, "global_step": global_step})
        table_cols = ["scenario","length","masspole","masscart","return_mean","return_std","len_mean","len_std"]
        wb_table = wandb.Table(columns=table_cols)
        for r in rows:
            wb_table.add_data(*[r[c] for c in table_cols])
        wandb.log({"eval/table": wb_table})

        if os.path.isdir(eval_dir):
            mp4s = sorted([f for f in os.listdir(eval_dir) if f.endswith(".mp4")])
            if mp4s:
                wandb.log({"eval/video": wandb.Video(os.path.join(eval_dir, mp4s[0]), fps=24, format="mp4")})

        art = wandb.Artifact(name="cheetah-agent-mlp", type="model")
        best_path = os.path.join(ckpt_dir, "best.pt")
        last_path = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(best_path): art.add_file(best_path)
        if os.path.exists(last_path): art.add_file(last_path)
        wandb.log_artifact(art)
        wandb.finish()

    envs.close()
    del envs
    import gc; gc.collect()