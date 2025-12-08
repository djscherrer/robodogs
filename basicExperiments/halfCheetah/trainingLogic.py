# basicExperiments/halfCheetah/trainingLogic.py

import os
import time
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from . import cheetahAgent, evaluateCheetah  # adjust relative import if needed


# --- checkpoint helpers (closure over local vars) ---
def _checkpoint_payload(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time):
    return {
        "state_dict": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": {
            "schema_version": 1,
            "agent_type": "mlp",
            "env_id": args.env_id,
            "obs_shape": tuple(envs.single_observation_space.shape),
            "act_shape": tuple(envs.single_action_space.shape),
            "train_stats": {
                "global_step": global_step,
                "global_update_idx": global_update_idx,
                "best_return": best_return,
                "episodes_done": episodes_done,
                "sps": int(global_step / max(1, (time.time() - start_time))),
                "saved_at_unix": int(time.time()),
            },
        },
        # keep RNG to allow proper resume
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
        },
    }

def save_last(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir):
    torch.save(_checkpoint_payload(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time), os.path.join(ckpt_dir, "last.pt"))

def save_best(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir):
    torch.save(_checkpoint_payload(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time), os.path.join(ckpt_dir, "best.pt"))
    
def save_checkpoint(ep_done: int, ckpt_dir, agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time):
    path = os.path.join(ckpt_dir, f"ep{ep_done:06d}.pt")
    torch.save(_checkpoint_payload(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time ), path)
    print(f"[save_checkpoint] saved checkpoint to: {path}")


def train_gru(args, envs, device, run_name) -> Tuple[torch.nn.Module, int, int]:
    """
    GRU-based PPO training loop.
    Returns: (agent, global_step, global_update_idx)
    """
    # --- Agent + optimizer ---
    agent = cheetahAgent.GRUAgent(
        envs,
        mlp_hidden_size=args.mlp_hidden_size,
        hidden_size=args.gru_hidden_size,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Checkpoint directory & trackers ---
    ckpt_dir = f"{args.global_ckpt_dir}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_return = -float("inf")
    episodes_done = 0
    global_update_idx = 0
    global_step = 0
    start_time = time.time()

    # --- Resume from checkpoint if requested ---
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        meta = ckpt.get("meta", {})
        train_stats = meta.get("train_stats", {})
        best_return = train_stats.get("best_return", best_return)
        episodes_done = train_stats.get("episodes_done", episodes_done)
        global_step = train_stats.get("global_step", 0)
        global_update_idx = train_stats.get("global_update_idx", 0)
        if "rng" in ckpt:
            random.setstate(ckpt["rng"]["python"])
            np.random.set_state(ckpt["rng"]["numpy"])
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
        print(f"[resume] loaded checkpoint from: {args.resume}")

    # --- Storage buffers ---
    T, B = args.num_steps, args.num_envs
    obs = torch.zeros((T, B) + envs.single_observation_space.shape,
                      dtype=torch.float32, device=device)
    actions = torch.zeros((T, B) + envs.single_action_space.shape,
                          dtype=torch.float32, device=device)
    logprobs = torch.zeros((T, B), device=device)
    rewards = torch.zeros((T, B), device=device)
    dones = torch.zeros((T, B), device=device)
    values = torch.zeros((T, B), device=device)

    H = args.gru_hidden_size
    h0_actor_buf = torch.zeros((T, B, H), device=device)
    h0_critic_buf = torch.zeros((T, B, H), device=device)

    # --- hidden states for GRU ---
    h_critic = torch.zeros(1, B, H, device=device)
    h_actor = torch.zeros(1, B, H, device=device)

    # --- Start rollout ---
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(B, device=device)

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # --- Rollout collection ---
        for step in range(T):
            global_step += B
            obs[step] = next_obs
            dones[step] = next_done

            # Store current hidden state
            h0_actor_buf[step] = h_actor.squeeze(0)
            h0_critic_buf[step] = h_critic.squeeze(0)

            with torch.no_grad():
                action, h_actor, logprob, _, value, h_critic = agent.get_action_and_value(
                    next_obs, h_actor=h_actor, h_critic=h_critic
                )
                values[step] = value.view(-1)

            actions[step] = action
            logprobs[step] = logprob

            # Step env
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            done_mask = torch.as_tensor(next_done_np, device=device, dtype=torch.bool)
            h_actor[:, done_mask] = 0
            h_critic[:, done_mask] = 0

            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            # Episode ends
            if "episode" in infos:
                ep = infos["episode"]
                ep_r = float(ep["r"].max())
                ep_l = int(ep["l"].max())

                proxy_r = infos.get("proxy_return", None)
                if proxy_r is not None:
                    proxy_r = proxy_r[proxy_r != 0]
                    proxy_r = proxy_r.mean() if proxy_r.size > 0 else 0.0
                else:
                    proxy_r = 0.0

                real_r = infos.get("real_return", None)
                if real_r is not None:
                    real_r = real_r[real_r != 0]
                    real_r = real_r.mean() if real_r.size > 0 else 0.0
                else:
                    real_r = 0.0

                print(f"global_step={global_step}, episodic_return={ep_r}, episodic_length={ep_l}, "
                      f"proxy_return={proxy_r}, real_return={real_r}")

                if args.track:
                    wandb.log(
                        {
                            "charts/episodic_return": ep_r,
                            "charts/episodic_length": ep_l,
                            "charts/proxy_return": proxy_r,
                            "charts/real_return": real_r,
                            "global_step": global_step,
                        }
                    )

                # checkpointing
                episodes_done += 1
                save_last(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir)
                if args.save_every_episodes and (episodes_done % args.save_every_episodes == 0):
                    print(f"[save_every_episodes {args.save_every_episodes} episodes {episodes_done}] "
                          f"Saving periodic checkpoint...")
                    save_checkpoint(episodes_done, ckpt_dir, agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time)

                if ep_r > best_return:
                    best_return = ep_r
                    save_best(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir)
                    if args.track:
                        wandb.log({"perf/best_return": best_return, "global_step": global_step})

        # --- GAE & returns ---
        with torch.no_grad():
            next_value, _ = agent.get_value(next_obs, h_critic)
            next_value = next_value.view(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # --- PPO update over sequences ---
        env_inds = np.arange(B)
        np.random.shuffle(env_inds)
        mb_envs_per_batch = B // args.num_minibatches
        clipfracs = []

        for epoch in range(args.update_epochs):
            for start in range(0, B, mb_envs_per_batch):
                mb_envs = env_inds[start:start + mb_envs_per_batch]    # [M]

                seq_obs     = obs[:, mb_envs]        # [T,M,...]
                seq_actions = actions[:, mb_envs]
                seq_oldlp   = logprobs[:, mb_envs]
                seq_returns = returns[:, mb_envs]
                seq_adv     = advantages[:, mb_envs]
                seq_dones   = dones[:, mb_envs]
                seq_oldV    = values.view(T, B)[:, mb_envs]

                h_a = h0_actor_buf[0, mb_envs].detach().unsqueeze(0).contiguous()
                h_c = h0_critic_buf[0, mb_envs].detach().unsqueeze(0).contiguous()

                newlp_list, newv_list, ent_list = [], [], []
                for t in range(T):
                    if t > 0:
                        mask = (seq_dones[t] > 0.5)
                        if mask.any():
                            h_a[:, mask] = 0
                            h_c[:, mask] = 0

                    _, h_a, lp_t, ent_t, v_t, h_c = agent.get_action_and_value(
                        seq_obs[t], h_actor=h_a, h_critic=h_c, action=seq_actions[t]
                    )
                    newlp_list.append(lp_t.view(-1))
                    newv_list.append(v_t.view(-1))
                    ent_list.append(ent_t.view(-1))

                newlogprob = torch.stack(newlp_list, dim=0)  # [T,M]
                newvalue   = torch.stack(newv_list, dim=0)
                entropy    = torch.stack(ent_list, dim=0)

                with torch.no_grad():
                    lp_diff = (newlogprob - seq_oldlp).abs().mean().item()
                    if args.track:
                        wandb.log({"debug/logprob_replay_absdiff": lp_diff, "global_step": global_step})

                logratio = newlogprob - seq_oldlp
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                adv = seq_adv
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio,
                                              1.0 - args.clip_coef,
                                              1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.clip_vloss:
                    v_unclipped = (newvalue - seq_returns).pow(2)
                    v_clipped = seq_oldV + torch.clamp(newvalue - seq_oldV,
                                                       -args.clip_coef,
                                                       args.clip_coef)
                    v_loss = 0.5 * torch.max(
                        v_unclipped, (v_clipped - seq_returns).pow(2)
                    ).mean()
                else:
                    v_loss = 0.5 * (newvalue - seq_returns).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # --- Logging ---
        b_values = values.view(-1)
        b_returns = returns.view(-1)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        if args.track:
            wandb.log(
                {
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
                }
            )
        print("SPS:", int(global_step / (time.time() - start_time)))

        # --- Lightweight evaluation every N updates ---
        if args.eval_every > 0 and (global_update_idx % args.eval_every) == 0:
            agent.eval()
            try:
                print("\n=== Running evaluation ===")
                eval_tag = f"u{global_update_idx:05d}"
                eval_video_root: Optional[str] = (
                    f"videos/{run_name}-eval" if args.eval_capture_video else None
                )

                rows, height_logs = evaluateCheetah.evaluate_on_fixed_scenarios(
                    agent,
                    args.env_id,
                    device,
                    episodes_per_scenario=args.eval_episodes,
                    video_root=eval_video_root,
                    seed=args.seed + 100 + global_update_idx,
                    num_envs=args.eval_num_envs,
                    eval_tag=eval_tag,
                    proxy_steps_per_period=args.eval_proxy_steps_per_period,
                    proxy_training_steps=args.eval_proxy_training_steps,
                    proxy_amplitude=args.eval_proxy_amplitude,
                    reset_after_proxy=args.reset_after_proxy,
                )

                # aggregate eval stats
                ret_means = np.array([r["return_mean"] for r in rows], dtype=np.float64)
                len_means = np.array([r["len_mean"] for r in rows], dtype=np.float64)
                ret_mean_proxy = np.array([r.get("proxy_return_mean", 0.0) for r in rows], dtype=np.float64)
                ret_mean_real = np.array([r.get("real_return_mean", 0.0) for r in rows], dtype=np.float64)
                eval_summary = {
                    "eval/return_mean_over_scenarios": float(ret_means.mean()),
                    "eval/return_std_over_scenarios":  float(ret_means.std(ddof=1) if len(ret_means) > 1 else 0.0),
                    "eval/len_mean_over_scenarios":    float(len_means.mean()),
                    "eval/len_std_over_scenarios":     float(len_means.std(ddof=1) if len(len_means) > 1 else 0.0),
                    "eval/return_mean_over_scenarios_proxy": float(ret_mean_proxy.mean()),
                    "eval/return_std_over_scenarios_proxy":  float(ret_mean_proxy.std(ddof=1) if len(ret_mean_proxy) > 1 else 0.0),
                    "eval/return_mean_over_scenarios_real": float(ret_mean_real.mean()),
                    "eval/return_std_over_scenarios_real":  float(ret_mean_real.std(ddof=1) if len(ret_mean_real) > 1 else 0.0),
                }

                print(
                    f"\n[eval @ update {global_update_idx}] "
                    f"mean_ret={eval_summary['eval/return_mean_over_scenarios']:.2f} "
                    f"std_ret={eval_summary['eval/return_std_over_scenarios']:.2f} "
                    f"mean_ret_proxy={eval_summary['eval/return_mean_over_scenarios_proxy']:.2f} "
                    f"mean_ret_real={eval_summary['eval/return_mean_over_scenarios_real']:.2f}"
                )

                # Log height trajectories (unchanged from your code)
                for tr in height_logs:
                    scen = tr["scenario"]
                    ep_idx = tr["episode_idx"]
                    h = tr["height"]
                    tgt = tr["target"]
                    T_h = len(h)
                    xs = list(range(T_h))
                    ys = [h.tolist(), tgt.tolist()]
                    keys = ["height", "target"]

                    line_plot = wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=keys,
                        title=f"{scen} ep {ep_idx}",
                        xname="t_step",
                    )

                    wandb.log({
                        f"height_logging/{scen}/ep_{ep_idx}": line_plot,
                        "global_step": global_step,
                    })

                if eval_summary["eval/return_mean_over_scenarios"] > best_return:
                    best_return = eval_summary["eval/return_mean_over_scenarios"]
                    save_best(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir)

                if args.track:
                    wandb.log({**eval_summary,
                               "global_step": global_step,
                               "eval/update_idx": global_update_idx})

                    for r in rows:
                        scen = r["scenario"]
                        wandb.log({
                            f"eval_meta/return_mean/{scen}": float(r["return_mean"]),
                            f"eval_meta/len_mean/{scen}":    float(r["len_mean"]),
                            f"eval_meta/return_std/{scen}":  float(r["return_std"]),
                            f"eval_meta/len_std/{scen}":     float(r["len_std"]),
                            f"eval_meta/return_mean_proxy/{scen}": float(r.get("proxy_return_mean", 0.0)),
                            f"eval_meta/return_mean_real/{scen}": float(r.get("real_return_mean", 0.0)),
                            f"eval_meta/return_std_proxy/{scen}": float(r.get("proxy_return_std", 0.0)),
                            f"eval_meta/return_std_real/{scen}": float(r.get("real_return_std", 0.0)),
                            "global_step": global_step,
                        })

                    if eval_video_root and os.path.isdir(eval_video_root):
                        mp4s = sorted(f for f in os.listdir(eval_video_root) if f.endswith(".mp4"))
                        if mp4s:
                            wandb.log({
                                "eval/video": wandb.Video(mp4s[0], fps=24, format="mp4")
                            })
            except Exception as e:
                print(f"[eval] skipped due to error: {e}")
            finally:
                agent.train()
                print("=== Finished evaluation ===\n")

        global_update_idx += 1

    # return agent + stats to main for final eval
    return agent, global_step, global_update_idx

def train_mlp(args, envs, device, run_name):
    """
    Vanilla MLP PPO training loop (no recurrence).
    Mirrors the original train_cheetah MLP script, but as a reusable function.

    Returns:
        agent, global_step, global_update_idx
    """
    # --- Agent + optimizer ---
    agent = cheetahAgent.Agent(envs).to(device)  # MLP actor-critic
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Checkpoint directory & trackers ---
    ckpt_dir = f"{args.global_ckpt_dir}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_return = -float("inf")
    episodes_done = 0
    global_update_idx = 0
    global_step = 0
    start_time = time.time()



    # --- Resume from checkpoint if requested ---
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        # new-style stats
        meta = ckpt.get("meta", {})
        ts = meta.get("train_stats", {})

        best_return = ts.get("best_return", ckpt.get("best_return", best_return))
        episodes_done = ts.get("episodes_done", ckpt.get("episodes_done", episodes_done))
        global_step = ts.get("global_step", ckpt.get("global_step", 0))
        global_update_idx = ts.get("global_update_idx", ckpt.get("global_update_idx", 0))

        # RNG restore (if present)
        if "rng" in ckpt:
            random.setstate(ckpt["rng"]["python"])
            np.random.set_state(ckpt["rng"]["numpy"])
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])

        print(f"[resume] loaded checkpoint from: {args.resume}")

    # --- Buffers (no GRU) ---
    T, B = args.num_steps, args.num_envs
    obs = torch.zeros(
        (T, B) + envs.single_observation_space.shape,
        dtype=torch.float32,
        device=device,
    )
    actions = torch.zeros(
        (T, B) + envs.single_action_space.shape,
        dtype=torch.float32,
        device=device,
    )
    logprobs = torch.zeros((T, B), device=device)
    rewards = torch.zeros((T, B), device=device)
    dones = torch.zeros((T, B), device=device)
    values = torch.zeros((T, B), device=device)

    b_inds = np.arange(args.batch_size)

    # --- Start rollout ---
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(B, device=device)

    for iteration in range(1, args.num_iterations + 1):
        # LR anneal
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # === Rollout collection ===
        for step in range(T):
            global_step += B
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.view(-1)

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done_np = np.logical_or(terminations, truncations)

            rewards[step] = torch.as_tensor(
                reward, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)

            # episode logging
            if "episode" in infos:
                ep = infos["episode"]
                ep_r = float(ep["r"].max())
                ep_l = int(ep["l"].max())
                print(f"global_step={global_step}, episodic_return={ep_r}")
                if args.track:
                    wandb.log(
                        {
                            "charts/episodic_return": ep_r,
                            "charts/episodic_length": ep_l,
                            "global_step": global_step,
                        }
                    )

                episodes_done += 1
                save_last(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir)
                if args.save_every_episodes and (episodes_done % args.save_every_episodes == 0):
                    save_checkpoint(episodes_done, ckpt_dir, agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir)

                if ep_r > best_return:
                    best_return = ep_r
                    save_best(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir)
                    if args.track:
                        wandb.log({"perf/best_return": best_return, "global_step": global_step})

        # === GAE & returns ===
        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # === Flatten for PPO update ===
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # === PPO update ===
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

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    mb_obs, mb_actions
                )

                logratio = new_logprob - mb_old_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                adv = mb_adv
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                if args.clip_vloss:
                    v_unclipped = (new_value - mb_ret).pow(2)
                    v_clipped = mb_val + torch.clamp(
                        new_value - mb_val,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss = 0.5 * torch.max(
                        v_unclipped, (v_clipped - mb_ret).pow(2)
                    ).mean()
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

        # === Logging ===
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        if args.track:
            wandb.log(
                {
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
                }
            )
        print("SPS:", int(global_step / (time.time() - start_time)))

        # === Lightweight evaluation every N updates ===
        if args.eval_every > 0 and (global_update_idx % args.eval_every) == 0:
            agent.eval()
            try:
                print("\n=== Running evaluation (MLP) ===")
                eval_tag = f"u{global_update_idx:05d}"
                eval_video_root: Optional[str] = (
                    f"videos/{run_name}-eval/{eval_tag}"
                    if args.eval_capture_video
                    else None
                )

                # NOTE: if evaluate_on_fixed_scenarios got extra kwargs later,
                # you can add them here similar to the GRU script.
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
                    "eval/return_std_over_scenarios": float(
                        ret_means.std(ddof=1) if len(ret_means) > 1 else 0.0
                    ),
                    "eval/len_mean_over_scenarios": float(len_means.mean()),
                    "eval/len_std_over_scenarios": float(
                        len_means.std(ddof=1) if len(len_means) > 1 else 0.0
                    ),
                }

                print(
                    f"[eval @ update {global_update_idx}] "
                    f"mean_ret={eval_summary['eval/return_mean_over_scenarios']:.2f} "
                    f"std_ret={eval_summary['eval/return_std_over_scenarios']:.2f}"
                )

                if eval_summary["eval/return_mean_over_scenarios"] > best_return:
                    best_return = eval_summary["eval/return_mean_over_scenarios"]
                    save_best(agent, optimizer, args, envs, global_step, global_update_idx, best_return, episodes_done, start_time, ckpt_dir)

                if args.track:
                    wandb.log(
                        {
                            **eval_summary,
                            "global_step": global_step,
                            "eval/update_idx": global_update_idx,
                        }
                    )
                    for r in rows:
                        scen = r["scenario"]
                        wandb.log(
                            {
                                f"eval_meta/return_mean/{scen}": float(r["return_mean"]),
                                f"eval_meta/len_mean/{scen}": float(r["len_mean"]),
                                f"eval_meta/return_std/{scen}": float(r["return_std"]),
                                f"eval_meta/len_std/{scen}": float(r["len_std"]),
                                "global_step": global_step,
                            }
                        )
                    if eval_video_root and os.path.isdir(eval_video_root):
                        mp4s = sorted(
                            f for f in os.listdir(eval_video_root) if f.endswith(".mp4")
                        )
                        if mp4s:
                            wandb.log(
                                {
                                    "eval/video": wandb.Video(
                                        os.path.join(eval_video_root, mp4s[0]),
                                        fps=24,
                                        format="mp4",
                                    )
                                }
                            )
            except Exception as e:
                print(f"[eval] skipped due to error: {e}")
            finally:
                agent.train()
                print("=== Finished evaluation ===\n")

        global_update_idx += 1

    # return for final eval in main
    return agent, global_step, global_update_idx