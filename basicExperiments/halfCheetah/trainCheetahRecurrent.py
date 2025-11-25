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
from basicExperiments.halfCheetah import cheetahAgent, cheetahEnv, evaluateCheetah


@dataclass
class Args:
    exp_name: str = "Cheetah_Recurrent_Rand5_ProxyOnly_256s"
    """the name of this experiment"""
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

    # Proxy task settings training
    proxy_training_steps: int = 256
    """Number of steps over which to train on proxy task before switching to main task"""
    proxy_period_steps: int = 64
    """Number of steps for one full sine wave period"""
    proxy_amplitude: float = 0.10 
    """Amplitude of the proxy sine wave relative to inital torso height"""

    # Proxy task settings evaluation
    eval_proxy_training_steps: int = 256
    """Number of steps over which to train on proxy task before switching to main task during evaluation"""
    eval_proxy_period_steps: int = 64
    """Number of steps for one full sine wave period during evaluation"""
    eval_proxy_amplitude: float = 0.10
    """Amplitude of the proxy sine wave during evaluation"""

    # Algorithm specific arguments
    env_id: str = "Cheetah_Recurrent"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512
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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = pick_device()
    #device = "cpu" # ML stuff to tiny to have an impact 
    print(f"Using device: {device}")

    # env setup
    env_fns = [
        cheetahEnv.make_env(args.env_id, i, args.capture_video, run_name, args.proxy_period_steps, args.proxy_training_steps, args.proxy_amplitude, args.randomize_morphology_every, args.morphology_jitter)
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

    # Safely pull per-env attributes from worker 0
    specs       = envs.get_attr("spec")          # list length = num_envs
    frame_skips = envs.get_attr("frame_skip")    # list
    dts         = envs.get_attr("dt")            # list (if your env defines .dt)

    spec0        = specs[0]
    frame_skip0  = frame_skips[0] if frame_skips else None
    dt0          = dts[0] if dts else None

    cfg = {
        "env_id": args.env_id,
        "dt": dt0,
        "frame_skip": frame_skip0,
        "render_fps": (int(round(1.0 / dt0)) if dt0 else None),
        "max_episode_steps": (spec0.max_episode_steps if spec0 else None),
    }
    text = "\n".join(f"{k}: {v}" for k, v in cfg.items())

    if args.track:
        wandb.config.update({"env_cfg": cfg}, allow_val_change=True)

    # GRU Agent init
    agent = cheetahAgent.GRUAgent(envs, mlp_hidden_size=args.mlp_hidden_size, hidden_size=args.gru_hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # === Checkpoint directory & trackers ===
    ckpt_dir = f"{args.global_ckpt_dir}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_return = -float("inf")
    episodes_done = 0  # count completed episodes this run (for periodic saving)
    global_update_idx = 0  # count PPO updates (outer loop iterations)

    def _checkpoint_payload():
        return {
            "state_dict": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": {
                "schema_version": 1,
                "agent_type": "gru" if hasattr(agent, "gru_Actor") else "mlp",
                "env_id": args.env_id,
                "obs_shape": tuple(envs.single_observation_space.shape),
                "act_shape": tuple(envs.single_action_space.shape),
                "gru_hidden_size": getattr(agent, "hidden_size", None),
                "train_stats": {
                    "global_step": global_step,
                    "global_update_idx": global_update_idx,
                    "best_return": best_return,
                    "episodes_done": episodes_done,
                    "sps": int(global_step / max(1, (time.time() - start_time))),
                    "saved_at_unix": int(time.time()),
                },
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def save_last():
        torch.save(_checkpoint_payload(), os.path.join(ckpt_dir, "last.pt"))

    def save_best():
        torch.save(_checkpoint_payload(), os.path.join(ckpt_dir, "best.pt"))

    # === Resume from checkpoint if requested (AFTER agent/optimizer/helpers exist) ===
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        # restore trackers
        best_return = ckpt.get("best_return", best_return)
        episodes_done = ckpt.get("episodes_done", episodes_done)
        global_step = ckpt.get("global_step", 0)
        global_update_idx = ckpt.get("global_update_idx", 0)
        # restore RNG (overrides earlier seeding; that’s okay when resuming)
        if "rng" in ckpt:
            random.setstate(ckpt["rng"]["python"])
            np.random.set_state(ckpt["rng"]["numpy"])
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
        print(f"[resume] loaded checkpoint from: {args.resume}")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape,dtype=torch.float32).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape,dtype=torch.float32).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    H = args.gru_hidden_size  # GRU hidden size
    h0_actor_buf  = torch.zeros((args.num_steps, args.num_envs, H), device=device)
    h0_critic_buf = torch.zeros((args.num_steps, args.num_envs, H), device=device)


    # hidden states for GRU
    h_critic = torch.zeros(1, args.num_envs, args.gru_hidden_size).to(device)
    h_actor = torch.zeros(1, args.num_envs, args.gru_hidden_size).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        # h_critic = torch.zeros(1, args.num_envs, 128).to(device)
        # h_actor = torch.zeros(1, args.num_envs, 128).to(device)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            h0_actor_buf[step]  = h_actor.squeeze(0)   # [B,H]
            h0_critic_buf[step] = h_critic.squeeze(0)  # [B,H]
            with torch.no_grad():
                action, h_actor, logprob, _, value, h_critic = agent.get_action_and_value(next_obs, h_actor=h_actor, h_critic=h_critic)
                values[step] = value.flatten()
                # action = action.squeeze()
                # logprob = logprob.squeeze()
            actions[step] = action
            logprobs[step] = logprob
            

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            done_mask = torch.as_tensor(next_done, device=device, dtype=torch.bool)
            h_actor[:, done_mask] = 0
            h_critic[:, done_mask] = 0
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            if "episode" in infos:
                ep = infos["episode"]
                ep_r = float(ep["r"].max())
                ep_l = int(ep["l"].max())
                proxy_r = infos.get("proxy_return", None)
                proxy_r = proxy_r[proxy_r != 0]
                proxy_r = proxy_r.mean() if proxy_r.size > 0 else 0.0

                real_r = infos.get("real_return", None)
                real_r = real_r[real_r != 0]
                real_r = real_r.mean() if real_r.size > 0 else 0.0
                print(f"global_step={global_step}, episodic_return={ep_r}, episodic_length={ep_l}, proxy_return={proxy_r}, real_return={real_r}")

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
                # === checkpointing ===
                episodes_done += 1
                # Always keep the rolling "last.pt"
                save_last()

                if args.save_every_episodes and (episodes_done % args.save_every_episodes == 0):
                    save_last()

                if ep_r > best_return:
                    best_return = ep_r
                    save_best()
                    if args.track:
                        wandb.log({"perf/best_return": best_return, "global_step": global_step})
        

        # bootstrap value if not done
        with torch.no_grad():
            next_value, _ = agent.get_value(next_obs, h_critic)   # h_critic from the last env step (after masking dones)
            next_value = next_value.view(-1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        T, B = args.num_steps, args.num_envs
        env_inds = np.arange(B)
        np.random.shuffle(env_inds)
        mb_envs_per_batch = B // args.num_minibatches
        clipfracs = []

        for epoch in range(args.update_epochs):
            for start in range(0, B, mb_envs_per_batch):
                mb_envs = env_inds[start:start + mb_envs_per_batch]        # [M]
                # slice sequences [T,M,...]
                seq_obs     = obs[:, mb_envs]                               # [T,M,obs_dim]
                seq_actions = actions[:, mb_envs]                           # [T,M]
                seq_oldlp   = logprobs[:, mb_envs]                           # [T,M]
                seq_returns = returns[:, mb_envs]                            # [T,M]
                seq_adv     = advantages[:, mb_envs]                         # [T,M]
                seq_dones   = dones[:, mb_envs]                              # [T,M]  # 1 if episode just ended before this step
                seq_oldV    = values.view(T, B)[:, mb_envs]                  # [T,M]  # for value clipping

                # initial hidden = rollout snapshot at t=0 for these envs
                h_a = h0_actor_buf[0, mb_envs].detach().unsqueeze(0).contiguous()   # [1,M,H]
                h_c = h0_critic_buf[0, mb_envs].detach().unsqueeze(0).contiguous()  # [1,M,H]

                newlp_list, newv_list, ent_list = [], [], []
                for t in range(T):
                    if t > 0:
                        mask = (seq_dones[t] > 0.5)                                 # reset at step t where done==1 entering t
                        if mask.any():
                            h_a[:, mask] = 0
                            h_c[:, mask] = 0

                    _, h_a, lp_t, ent_t, v_t, h_c = agent.get_action_and_value(
                        seq_obs[t], h_actor=h_a, h_critic=h_c, action=seq_actions[t]
                    )
                    newlp_list.append(lp_t.view(-1))           # [M]
                    newv_list.append(v_t.view(-1))             # [M]
                    ent_list.append(ent_t.view(-1))            # [M]

                newlogprob = torch.stack(newlp_list, dim=0)    # [T,M]
                newvalue   = torch.stack(newv_list, dim=0)     # [T,M]
                entropy    = torch.stack(ent_list, dim=0)      # [T,M]
                with torch.no_grad():
                    lp_diff = (newlogprob - seq_oldlp).abs().mean().item()
                    if args.track:
                        wandb.log({"debug/logprob_replay_absdiff": lp_diff, "global_step": global_step})

                logratio = newlogprob - seq_oldlp              # [T,M]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                adv = seq_adv
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss with proper old-value indexing
                if args.clip_vloss:
                    v_unclipped = (newvalue - seq_returns).pow(2)
                    v_clipped = seq_oldV + torch.clamp(newvalue - seq_oldV, -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - seq_returns).pow(2)).mean()
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
                print("\n=== Running evaluation ===")
                eval_tag = f"u{global_update_idx:05d}"
                eval_video_root: Optional[str] = (f"videos/{run_name}-eval" if args.eval_capture_video else None)

                rows = evaluateCheetah.evaluate_on_fixed_scenarios(
                    agent,
                    args.env_id,
                    device,
                    episodes_per_scenario=args.eval_episodes,
                    video_root=eval_video_root,
                    seed=args.seed + 100 + global_update_idx,   # different seed per eval
                    num_envs=args.eval_num_envs,
                    eval_tag=eval_tag,
                    proxy_period_steps=args.eval_proxy_period_steps,
                    proxy_training_steps=args.eval_proxy_training_steps,
                    proxy_amplitude=args.eval_proxy_amplitude,
                )

                # Summaries
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

                print(f"\n[eval @ update {global_update_idx}] "
                    f"mean_ret={eval_summary['eval/return_mean_over_scenarios']:.2f} "
                    f"std_ret={eval_summary['eval/return_std_over_scenarios']:.2f}"
                    f"mean_ret_proxy={eval_summary['eval/return_mean_over_scenarios_proxy']:.2f} "
                    f"mean_ret_real={eval_summary['eval/return_mean_over_scenarios_real']:.2f}")

                # Optional: treat eval mean as "best" gate too
                if eval_summary["eval/return_mean_over_scenarios"] > best_return:
                    best_return = eval_summary["eval/return_mean_over_scenarios"]
                    save_best()

                if args.track:
                    # Log rolled-up scalars
                    wandb.log({**eval_summary, "global_step": global_step,
                            "eval/update_idx": global_update_idx})

                    # Log per-scenario statistics
                    for r in rows:
                        scen = r["scenario"]              # e.g. "baseline", "torso_lor", ...
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

                    # Optionally log one short video if present
                    if eval_video_root and os.path.isdir(eval_video_root):
                        mp4s = sorted(f for f in os.listdir(eval_video_root) if f.endswith(".mp4"))
                        if mp4s:
                            wandb.log({
                                "eval/video": wandb.Video(os.path.join(eval_video_root, mp4s[0]), fps=24, format="mp4")
                            })
            except Exception as e:
                print(f"[eval] skipped due to error: {e}")
            finally:
                agent.train()
                print("=== Finished evaluation ===\n")

        global_update_idx += 1

    rows = evaluateCheetah.evaluate_on_fixed_scenarios(agent, args.env_id, device, 6, video_root=f"videos/{run_name}-eval", seed=args.seed+100)

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
            "scenario", "length", "masspole", "masscart",
            "return_mean", "return_std", "len_mean", "len_std",
        ]
        wb_table = wandb.Table(columns=table_cols)
        for r in rows:
            wb_table.add_data(*[r[c] for c in table_cols])
        wandb.log({"eval/table": wb_table})

        # Log one short eval clip (if any video produced)
        if os.path.isdir(eval_dir):
            mp4s = sorted([f for f in os.listdir(eval_dir) if f.endswith(".mp4")])
            if mp4s:
                wandb.log({"eval/video": wandb.Video(os.path.join(eval_dir, mp4s[0]), fps=24, format="mp4")})

        # Save checkpoints as an artifact
        art = wandb.Artifact(name="cheetah-agent-recurrent", type="model")
        best_path = os.path.join(ckpt_dir, "best.pt")
        last_path = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(best_path): art.add_file(best_path)
        if os.path.exists(last_path): art.add_file(last_path)
        wandb.log_artifact(art)
        wandb.finish()

    envs.close()
    del envs
    import gc; gc.collect()
            