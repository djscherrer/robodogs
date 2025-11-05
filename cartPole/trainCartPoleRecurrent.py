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

from gymnasium.envs.registration import register
from cartPole import cartPoleEnv, cartPoleAgent, evaluateCartPole


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
    wandb_project_name: str = "robodogs-cartpole"
    """the wandb's project name"""
    wandb_entity: str = None #"robodogs"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPoleCustom-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
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
    target_kl: float = None
    """the target KL divergence threshold"""

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
    global_ckpt_dir = f"cartpole/basicExperiments/checkpoints"


def return_config(env: gym.Env):
    base = env.unwrapped
    cfg = {
        "length": base.length,
        "gravity": base.gravity,
        "masscart": base.masscart,
        "masspole": base.masspole,
        "force_mag": base.force_mag,
        "tau": base.tau,
        "max_episode_steps": base.spec.max_episode_steps,
    }
    return cfg

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    register(
        id=args.env_id,
        entry_point=cartPoleEnv.CartPoleCustom,
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
    device = "cpu" # ML stuff to tiny to have an impact
    print(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [cartPoleEnv.make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    outer = envs.envs[0] 
    base = outer.unwrapped        # == CartPoleCustom


    cfg = {
        "length": base.length,
        "gravity": base.gravity,
        "masscart": base.masscart,
        "masspole": base.masspole,
        "force_mag": base.force_mag,
        "tau": base.tau,
        "max_episode_steps": base.spec.max_episode_steps,
    }
    text = "\n".join(f"{k}: {v}" for k, v in cfg.items())

    if args.track:
        wandb.config.update({"env_cfg": cfg}, allow_val_change=True)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # GRU Agent init
    agent = cartPoleAgent.GRUAgent(envs).to(device)
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
            "best_return": best_return,
            "episodes_done": episodes_done,
            "global_step": global_step,
            "global_update_idx": global_update_idx,
            "args": vars(args),
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
        ckpt = torch.load(args.resume, map_location=device)
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
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    H = 128  # GRU hidden size
    h0_actor_buf  = torch.zeros((args.num_steps, args.num_envs, H), device=device)
    h0_critic_buf = torch.zeros((args.num_steps, args.num_envs, H), device=device)


    # hidden states for GRU
    h_critic = torch.zeros(1, args.num_envs, 128).to(device)
    h_actor = torch.zeros(1, args.num_envs, 128).to(device)

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
                print(f"global_step={global_step}, episodic_return={ep_r}")

                if args.track:
                    wandb.log(
                        {
                            "charts/episodic_return": ep_r,
                            "charts/episodic_length": ep_l,
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
        # b_inds = np.arange(args.batch_size)
        # clipfracs = []
        # for epoch in range(args.update_epochs):
        #     np.random.shuffle(b_inds)
        #     h_a = torch.zeros(1, 128, 128).to(device)
        #     h_c = torch.zeros(1, 128, 128).to(device)
        #     for start in range(0, args.batch_size, args.minibatch_size):
        #         end = start + args.minibatch_size
        #         mb_inds = b_inds[start:end]
        #         mb = b_inds[start:start+args.minibatch_size]
        #         h_a = torch.zeros(1, len(mb), 128, device=device)
        #         h_c = torch.zeros(1, len(mb), 128, device=device)
        #         _, _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
        #             b_obs[mb], h_actor=h_a, h_critic=h_c, action=b_actions.long()[mb]
        #         )
        #         logratio = newlogprob - b_logprobs[mb_inds]
        #         ratio = logratio.exp()

        #         with torch.no_grad():
        #             old_approx_kl = (-logratio).mean()
        #             approx_kl = ((ratio - 1) - logratio).mean()
        #             clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        #         mb_advantages = b_advantages[mb_inds]
        #         if args.norm_adv:
        #             mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        #         # Policy loss
        #         pg_loss1 = -mb_advantages * ratio
        #         pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        #         pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        #         # Value loss
        #         newvalue = newvalue.view(-1)
        #         if args.clip_vloss:
        #             v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        #             v_clipped = b_values[mb_inds] + torch.clamp(
        #                 newvalue - b_values[mb_inds],
        #                 -args.clip_coef,
        #                 args.clip_coef,
        #             )
        #             v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        #             v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        #             v_loss = 0.5 * v_loss_max.mean()
        #         else:
        #             v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        #         entropy_loss = entropy.mean()
        #         loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        #         optimizer.zero_grad()
        #         loss.backward()
        #         nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        #         optimizer.step()
            # assume obs, actions, logprobs, values, rewards, dones are [T, B, ...]
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
                seq_actions = actions[:, mb_envs].long()                     # [T,M]
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

        global_update_idx += 1

    rets, lens = evaluateCartPole.evaluate_on_fixed_scenarios(agent, args.env_id, device, 6, video_root=f"videos/{run_name}-eval", seed=args.seed+100)
    print("eval/return_mean:", rets.mean(), "eval/len_mean:", lens.mean())

    # === F) Eval metrics + eval video to W&B ===
    if args.track:
        art = wandb.Artifact(name="cartpole-agent-recurrent", type="model")
        best_path = f"{ckpt_dir}/best.pt"
        last_path = f"{ckpt_dir}/last.pt"
        if os.path.exists(best_path):
            art.add_file(best_path)
        if os.path.exists(last_path):
            art.add_file(last_path)
        wandb.log_artifact(art)
    
        wandb.log(
            {
                "eval/return_mean": float(rets.mean()),
                "eval/len_mean": float(lens.mean()),
            }
        )
        # Log one short eval clip (adjust filename if needed)
        eval_dir = f"videos/{run_name}-eval"
        if os.path.isdir(eval_dir):
            mp4s = sorted([f for f in os.listdir(eval_dir) if f.endswith(".mp4")])
            if mp4s:
                wandb.log({"eval/video": wandb.Video(os.path.join(eval_dir, mp4s[0]), fps=24, format="mp4")})

        wandb.finish()

    envs.close()
        