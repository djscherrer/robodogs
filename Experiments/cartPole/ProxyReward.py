# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Any

import wandb

import gymnasium as gym
import gymnasium.vector
from gymnasium.vector import SyncVectorEnv
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from gymnasium.envs.registration import register

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import RecordVideo

import cartPoleAgent


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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "robodogs-cartpole"
    """the wandb's project name"""
    wandb_entity: str = "robodogs"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPoleParam-Train-v0" # Use our registered env
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
    
    # Environment parameters (passed to gym.vector.make() as kwargs)
    Tc: int = 128                    # calibration steps
    tol_deg: float = 3.0             # soft deviation termination (optional)
    patience: int = 50               # soft deviation termination (optional)
    ignore_during_calib: bool = True # soft deviation termination (optional)
    
    # These are kwargs for CartPoleParam, also passed to gym.vector.make()
    len_range: Tuple[float, float] = (0.5, 1.8)
    mc_range: Tuple[float, float] = (0.5, 2.0)

    output : str = "videos/proxy_demo"  # video output directory

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class CartPoleParam(CartPoleEnv):
    """
    A version of CartPoleEnv where physics parameters are randomized
    on every reset.
    """
    def __init__(self, len_range=(0.5, 1.8), mp_range=(0.3, 1.8),
                 mc_range=(0.5, 2.0), g_range=(8.5, 11.5), **kw):
        super().__init__(**kw)
        self.len_range, self.mp_range, self.mc_range, self.g_range = \
            len_range, mp_range, mc_range, g_range
        # enlarge track; we won’t terminate on x
        self.x_threshold = 5.0
        # print(f"Initialized CartPoleParam with len_range={len_range}, mc_range={mc_range}") # Debug print

    def _sample_params(self, rng: np.random.RandomState):
        """Samples new physics parameters."""
        self.length   = rng.uniform(*self.len_range)     # pole half-length in the base env
        self.masspole = 1.0 #rng.uniform(*self.mp_range) # Using fixed masspole as in your snippet
        self.masscart = rng.uniform(*self.mc_range)
        self.gravity  = 9.81 #rng.uniform(*self.g_range) # Using fixed gravity as in your snippet
        
        # Update parameters used by the superclass
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        # We need to set the seed for the *base* env first
        super().reset(seed=seed) 
        
        # Now, sample parameters using an RNG derived from the seed
        if seed is not None:
            param_seed = seed
        else:
            # If no seed, create one. np_random is the env's RNG
            param_seed = self.np_random.integers(1<<31)
            
        self._sample_params(np.random.RandomState(param_seed))
        
        # Call super().reset() *again* to ensure the *new* params are used
        # to generate the initial state.
        obs, info = super().reset(seed=seed, options=options)
        
        # Expose params for logging
        info = dict(info)
        info.update(dict(length=self.length, masspole=self.masspole,
                         masscart=self.masscart, gravity=self.gravity))
        
        return obs.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Override termination: only angle
        theta = float(obs[2])
        th_thr = self.theta_threshold_radians
        terminated = bool(abs(theta) > th_thr)
        
        # Keep cart within wide bounds for numerical safety
        obs[0] = float(np.clip(obs[0], -self.x_threshold, self.x_threshold))
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info

class SetpointWrapper(gym.Wrapper):
    """
    Wraps the environment to add a setpoint (target angle) theta_star.
    - Adds theta_star to the observation.
    - Modifies the reward to be a squared error cost.
    """
    def __init__(self, env: gym.Env, theta_sampler: Callable[[], float], 
                 weights: Tuple[float, float, float, float] = (5.0, 0.5, 0.1, 0.001)):
        super().__init__(env)
        self.theta_sampler = theta_sampler
        self.w_theta, self.w_omega, self.w_x, self.w_a = weights
        
        # Augment observation space
        low  = np.concatenate([self.observation_space.low,  [-np.pi]])
        high = np.concatenate([self.observation_space.high, [+np.pi]])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        self.theta_star = 0.0

    def reset(self, **kw: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kw)
        self.theta_star = float(self.theta_sampler())
        info = dict(info); info["theta_star"] = self.theta_star
        return np.concatenate([obs, [self.theta_star]]).astype(np.float32), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, term, trunc, info = self.env.step(action)
        if info.get("calib_end", False):
            base = self.env.unwrapped              # CartPoleParam
            # place cart at origin and pole at the target angle
            base.state = np.array([0.0, 0.0, self.theta_star, 0.0], dtype=np.float32)
            obs = base.state.copy()                # keep continuity in the same episode
        x, xdot, th, thdot = map(float, obs)
        
        # Calculate reward
        action_cost = 0.0
        if isinstance(action, np.ndarray):
            action_cost = abs(action[0])
        elif np.isscalar(action):
            action_cost = abs(action)
        
        r = (- self.w_theta * (th - self.theta_star) ** 2
             - self.w_omega * thdot**2
            #  - self.w_x * x**2
             - self.w_a * action_cost)
        
        obs_aug = np.concatenate([obs, [self.theta_star]]).astype(np.float32)
        info = dict(info); info["theta_star"] = self.theta_star
        return obs_aug, float(r), term, trunc, info

class DeviationTerminationWrapper(gym.Wrapper):
    """
    Terminates the episode if the angle `theta` deviates from the
    setpoint `theta_star` by more than `tol_deg` for `patience` steps.
    """
    def __init__(self, env: gym.Env, tol_deg: float = 3.0, patience: int = 50, 
                 ignore_during_calib: bool = True):
        super().__init__(env)
        self.tol = np.deg2rad(tol_deg)
        self.patience = patience
        self.ignore_during_calib = ignore_during_calib
        self.bad_steps = 0
        self.theta_star = 0.0

    def reset(self, **kw: Any) -> Tuple[np.ndarray, dict]:
        self.bad_steps = 0
        obs, info = self.env.reset(**kw)
        # Try to read theta* from upstream wrapper (SetpointWrapper)
        self.theta_star = float(info.get("theta_star", getattr(self, "theta_star", 0.0)))
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, r, term, trunc, info = self.env.step(action)
        
        # Skip during calibration if requested
        in_calib = bool(info.get("calib", False))
        if self.ignore_during_calib and in_calib:
            return obs, r, term, trunc, info

        # Get current target (if upstream changes it mid-episode)
        self.theta_star = float(info.get("theta_star", self.theta_star))
        
        # obs can be (x, xdot, th, thdot) or (x, xdot, th, thdot, th_star)
        theta = float(obs[2])
        
        if abs(theta - self.theta_star) > self.tol:
            self.bad_steps += 1
        else:
            self.bad_steps = 0

        if self.bad_steps >= self.patience:
            term = True # Set terminated flag
            info = dict(info)
            info["terminated_reason"] = "setpoint_deviation"
            info["deviation_steps"] = self.bad_steps

        return obs, r, term, trunc, info

class CalibrationWrapper(gym.Wrapper):
    """
    Adds a calibration period of `Tc` steps at the beginning of an episode.
    During calibration:
    - `info["calib"]` is True.
    - The reward is 0.0.
    - `info["calib_end"]` is True on the last step (t == Tc).
    """
    def __init__(self, env: gym.Env, Tc: int = 128):
        super().__init__(env)
        self.Tc = Tc
        self.t = 0
        
    def reset(self, **kw: Any) -> Tuple[np.ndarray, dict]:
        self.t = 0
        obs, info = self.env.reset(**kw)
        info = dict(info); info["calib"] = self.t < self.Tc; info["calib_end"] = False
        return obs, info
        
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, r, term, trunc, info = self.env.step(action)
        self.t += 1
        info = dict(info)
        
        if self.t <= self.Tc:
            info["calib"] = True
            info["calib_end"] = (self.t == self.Tc)
            r = 0.0 # Zero reward during calibration
        else:
            info["calib"] = False
            info["calib_end"] = False
            
        return obs, r, term, trunc, info
    
def theta_train_sampler() -> float:
    """Samples a target angle from two bands, excluding the center."""
    deg = np.random.uniform(3, 15) * (1 if np.random.rand() < 0.5 else -1)
    return np.deg2rad(deg)

def theta_eval_zero() -> float:
    """Returns a zero-degree target angle."""
    return 0.0

def make_training_env(
    render_mode: Optional[str] = None,
    Tc: int = 128,
    tol_deg: float = 3.0,
    patience: int = 50,
    ignore_during_calib: bool = True,
    **kwargs
) -> gym.Env:
    """
    Entry point for creating the training environment stack.
    `kwargs` are passed to CartPoleParam.
    """
    # Note: `render_mode` is passed to CartPoleParam,
    # other kwargs are split
    
    # Extract CartPoleParam kwargs from **kwargs
    cartpole_param_keys = ["len_range", "mp_range", "mc_range", "g_range"]
    cartpole_kwargs = {"render_mode": render_mode}
    for key in cartpole_param_keys:
        if key in kwargs:
            cartpole_kwargs[key] = kwargs[key]

    env = CartPoleParam(**cartpole_kwargs)
    env = CalibrationWrapper(env, Tc=Tc)
    env = SetpointWrapper(env, theta_sampler=theta_train_sampler)
    env = DeviationTerminationWrapper(
        env, 
        tol_deg=tol_deg, 
        patience=patience, 
        ignore_during_calib=ignore_during_calib
    )
    return env

def make_eval_env(
    render_mode: Optional[str] = None,
    Tc: int = 128,
    tol_deg: float = 3.0,
    patience: int = 50,
    ignore_during_calib: bool = True,
    **kwargs
) -> gym.Env:
    """
    Entry point for creating the evaluation environment stack.
    Uses `theta_eval_zero` sampler.
    `kwargs` are passed to CartPoleParam.
    """
    # Extract CartPoleParam kwargs from **kwargs
    cartpole_param_keys = ["len_range", "mp_range", "mc_range", "g_range"]
    cartpole_kwargs = {"render_mode": render_mode}
    for key in cartpole_param_keys:
        if key in kwargs:
            cartpole_kwargs[key] = kwargs[key]
            
    env = CartPoleParam(**cartpole_kwargs)
    env = CalibrationWrapper(env, Tc=Tc)
    env = SetpointWrapper(env, theta_sampler=theta_eval_zero) # Difference is here
    env = DeviationTerminationWrapper(
        env, 
        tol_deg=tol_deg, 
        patience=patience, 
        ignore_during_calib=ignore_during_calib
    )
    return env


# --- Registration ---
# This is good practice, do it once at the top level
# We set max_episode_steps here, which gym.make() will use
# to add a TimeLimit wrapper.

register(
    id='CartPoleParam-Train-v0',
    entry_point=make_training_env,
    max_episode_steps=1000 
)

register(
    id='CartPoleParam-Eval-v0',
    entry_point='__main__:make_eval_env',
    max_episode_steps=1000
)


def main():
    """
    Demonstrates creating a spec-aware vector env
    """
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(f"Creating vector env with ID: {args.env_id}")

    # Build a vector env manually so we can wrap only env[0] with RecordVideo
    def make_one(i: int):
        def thunk():
            rm = "rgb_array" if (args.capture_video and i == 0) else None
            env = gym.make(
                args.env_id,
                render_mode=rm,
                # kwargs for make_training_env
                Tc=args.Tc,
                tol_deg=args.tol_deg,
                patience=args.patience,
                ignore_during_calib=args.ignore_during_calib,
                # kwargs for CartPoleParam
                len_range=args.len_range,
                mc_range=args.mc_range,
            )
            if args.capture_video and i == 0:
                os.makedirs(args.output, exist_ok=True)
                env = RecordVideo(
                    env,
                    video_folder=args.output,
                    name_prefix="proxy_first_env",
                    episode_trigger=lambda e: True,
                )
            return env
        return thunk

    env_fns = [make_one(i) for i in range(args.num_envs)]
    vec_env = SyncVectorEnv(env_fns)
    # `vec_env.envs` gives you access to the individual envs
    # `outer` is the *outermost* wrapper (likely TimeLimit)
    outer = vec_env.envs[0]
    
    # `unwrapped` drills down to the *innermost* env (CartPoleParam)
    base = outer.unwrapped 

    print(f"\nMax episode steps from spec: {outer.spec.max_episode_steps}")
    device = pick_device()
    print(f"Using device: {device}")
    
    if args.track:
        # Example: Log config to W&B
        # We get max_episode_steps from the *outer* spec
        cfg = {
            "max_episode_steps": outer.spec.max_episode_steps,
            "Tc": args.Tc,
            "tol_deg": args.tol_deg,
            "patience": args.patience,
            # You can't get the *sampled* params here,
            # as reset hasn't been called.
            "len_range": args.len_range,
            "mc_range": args.mc_range,
        }
        print(f"\nLogging to W&B: {cfg}")
        # wandb.config.update({"env_cfg": cfg}, allow_val_change=True)
    
    
    # --- Roll until the first env finishes (or cap at 2000 steps) ---
    print("\nRecording env[0] until done or cap...")
    obs, info = vec_env.reset(seed=args.seed)
    print(f"Reset obs shape: {obs.shape}")
    print(f"Reset info 'theta_star': {info['theta_star']}")
    print("\nCartPoleParam params:")
    print(f"  length: {base.length}")
    print(f"  gravity: {base.gravity}")
    print(f"  masscart: {base.masscart}")
    print(f"  masspole: {base.masspole}")
    print(f"  force_mag: {base.force_mag}")
    print(f"  tau: {base.tau}")


    # Starting main part
    
    # Setting up agent, optimizer, etc.

    agent = cartPoleAgent.GRUAgent(vec_env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + vec_env.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + vec_env.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    H = 128  # GRU hidden size
    h0_actor_buf  = torch.zeros((args.num_steps, args.num_envs, H), device=device)
    h0_critic_buf = torch.zeros((args.num_steps, args.num_envs, H), device=device)

    # 3 Step procedure 

    # 1. Calibration Phase (no reward, Tc steps) / Burn-in hidden state
    print("Starting Calibration Phase...")
    # Initialize hidden states
    # after reset(...)
    h_actor = torch.zeros(1, args.num_envs, H, device=device)
    h_critic = torch.zeros(1, args.num_envs, H, device=device)
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    next_obs, _ = vec_env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    with torch.no_grad():
        for _ in range(args.Tc):
            # policy forward without logging or grads
            action, h_actor, logprob, _, value, h_critic = agent.get_action_and_value(next_obs, None, None)
            # pi, v, h_actor = agent.actor_forward(obs_t, h_actor)
            # _, h_critic = agent.critic_forward(obs_t, h_critic)
            # random actions or greedy, but no learning
            # actions_np = vec_env.action_space.sample()
            next_obs, rew, term, trunc, info = vec_env.step(action.cpu().numpy())
            print(f"Step calib rew: {rew[0]}, term: {term[0]}, trunc: {trunc[0]}")
            # reset hidden where env ended
            done_mask = torch.as_tensor((term | trunc), device=device).view(1, -1, 1)
            h_actor = h_actor * (~done_mask)
            h_critic = h_critic * (~done_mask)
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)

    print("Completed Calibration Phase.")
    # 2. Normal Operation Phase (proxy reward, until termination)
    print("Starting Normal Operation Phase...")
    # We continue from `next_obs`, `h_actor`, `h_critic` produced by calibration.
    # Roll until all envs terminate once or we reach num_steps.

    h_actor = h_actor.detach()
    h_critic = h_critic.detach()
    next_obs = next_obs  # already on device from calibration
    next_done = torch.zeros(args.num_envs, device=device)
    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    t = 0
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    global_update_idx = 0
    # envs = vec_env  # already set earlier
    # next_obs, _ = envs.reset(seed=args.seed)   # do NOT reset again; continue from calibration
    # next_obs = torch.Tensor(next_obs).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # cache hidden for sequence training
            h0_actor_buf[step]  = h_actor.squeeze(0)
            h0_critic_buf[step] = h_critic.squeeze(0)

            with torch.no_grad():
                action, h_actor, logprob, _, value, h_critic = agent.get_action_and_value(
                    next_obs, h_actor=h_actor, h_critic=h_critic
                )
                values[step] = value.flatten()

            actions[step]  = action
            logprobs[step] = logprob

            # step env
            next_obs_np, reward_np, term_np, trunc_np, infos = vec_env.step(action.cpu().numpy())
            next_done_np = np.logical_or(term_np, trunc_np)

            # reset hidden where env finished
            done_mask = torch.as_tensor(next_done_np, device=device, dtype=torch.bool)
            if done_mask.any():
                h_actor[:, done_mask] = 0
                h_critic[:, done_mask] = 0

            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device).view(-1)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            # logging via RecordEpisodeStatistics
            if "episode" in infos:
                ep = infos["episode"]
                ep_r = float(ep["r"].max())
                ep_l = int(ep["l"].max())
                print(f"global_step={global_step}, episodic_return={ep_r}")
                if args.track:
                    wandb.log({"charts/episodic_return": ep_r,
                            "charts/episodic_length": ep_l,
                            "global_step": global_step})
                episodes_done += 1
                # save_last()
                if args.save_every_episodes and (episodes_done % args.save_every_episodes == 0):
                    pass
                    # save_last()
                if ep_r > best_return:
                    best_return = ep_r
                    # save_best()
                    if args.track:
                        wandb.log({"perf/best_return": best_return, "global_step": global_step})

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs)[0].view(-1)
            advantages = torch.zeros_like(rewards, device=device)
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

        # flatten
        b_obs = obs.reshape((-1,) + vec_env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + vec_env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        T, B = args.num_steps, args.num_envs
        env_inds = np.arange(B); np.random.shuffle(env_inds)
        mb_envs_per_batch = B // args.num_minibatches
        clipfracs = []

        for epoch in range(args.update_epochs):
            for start in range(0, B, mb_envs_per_batch):
                mb_envs = env_inds[start:start + mb_envs_per_batch]
                seq_obs     = obs[:, mb_envs]
                seq_actions = actions[:, mb_envs].long()
                seq_oldlp   = logprobs[:, mb_envs]
                seq_returns = returns[:, mb_envs]
                seq_adv     = advantages[:, mb_envs]
                seq_dones   = dones[:, mb_envs]
                seq_oldV    = values.view(T, B)[:, mb_envs]

                # re-run RNN with proper resets along the sequence
                H = h0_actor_buf.shape[-1]
                h_a = torch.zeros(1, len(mb_envs), H, device=device)
                h_c = torch.zeros(1, len(mb_envs), H, device=device)
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

                newlogprob = torch.stack(newlp_list, dim=0)
                newvalue   = torch.stack(newv_list, dim=0)
                entropy    = torch.stack(ent_list, dim=0)

                logratio = newlogprob - seq_oldlp
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                adv = seq_adv
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

        # metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
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
        global_update_idx += 1
    # 3. Evaluation Phase (zero setpoint, until termination) every k episodes

    vec_env.close()
if __name__ == "__main__":
    # Note: Need to change registration entry_point to '__main__:make_training_env'
    # because this file is being run as a script.
    
    # We must check if the envs are already registered
    # to avoid errors on hot-reload
    registered_envs = set(gym.registry.keys())
    
    # if 'CartPoleParam-Train-v0' not in registered_envs:
    #     register(
    #         id='CartPoleParam-Train-v0',
    #         entry_point='__main__:make_training_env',
    #         max_episode_steps=1000 
    #     )

    # if 'CartPoleParam-Eval-v0' not in registered_envs:
    #     register(
    #         id='CartPoleParam-Eval-v0',
    #         entry_point='__main__:make_eval_env',
    #         max_episode_steps=1000
        # )
    
    main()