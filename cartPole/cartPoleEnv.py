from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import gymnasium as gym
from gymnasium.utils import seeding

class CartPoleCustom(CartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # # Physics knobs
        # self.gravity = 10.0          # default 9.8
        # self.masscart = 1.0          # default 1.0
        self.masspole = 0.3          # default 0.1
        self.length = 0.7            # half-length (m); default 0.5
        # self.force_mag = 10.0        # push force; default 10.0
        # self.tau = 0.01              # seconds between state updates

        # # Recompute derived terms
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length


class CartPoleDomainRandEnv(CartPoleEnv):
   
    def __init__(self, change_every: int = 5, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.change_every = int(change_every)
        self._episode_idx = 0
        self.np_random, _ = seeding.np_random(seed)
        # initial params
        self._randomize_params()

    # ---- sampling policy: adjust ranges as you like ----
    def _randomize_params(self):
        # sample around Gym defaults with wider support
        self.gravity   = float(self.np_random.uniform(7.0, 13.0))   # default 9.8
        self.masscart  = float(self.np_random.uniform(0.8, 1.2))    # default 1.0
        self.masspole  = float(self.np_random.uniform(0.05, 0.5))   # default 0.1
        self.length    = float(self.np_random.uniform(0.3, 1.0))    # default 0.5 
        self.force_mag = float(self.np_random.uniform(8.0, 14.0))   # default 10.0
        self.tau       = float(self.np_random.uniform(0.01, 0.03))  # default 0.02

        # recompute derived terms
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # allow per-reset seeding from vector env
            self.np_random, _ = seeding.np_random(seed)
        # increment episode counter and randomize when required
        self._episode_idx += 1
        if (self._episode_idx - 1) % self.change_every == 0:
            # randomize before ep 1, 6, 11, ...
            self._randomize_params()
        return super().reset(seed=seed, options=options)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            # Apply RecordEpisodeStatistics FIRST
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # Then apply RecordVideo SECOND
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            # Apply RecordEpisodeStatistics here as well
            env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def make_random_env(env_id, idx, capture_video, run_name, change_every=5):
    def thunk():
        # give each worker a different seed and change cadence
        env = gym.make(env_id, render_mode="rgb_array" if (capture_video and idx == 0) else None,
                       change_every=change_every, seed=42 + idx * 9973)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return thunk

def make_evaluate_env(env_id, video_dir=None, seed=0) -> gym.wrappers.RecordVideo | gym.wrappers.RecordEpisodeStatistics:
    render_mode = "rgb_array" if video_dir else None
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_dir:
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True)
    env.reset(seed=seed)
    return env