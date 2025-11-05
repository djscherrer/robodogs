from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import gymnasium as gym

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

def make_evaluate_env(env_id, video_dir=None, seed=0) -> gym.wrappers.RecordVideo | gym.wrappers.RecordEpisodeStatistics:
    render_mode = "rgb_array" if video_dir else None
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_dir:
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True)
    env.reset(seed=seed)
    return env