from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
import gymnasium as gym
from typing import Any, Dict, Optional
import numpy as np


class CheetahCustom(HalfCheetahEnv):
    """
    Starts identical to HalfCheetah-v5.
    You can later modify geometry (leg length, etc.) in _apply_morphology().
    Optional: pad observations to a fixed size via obs_pad.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, obs_pad: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.obs_pad = obs_pad
        if obs_pad is not None:
            high = np.full((obs_pad,), np.inf, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        self._morphology: Dict[str, float] = {}  # will stay empty for now

    # ---- hooks you'll fill in later ----
    def set_morphology(self, **scales: float) -> None:
        """Store desired scales (e.g., thigh_scale=1.2)."""
        self._morphology.update(scales)

    def _apply_morphology(self) -> None:
        if not self._morphology:
            return
        # TODO: edit self.model.geom_size[...] or body_pos[...] based on the ids
        # Example (pseudo):
        # gid = self.model.geom_name2id("femur")
        # self.model.geom_size[gid, 0] *= self._morphology.get("thigh_scale", 1.0)
        pass

    # ---- standard Gym API with optional padding ----
    def _pad(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_pad is None:
            return obs.astype(np.float32)
        d = self.obs_pad
        obs = obs.astype(np.float32)
        return np.pad(obs, (0, d - obs.shape[0])) if obs.shape[0] < d else obs[:d]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        # Apply morphology on first reset (or do it in __init__ if you prefer)
        self._apply_morphology()
        return self._pad(obs), info

    def step(self, action: np.ndarray):
        obs, rew, term, trunc, info = super().step(action)
        return self._pad(obs), float(rew), bool(term), bool(trunc), info


# --- vector-eval helpers (unchanged logic, just pointing to your env id) ---
def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    def thunk():
        rm = "rgb_array" if (capture_video and idx == 0) else None
        env = gym.make(env_id, render_mode=rm)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return thunk


def make_evaluate_env(env_id: str, video_dir: str | None = None, seed: int = 0):
    rm = "rgb_array" if video_dir else None
    env = gym.make(env_id, render_mode=rm)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_dir:
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True)
    env.reset(seed=seed)
    return env