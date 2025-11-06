#!/usr/bin/env python3
"""
HalfCheetah RL Harness (Gymnasium + MuJoCo)
-------------------------------------------

A plug‑and‑play setup to evaluate your existing policy (FFN/GRU) on
Gymnasium's MuJoCo HalfCheetah‑v4 environment.

Key features
------------
- Uses Gymnasium API (reset -> step) and MuJoCo physics under the hood.
- Optional observation padding/truncation to a fixed size (default 128)
  to match your quadruped env shapes.
- Action space is already Box([-1, 1]^6) in HalfCheetah‑v4, matching
  typical policy heads; no PD inner loop needed.
- Simple evaluation loop that supports recurrent state (meta‑episode
  memory) via a (obs, hidden) -> (action, next_hidden) policy interface.
- Minimal dependencies: gymnasium[mujoco], numpy. (torch/jax optional.)

Install notes
-------------
    pip install gymnasium[mujoco]
    # If rendering locally, ensure a GL backend is available.

Run
---
    python halfcheetah_harness.py --episodes 5 --render 0

Replace `RandomPolicy` with your model. See `YourPolicyAdapter` for a
ready‑made interface shim.
"""
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym



# -----------------------------
# Environment adapter
# -----------------------------
class HalfCheetahAdapter(gym.Env):
    """Wraps Gymnasium's HalfCheetah‑v4 and optionally pads obs to fixed size.

    This mirrors the style of your QuadrupedMujocoEnv enough to be a drop‑in
    for rollout/eval code that expects a fixed‑dimensional observation vector
    and an action in [-1, 1].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, obs_dim: Optional[int] = 128, render_mode: Optional[str] = None):
        super().__init__()
        # Underlying Gymnasium env (MuJoCo physics)
        self.env = gym.make("HalfCheetah-v5", render_mode=render_mode)
        self.obs_dim_target = obs_dim

        # Expose spaces
        self.action_space = self.env.action_space  # Box([-1,1], shape=(6,))
        base_obs_space = self.env.observation_space  # Box(shape=(17,))

        if obs_dim is None:
            self.observation_space = base_obs_space
        else:
            high = np.full((obs_dim,), np.inf, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        self._last_obs_raw: Optional[np.ndarray] = None

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_dim_target is None:
            return obs.astype(np.float32)
        d = self.obs_dim_target
        obs = obs.astype(np.float32)
        if obs.shape[0] < d:
            return np.pad(obs, (0, d - obs.shape[0]))
        return obs[:d]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs_raw = obs
        return self._pad_obs(obs), info

    def step(self, action: np.ndarray):
        # Gymnasium handles clipping internally when stepping; clip defensively
        action = np.asarray(action, dtype=np.float32)
        if isinstance(self.action_space, gym.spaces.Box):
            lo, hi = self.action_space.low, self.action_space.high
            action = np.clip(action, lo, hi)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs_raw = obs
        return self._pad_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

