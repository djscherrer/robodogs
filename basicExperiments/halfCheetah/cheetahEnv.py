from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
import gymnasium as gym
from typing import Any, Dict, Optional
import numpy as np
import mujoco


class CheetahCustom(HalfCheetahEnv):
    """
    Starts identical to HalfCheetah-v5, with:
      - optional obs padding,
      - upright shaping + early termination when on the back,
      - simple morphology scaling hooks,
      - obs cast to float32 and observation_space set to float32 to avoid dtype warnings.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        obs_pad: Optional[int] = None,
        terminate_on_back: bool = True,
        min_upright: float = 0.3,       # terminate if cosine(up_world, up_body) < this
        min_torso_h: float = 0.30,      # terminate if torso COM too low
        upright_bonus_k: float = 0.30,  # smoothing weight added to default reward
        upright_bonus_margin: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        # ---- obs settings ----
        self.obs_pad = obs_pad
        # make the observation_space say float32 (Gym default here is float64)
        base_shape = self.observation_space.shape
        if obs_pad is None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=base_shape, dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_pad,), dtype=np.float32
            )

        # ---- morphology / shaping settings ----
        self._morphology: Dict[str, float] = {}  # name->scale
        self.terminate_on_back = bool(terminate_on_back)
        self.min_upright = float(min_upright)
        self.min_torso_h = float(min_torso_h)
        self.upright_bonus_k = float(upright_bonus_k)
        self.upright_bonus_margin = float(upright_bonus_margin)

        # ---- cached ids ----
        # robust name->id resolution for body "torso"
        self._torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if self._torso_bid < 0:
            raise RuntimeError("Could not resolve body id for 'torso'. Check MuJoCo version or body name.")

        self._world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # ---------- utilities ----------
    def _upright_and_height(self):
        """Returns (upright_cos, torso_height)."""
        # data.xmat is (nbody, 9) row-major rotation matrices
        R = self.data.xmat[self._torso_bid].reshape(3, 3)    # body->world rotation
        torso_up_world = R[:, 2]                             # body +Z in world coords
        upright = float(np.clip(self._world_up.dot(torso_up_world), -1.0, 1.0))
        torso_h = float(self.data.xpos[self._torso_bid][2])  # torso COM height
        return upright, torso_h

    # ---------- morphology hooks ----------
    def set_morphology(self, **scales: float) -> None:
        """
        Store desired scales (e.g., thigh_scale=1.2).
        Supported keys (applied to geoms if present): 
            'thigh_scale', 'shin_scale', 'foot_scale', 'torso_scale'
        Applied to both front/back legs when applicable.
        """
        self._morphology.update(scales)

    def _apply_morphology(self) -> None:
        """Best-effort scaling of common HalfCheetah geoms by name."""
        if not self._morphology:
            return

        # Map of scale-key -> list of geom names to scale (radius/length via geom_size[:,0])
        groups = {
            "thigh_scale": ["bthigh", "fthigh"],
            "shin_scale":  ["bshin", "fshin"],
            "foot_scale":  ["bfoot", "ffoot"],
            "torso_scale": ["torso"],  # note: body has multiple geoms in some mjcfs; this is safe if present
        }

        for key, names in groups.items():
            s = float(self._morphology.get(key, 1.0))
            if abs(s - 1.0) < 1e-8:
                continue
            for gname in names:
                try:
                    gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                    if gid >= 0:
                        # geom_size: (ngeom, 3) — for capsules/cylinders, size[0] is radius, size[1] is half-length
                        # We scale length-like dimension conservatively: both radius and half-length.
                        self.model.geom_size[gid, 0] *= s
                        if self.model.geom_type[gid] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                            self.model.geom_size[gid, 1] *= s
                except Exception:
                    # Name may not exist; skip silently.
                    pass

        # After editing model parameters, let MuJoCo recompute derived data
        mujoco.mj_forward(self.model, self.data)

    # ---------- observation helpers ----------
    def _pad(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32, copy=False)
        if self.obs_pad is None:
            return obs
        d = self.obs_pad
        if obs.shape[0] >= d:
            return obs[:d]
        out = np.zeros((d,), dtype=np.float32)
        out[: obs.shape[0]] = obs
        return out

    # ---------- standard Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        # Apply morphology once at the beginning of the first episode (safe to call multiple times).
        self._apply_morphology()
        return self._pad(obs), info

    def step(self, action: np.ndarray):
        obs, rew, term, trunc, info = super().step(action)

        # diagnostics
        up, h = self._upright_and_height()
        info["upright"] = up
        info["torso_h"] = h

        # stability shaping (in [0,1], softened by margin)
        denom = (1.0 - self.upright_bonus_margin + 1e-8)
        upright_piece = max(0.0, (up - self.upright_bonus_margin)) / denom
        rew = float(rew + self.upright_bonus_k * upright_piece)

        # early termination when clearly on back / collapsed
        fell = (up < self.min_upright) or (h < self.min_torso_h)
        if self.terminate_on_back and fell:
            term = True
            info["terminated_back"] = True
        else:
            info["terminated_back"] = False

        return self._pad(obs), rew, term, trunc, info


# --- vector/eval helpers ---
def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    def thunk():
        rm = "rgb_array" if (capture_video and idx == 0) else None
        env = CheetahCustom(render_mode=rm)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            # record every episode from env #0
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda ep: True)
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