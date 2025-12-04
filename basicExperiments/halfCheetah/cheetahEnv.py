from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
import gymnasium as gym
from typing import Any, Dict, Optional
import numpy as np
import mujoco
from gymnasium.utils import seeding


class CheetahCustom(HalfCheetahEnv):
    """
    Starts identical to HalfCheetah-v5, with:
      - optional obs padding,
      - upright shaping + early termination when on the back,
      - simple morphology scaling hooks,
      - OPTIONAL domain randomization via `change_every`,
      - obs cast to float32 and observation_space set to float32 to avoid dtype warnings.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        obs_pad: Optional[int] = None,
        terminate_on_back: bool = True,
        # ---- Reward for uprightness ----
        min_upright: float = 0.3,       # terminate if cosine(up_world, up_body) < this
        min_torso_h: float = 0.30,      # terminate if torso COM too low
        upright_bonus_k: float = 0.30,  # smoothing weight added to default reward
        upright_bonus_margin: float = 0.0,
        # ---- Proxy task settings ----
        proxy_period_steps: int = 32,
        proxy_training_steps: int = 128,  # duration over which proxy is learned
        proxy_amplitude: float = 0.10,

        proxy_track_weight: float = 1.0,
        proxy_vel_penalty_weight: float = 0.2,
        # ---- Domain randomization controls ----
        change_every: int = 0,          # 0 => no domain randomization; >0 => randomize every N episodes
        morphology_jitter: float = 0.2, # +/- 20% scaling around 1.0 by default
        seed: Optional[int] = None,
        reset_after_proxy: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._geom_size0 = self.model.geom_size.copy()
        self._body_mass0 = self.model.body_mass.copy()
        self._body_pos0 = self.model.body_pos.copy()
        self._body_inertia0 = self.model.body_inertia.copy()

        self._reset_after_proxy = reset_after_proxy
        # cache common geom/body ids by name (standard HalfCheetah asset names)
        def _gid(nm): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        def _bid(nm): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
        self._ids = {
            "geom": {
                "torso": _gid("torso"),
                "bthigh": _gid("bthigh"), "bshin": _gid("bshin"), "bfoot": _gid("bfoot"),
                "fthigh": _gid("fthigh"), "fshin": _gid("fshin"), "ffoot": _gid("ffoot"),
            },
            "body": {
                "torso": _bid("torso"),
                "bthigh": _bid("bthigh"), "bshin": _bid("bshin"), "bfoot": _bid("bfoot"),
                "fthigh": _bid("fthigh"), "fshin": _bid("fshin"), "ffoot": _bid("ffoot"),
            }
        }

        # ---- obs settings ----
        self.obs_pad = obs_pad
        orig_shape = self.observation_space.shape

        if isinstance(orig_shape, (int, np.integer)):
            orig_dim = int(orig_shape)
        else:
            # Gym / Gymnasium Box normally uses a tuple, e.g. (17,)
            assert len(orig_shape) == 1, f"Expected 1D obs, got shape={orig_shape}"
            orig_dim = int(orig_shape[0])

        base_dim = orig_dim + 3
        dim = obs_pad if obs_pad is not None else base_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(dim,),
            dtype=np.float32,
        )
        # ---- morphology / shaping settings ----
        self._morphology: Dict[str, float] = {}  # name->scale
        self.terminate_on_back = bool(terminate_on_back)
        self.min_upright = float(min_upright)
        self.min_torso_h = float(min_torso_h)
        self.upright_bonus_k = float(upright_bonus_k)
        self.upright_bonus_margin = float(upright_bonus_margin)

        # ---- cached ids ----
        self._torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if self._torso_bid < 0:
            raise RuntimeError("Could not resolve body id for 'torso'. Check MuJoCo version or body name.")

        self._world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # ---- Domain randomization bookkeeping ----
        self.change_every = int(change_every)
        self.morphology_jitter = float(morphology_jitter)
        self._episode_idx = 0
        self.np_random, _ = seeding.np_random(seed)

        # ---- Proxy task reward ----
        self.proxy_period_time = proxy_period_steps * self.dt    
        self.proxy_learning_time = proxy_training_steps * self.dt
        self.proxy_amp_relative = float(proxy_amplitude)
        self.proxy_amp_morphology = None  # to be set on reset
        self.proxy_center = None
        self.proxy_track_weight = float(proxy_track_weight)
        self.proxy_vel_penalty_weight = float(proxy_vel_penalty_weight)

        self._proxy_return = 0.0
        self._real_return = 0.0
        self._last_reward = 0.0
        

    # ---------- utilities ----------
    def _upright_and_height(self):
        """Returns (upright_cos, torso_height)."""
        R = self.data.xmat[self._torso_bid].reshape(3, 3)    # body->world rotation
        torso_up_world = R[:, 2]                             # body +Z in world coords
        upright = float(np.clip(self._world_up.dot(torso_up_world), -1.0, 1.0))
        torso_h = float(self.data.xpos[self._torso_bid][2])  # torso COM height
        return upright, torso_h

    # ---------- morphology hooks ----------
    def set_morphology(self, **scales: float) -> None:
        """
        Store desired *embodiment-only* scales. These are applied idempotently
        (we restore baseline sizes/masses before re-applying).

        Torso:
        - torso_len_scale, torso_rad_scale, torso_mass_scale
        - torso_scale (uniform size; used if *_len/_rad not given)

        Front leg segments (prefix 'f'): fthigh, fshin, ffoot
        Back  leg segments (prefix 'b'): bthigh, bshin, bfoot
        - <seg>_len_scale, <seg>_rad_scale, <seg>_mass_scale

        Leg-level shortcuts (fan out to all segments if per-seg keys are None):
        - fleg_len_scale, fleg_rad_scale, fleg_mass_scale
        - bleg_len_scale, bleg_rad_scale, bleg_mass_scale

        Legacy per-segment uniform (lowest precedence across both legs):
        - thigh_scale, shin_scale, foot_scale
        - front_leg_scale, back_leg_scale (size uniform per leg)
        - torso_scale (as above)
        """
        self._morphology.update(scales)

    def _apply_morphology(self) -> None:
        """Apply size & mass scales to geoms/bodies idempotently (no compounding)."""
        if not self._morphology:
            return

        m, d = self.model, self.data
        g0 = self._geom_size0
        m0 = self._body_mass0
        pos0 = self._body_pos0
        I0 = self._body_inertia0
        ids = self._ids
        S = self._morphology

        # --- reset to baseline first (avoid compounding across episodes) ---
        m.geom_size[:]    = g0
        m.body_mass[:]    = m0
        m.body_pos[:]     = pos0
        m.body_inertia[:] = I0

        # --- small helper: scale capsule/cylinder geom size ---
        def _apply_geom_scale(gid: int, len_scale, rad_scale, uni_scale):
            if gid < 0:
                return
            rs = rad_scale if rad_scale is not None else uni_scale
            ls = len_scale if len_scale is not None else uni_scale
            if rs is not None:
                m.geom_size[gid, 0] = g0[gid, 0] * float(rs)  # radius
            if ls is not None and m.geom_type[gid] in (
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                m.geom_size[gid, 1] = g0[gid, 1] * float(ls)  # half-length

        def _apply_body_mass(bid: int, mass_scale):
            if bid < 0 or mass_scale is None:
                return
            m.body_mass[bid] = m0[bid] * float(mass_scale)

        def _scale_child_offset(parent_body: str, child_body: str, len_scale: Optional[float]):
            if len_scale is None or len_scale == 1.0:
                return
            pid = ids["body"][parent_body]
            cid = ids["body"][child_body]
            if pid < 0 or cid < 0:
                return
            # body_pos[cid] is the position of the child frame in *parent* frame
            v0 = pos0[cid].copy()          # baseline offset
            m.body_pos[cid] = v0 * float(len_scale)

        def _eff_len(seg_key: str, leg_len: Optional[float]) -> float:
            """
            Effective length scale for a given segment:
            - use seg-specific <seg>_len_scale if provided
            - else use leg-level fleg_len_scale / bleg_len_scale if provided
            - else 1.0 (no scaling)
            """
            seg_len = S.get(seg_key)
            if seg_len is not None:
                return float(seg_len)
            if leg_len is not None:
                return float(leg_len)
            return 1.0
        # =========================================================
        # 1) TORSO: length / radius / mass / inertia + hip anchors
        # =========================================================
        torso_gid = ids["geom"]["torso"]
        torso_bid = ids["body"]["torso"]

        torso_len_scale = S.get("torso_len_scale")
        torso_rad_scale = S.get("torso_rad_scale")
        torso_uni_scale = S.get("torso_scale")

        # apply geom scaling for torso (visual / collision)
        _apply_geom_scale(torso_gid, torso_len_scale, torso_rad_scale, torso_uni_scale)

        # choose an effective length scale sL to drive hips + inertia
        sL = None
        if torso_len_scale is not None:
            sL = float(torso_len_scale)
        elif torso_uni_scale is not None:
            sL = float(torso_uni_scale)

        if sL is not None:
            # --- move hip bodies so they sit at the "ends" of the scaled torso ---
            for name in ("fthigh", "bthigh"):
                bid = ids["body"][name]
                if bid < 0:
                    continue
                base = pos0[bid].copy()
                # scale x-offset from torso center
                base[0] *= sL
                m.body_pos[bid] = base

            # --- scale torso mass if user did NOT explicitly provide torso_mass_scale ---
            if S.get("torso_mass_scale") is not None:
                _apply_body_mass(torso_bid, S.get("torso_mass_scale")) 
            else: 
                _apply_body_mass(torso_bid, sL)

            # --- scale torso inertia (rod-like approx: I ∝ mass * length^2, mass ∝ length) => I ∝ length^3 ---
            I_base = I0[torso_bid].copy()
            I_scale = sL ** 3
            m.body_inertia[torso_bid] = I_base * I_scale

        else:
            # no torso length scaling: maybe just a mass scale
            if S.get("torso_mass_scale") is not None:
                _apply_body_mass(torso_bid, S.get("torso_mass_scale"))

        # =========================================================
        # 2) LEGS: keep your previous shortcuts & legacy options
        # =========================================================
        fleg_len  = S.get("fleg_len_scale")
        fleg_rad  = S.get("fleg_rad_scale")
        fleg_mass = S.get("fleg_mass_scale")
        bleg_len  = S.get("bleg_len_scale")
        bleg_rad  = S.get("bleg_rad_scale")
        bleg_mass = S.get("bleg_mass_scale")

        thigh_uni = S.get("thigh_scale")
        shin_uni  = S.get("shin_scale")
        foot_uni  = S.get("foot_scale")
        front_uni = S.get("front_leg_scale")  # size-only
        back_uni  = S.get("back_leg_scale")   # size-only

        # effective length scales per body
        fthigh_L = _eff_len("fthigh_len_scale", fleg_len)
        bthigh_L = _eff_len("bthigh_len_scale", bleg_len)
        fshin_L  = _eff_len("fshin_len_scale",  fleg_len)
        bshin_L  = _eff_len("bshin_len_scale",  bleg_len)

        _scale_child_offset("fthigh", "fshin", fthigh_L)
        _scale_child_offset("bthigh", "bshin", bthigh_L)
        _scale_child_offset("fshin",  "ffoot",  fshin_L)
        _scale_child_offset("bshin",  "bfoot",  bshin_L)

        for seg in ("thigh", "shin", "foot"):
            # front
            gk = f"f{seg}"
            _apply_geom_scale(
                ids["geom"][gk],
                S.get(f"{gk}_len_scale", fleg_len),
                S.get(f"{gk}_rad_scale", fleg_rad),
                (front_uni if front_uni is not None else S.get(f"{seg}_scale"))
            )
            _apply_body_mass(ids["body"][gk], S.get(f"{gk}_mass_scale", fleg_mass))

            # back
            gk = f"b{seg}"
            _apply_geom_scale(
                ids["geom"][gk],
                S.get(f"{gk}_len_scale", bleg_len),
                S.get(f"{gk}_rad_scale", bleg_rad),
                (back_uni if back_uni is not None else S.get(f"{seg}_scale"))
            )
            _apply_body_mass(ids["body"][gk], S.get(f"{gk}_mass_scale", bleg_mass))

        # recompute kinematics/dynamics after modifications
        mujoco.mj_forward(m, d)

    # ---- random morphology sampler (like CartPoleDomainRandEnv._randomize_params) ----
    def _randomize_morphology(self):
        j = self.morphology_jitter
        def sample(): return float(self.np_random.uniform(1.0 - j, 1.0 + j))
        scales = dict(
            # torso
            torso_len_scale=sample(), torso_rad_scale=sample(), torso_mass_scale=sample(),
            # leg-level shortcuts (spread to segments)
            fleg_len_scale=sample(), fleg_rad_scale=sample(), fleg_mass_scale=sample(),
            bleg_len_scale=sample(), bleg_rad_scale=sample(), bleg_mass_scale=sample(),
        )
        self.set_morphology(**scales)

    # ---------- observation helpers ----------
    def _add_obs_and_pad(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32, copy=False)
        t = float(self.data.time)

        # Determine current task
        flag = 0.0 if t <= self.proxy_learning_time else 1.0

        # Target height for current time
        target_h = self._target_height(t)

        extras = np.array(
            [flag, target_h, self._last_reward],
            dtype=np.float32,
        )

        obs = np.concatenate([obs, extras], axis=0)

        # Padding logic (unchanged)
        if self.obs_pad is None:          
            return obs
    
        if obs.shape[0] >= self.obs_pad:
            return obs[: self.obs_pad]

        out = np.zeros((self.obs_pad,), dtype=np.float32)
        out[: obs.shape[0]] = obs

        return out

    # ---------- standard Gym API ----------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self._proxy_return = 0.0
        self._real_return = 0.0
        self._last_reward = 0.0
        # keep vector-env seeding compatible
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        # domain randomization: like CartPoleDomainRandEnv
        if self.change_every > 0:
            # episode index is per-environment
            self._episode_idx += 1
            if (self._episode_idx - 1) % self.change_every == 0:
                # randomize before ep 1, 1+change_every, ...
                self._randomize_morphology()

        obs, info = super().reset(seed=seed, options=options)
        # morphology is already applied in _randomize_morphology();
        # if you ever call set_morphology manually, this keeps it consistent:
        self._apply_morphology()

        # Proxy Reward center initialization
        self.proxy_center = float(self.data.xpos[self._torso_bid][2])
        self.proxy_amp_morphology = self.proxy_amp_relative * self.proxy_center
        return self._add_obs_and_pad(obs), info
    

    def _target_height(self, t: Optional[float] = None) -> float:
        if t is None:
            t = float(self.data.time)
        return float(
            (self.proxy_center * 0.8)
            + self.proxy_amp_morphology * np.sin(2.0 * np.pi * t / self.proxy_period_time) 
        )
    # NOTE: factor 0.75 comes from observation, that the HalfCheetah torso hovers around 75% of its COM height

    def get_proxy_reward(self, torso_h: float) -> float:
        """
        Sinusoidal tracking reward on torso height.
        r = - (h - h_target(t))^2  (optionally scaled)
        """
        t = float(self.data.time)

        # target height according to sine
        target_h = self._target_height(t)

        # Quadratic penalty
        err = torso_h - target_h
        proxy_sensitivity = 0.02
        return np.exp(-float(err * err)/proxy_sensitivity)
    
    def step(self, action: np.ndarray):
        obs, forward_rew, term, trunc, info = super().step(action)

        # diagnostics
        up, h = self._upright_and_height()
        info["upright"] = up
        info["torso_h"] = h

        t = float(self.data.time)
        target_h = self._target_height(t)
        info["target_h"] = target_h

        # stability shaping
        denom = (1.0 - self.upright_bonus_margin + 1e-8)
        upright_piece = max(0.0, (up - self.upright_bonus_margin)) / denom

        # If in training 
        if (t <= self.proxy_learning_time):
            # proxy task reward
            proxy_score = float(self.get_proxy_reward(h))

            # base velocity in x
            vx = float(self.data.qvel[0])
            info["vx"] = vx # usually between +- (0, 1.5)
            vel_penalty = vx * vx
            vel_sensitivity = 0.6 
            vel_penalty_score = np.exp(-vel_penalty/vel_sensitivity)

            alive_bonus = 0 # TODO: do we need this actually, bc now scores are positive?
            rew = self.proxy_track_weight * proxy_score - self.proxy_vel_penalty_weight * vel_penalty_score + alive_bonus
            self._proxy_return += rew
            info["current_proxy_reward"] = rew
            info["proxy_track"] = proxy_score
            info["proxy_vel_penalty"] = -vel_penalty_score
        else:
            # default forward reward + upright bonus
            rew = float(forward_rew + self.upright_bonus_k * upright_piece)
            self._real_return += rew
            info["current_real_reward"] = rew

        self._last_reward = rew
        # early termination when clearly on back / collapsed
        fell = (up < self.min_upright) or (h < self.min_torso_h)
        if self.terminate_on_back and fell:
            term = True
            info["terminated_back"] = True
        else:
            info["terminated_back"] = False

        

        # episode return logging
        if term or trunc:
            if self._reset_after_proxy and float(self.data.time) <= self.proxy_learning_time:
                # Simply reset the environment and do not terminate and skip to real task phase
                info["proxy_return"] = self._proxy_return
                info["real_return"] = 0.0

                # Reset the state to the beginning of real task phase
                obs, _ = super().reset()
                self.data.time = self.proxy_learning_time + self.dt
                term = False
                trunc = False

            else:
                # self._proxy_return = 0.0
                # self._real_return = 0.0
                info["proxy_return"] = self._proxy_return
                info["real_return"] = self._real_return

        return self._add_obs_and_pad(obs), rew, term, trunc, info


# --- vector/eval helpers ---
def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, proxy_period_steps: int, proxy_training_steps: int, proxy_amplitude: float, change_every: int = 0, morphology_jitter: float = 0.2, proxy_track_weight: float = 1.0, proxy_vel_penalty_weight: float = 0.2, reset_after_proxy: bool = False):
    """
    Default env factory. `change_every=0` means no domain randomization.
    To enable domain randomization, pass change_every>0 from your training script.
    """
    def thunk():
        rm = "rgb_array" if (capture_video and idx == 0) else None
        env = CheetahCustom(
            render_mode=rm, 
            change_every=change_every, 
            morphology_jitter=morphology_jitter, 
            proxy_period_steps=proxy_period_steps, 
            proxy_training_steps=proxy_training_steps, 
            proxy_amplitude=proxy_amplitude, 
            reset_after_proxy=reset_after_proxy,
            proxy_track_weight=proxy_track_weight,
            proxy_vel_penalty_weight=proxy_vel_penalty_weight
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda ep: (ep % 10) == 0,
            )
        return env
    return thunk


def make_evaluate_env(env_id: str, video_dir: str | None = None, seed: int = 0, proxy_period_steps: int = 32, proxy_training_steps: int = 128, proxy_amplitude: float = 0.10, reset_after_proxy: bool = False):
    rm = "rgb_array" if video_dir else None
    env = gym.make(
        env_id,
        render_mode=rm,
        proxy_period_steps=proxy_period_steps,
        proxy_training_steps=proxy_training_steps,
        proxy_amplitude=proxy_amplitude,
        reset_after_proxy=reset_after_proxy,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_dir:
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True)
    # env.reset(seed=seed)
    return env