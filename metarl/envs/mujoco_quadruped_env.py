"""
MuJoCo quadruped env with:
- PD low-level control at 200 Hz
- Policy control at 50 Hz
- Imitation rewards vs kinematic reference generator
- Meta-episode memory handled in the algorithm (hidden state persists across K episodes)

This is a *starter*; plug in your MJCF and state extraction.
"""
import time
import math
import numpy as np
import gymnasium as gym
import mujoco


from .gait import contact_phase, phase_embedding, trot_contact_schedule
from .reward import imitation_and_regularization

class QuadrupedMujocoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, model_xml, control_frequency_hz=50, pd_frequency_hz=200,
                 episode_steps=128, terminate_on_fall=True, rsi=True,
                 reward_params=None, obs_include=None):
        super().__init__()
        self.model_xml = model_xml
        self.control_dt = 1.0/float(control_frequency_hz)
        self.pd_dt = 1.0/float(pd_frequency_hz)
        self.inner_steps = int(pd_frequency_hz // control_frequency_hz)
        self.episode_steps = episode_steps
        self.terminate_on_fall = terminate_on_fall
        self.rsi = rsi
        self.reward_params = reward_params or {}
        self.obs_include = obs_include or []

        self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        self.data = mujoco.MjData(self.model)


        self.num_joints = 12
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

        # Example observation size; adjust to your signals
        obs_dim = 128
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._t = 0
        self._step_count = 0
        self._prev_action = np.zeros(self.num_joints, dtype=np.float32)


        # Joint and actuator order you want for the action vector (edit to match your a1.xml)
        self.JOINT_ORDER = [
            "LF_hip_abd", "LF_hip_pitch", "LF_knee",
            "RF_hip_abd", "RF_hip_pitch", "RF_knee",
            "LH_hip_abd", "LH_hip_pitch", "LH_knee",
            "RH_hip_abd", "RH_hip_pitch", "RH_knee",
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.JOINT_ORDER]
        self._auto_joint_limits()
        self.dof_addrs = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]


        # If you use position actuators (recommended), map them as well:
        self.ACT_ORDER = [f"a_{n}" for n in self.JOINT_ORDER]   # or whatever your XML actuator names are
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self.ACT_ORDER]

        # Base body id for sensing
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")


        # Optional foot sites (add in MJCF for clean sensing)
        def sid(name):
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            except Exception:
                return -1
        self.site_ids = {
            "LF": sid("LF_foot_site"),
            "RF": sid("RF_foot_site"),
            "LH": sid("LH_foot_site"),
            "RH": sid("RH_foot_site"),
        }

    # --------- Helpers ---------
    def _auto_joint_limits(self):
        lo, hi = [], []
        for jid in self.joint_ids:
            r0, r1 = self.model.jnt_range[jid]
            if r0 == 0.0 and r1 == 0.0: r0, r1 = -0.5, 0.5  # fallback
            lo.append(r0); hi.append(r1)
        self.joint_low  = np.array(lo, dtype=np.float32)
        self.joint_high = np.array(hi, dtype=np.float32)

    def _action_to_targets(self, a_norm):
        a_norm = np.clip(a_norm, -1.0, 1.0)
        return 0.5 * (a_norm + 1.0) * (self.joint_high - self.joint_low) + self.joint_low

    def _reference(self, t):
        """Kinematic reference generator placeholder. Returns dict with h, v, feet, yawrate, phase emb."""
        # TODO: integrate vcmd, wcmd; foot placement heuristics (Kang et al. Eq. 2-3 in ref [43] of paper)
        h_ref = 0.28
        v_ref = np.array([0.2, 0.0, 0.0])  # forward, vertical, sideways
        feet_ref = np.zeros((4,3))         # target feet positions in body/world frame
        yawrate_ref = 0.0
        ph = contact_phase(t, freq_hz=2.0)
        c,s = phase_embedding(ph)
        return {"h": h_ref, "v": v_ref, "feet": feet_ref, "yawrate": yawrate_ref, "phase": (c,s)}
    
    def _sense_for_reward(self):
        # Base height
        h = float(self.data.xpos[self.base_bid, 2])

        # Base linear velocity (finite-diff world position for simplicity)
        if not hasattr(self, "_prev_xpos"):
            self._prev_xpos = self.data.xpos[self.base_bid].copy()
        cur = self.data.xpos[self.base_bid].copy()
        v_world = (cur - self._prev_xpos) / max(self.control_dt, 1e-6)
        self._prev_xpos = cur

        # Yaw rate (approx from free joint if present; otherwise from cvel/body frame)
        yaw_rate = float(self.data.cvel[self.base_bid, 2])  # body z ang vel; acceptable proxy

        # Feet positions (prefer sites)
        def foot_xyz(tag):
            sid = self.site_ids.get(tag, -1)
            return self.data.site_xpos[sid].copy() if sid >= 0 else np.zeros(3)
        feet = np.vstack([foot_xyz("LF"), foot_xyz("RF"), foot_xyz("LH"), foot_xyz("RH")])

        return {
            "base_height": h,
            "base_linvel": v_world,
            "feet_pos": feet,
            "yaw_rate": yaw_rate,
        }

    def _get_obs(self, ref):
        qpos = self.data.qpos.copy()      # contains floating base + joints
        qvel = self.data.qvel.copy()
        base_xyz = self.data.xpos[self.base_bid].copy()
        base_rotm = self.data.xmat[self.base_bid].reshape(3,3).copy()
        base_omega = self.data.cvel[self.base_bid, :3].copy()   # body ang vel
        gravity_world = np.array([0, 0, -1.0], dtype=np.float32)

        cos_phi, sin_phi = ref["phase"]
        phase_emb = np.array([cos_phi, sin_phi], dtype=np.float32)

        # Commands (if you have a command sampler; here from ref)
        vcmd = ref["v"]      # (3,)
        wcmd = np.array([0,0,ref["yawrate"]], dtype=np.float32)

        obs = np.concatenate([
            qpos.ravel()[:len(qpos)],  # trim/pad as you prefer
            qvel.ravel()[:len(qvel)],
            base_xyz,
            base_omega,
            gravity_world,
            vcmd, wcmd,
            phase_emb,
            self._prev_action,         # a_{t-1}
        ]).astype(np.float32)

        # If you want a fixed size (e.g. 128), pad with zeros:
        dim = self.observation_space.shape[0]
        if obs.shape[0] < dim:
            obs = np.pad(obs, (0, dim - obs.shape[0]))
        else:
            obs = obs[:dim]
        return obs

    def _fallen(self):
        z = self.data.xpos[self.base_bid, 2]
        if z < 0.12:  # tune
            return True
        # Optional: pitch/roll check via base xmat -> derive Euler and threshold
        return False

    # --------- Gym API ---------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0.0
        self._step_count = 0
        self._prev_action[:] = 0.0
        mujoco.mj_resetData(self.model, self.data)

        # RSI: small noise in base height/orientation, joint angles
        noise = 0.01
        self.data.qpos[:3] += self.np_random.uniform(-noise, noise, size=3)  # base pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        ref = self._reference(self._t)
        obs = self._get_obs(ref)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # --- PD inner loop (omitted) ---
        targets = self._action_to_targets(action)
        for _ in range(max(1, self.inner_steps)):
            for i, aid in enumerate(self.actuator_ids):
                self.data.ctrl[aid] = targets[i]   # position actuators
            mujoco.mj_step(self.model, self.data)

        self._t += self.control_dt
        self._step_count += 1

        # Reference & observations
        ref = self._reference(self._t)
        obs_vec = self._get_obs(ref)             # flat vector for the policy

        obs_for_reward = self._sense_for_reward()  # << must be a dict
        # SAFETY: assert required keys exist
        assert isinstance(obs_for_reward, dict)
        for k in ("base_height", "base_linvel", "feet_pos", "yaw_rate"):
            if k not in obs_for_reward:
                raise KeyError(f"Reward dict missing '{k}'. Got keys: {list(obs_for_reward.keys())}")

        r = imitation_and_regularization(
            obs=obs_for_reward,
            act=action,
            prev_act=self._prev_action,
            refs=ref,
            sigma=self.reward_params.get("sigma", {}),
            weights=self.reward_params.get("weights", {}),
        )
        self._prev_action = action

        terminated = False
        truncated = self._step_count >= self.episode_steps
        if self.terminate_on_fall and self._fallen():
            terminated = True

        info = {"ref_phase": ref["phase"]}
        return obs_vec, float(r), terminated, truncated, info

    # def render(self):
    #     pass
    def render(self):
        """Live viewer using mujoco.viewer if available; fallback to GLFW if not."""
        # Try the official viewer first (works with pip mujoco>=3.x)
        try:
            from mujoco import viewer  # <-- explicit import of submodule
            if not hasattr(self, "_viewer") or self._viewer is None:
                # Set your backend before launching:
                #   export MUJOCO_GL=glfw   # local interactive
                #   export MUJOCO_GL=egl    # headless GPU
                #   export MUJOCO_GL=osmesa # CPU
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return
        except Exception as e:
            # Fall back to manual GLFW renderer if the official viewer isn't available
            import traceback
            _viewer_err = "".join(traceback.format_exception_only(type(e), e)).strip()

        # ---------- GLFW fallback (MuJoCo 2.3.x / no viewer build) ----------
        try:
            import glfw
        except Exception as e:
            raise RuntimeError(
                f"Failed to use mujoco.viewer ({_viewer_err}) and glfw is missing. "
                "Install fallback with: pip install glfw"
            ) from e

        if not glfw.init():
            raise RuntimeError("GLFW init failed. On headless, set MUJOCO_GL=egl or osmesa.")

        if not hasattr(self, "_window") or self._window is None:
            self._win_w, self._win_h = 1280, 720
            self._window = glfw.create_window(self._win_w, self._win_h, "MuJoCo Viewer", None, None)
            if not self._window:
                glfw.terminate()
                raise RuntimeError("Failed to create GLFW window.")
            glfw.make_context_current(self._window)

            self._cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self._cam)
            self._opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(self._opt)
            self._scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self._context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

            # camera follow trunk
            self._cam.lookat[:] = self.data.xpos[self.base_bid]
            self._cam.distance = 1.8; self._cam.elevation = -15; self._cam.azimuth = 180

        if glfw.window_should_close(self._window):
            return
        fb_w, fb_h = glfw.get_framebuffer_size(self._window)
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        self._cam.lookat[:] = self.data.xpos[self.base_bid]

        mujoco.mjv_updateScene(self.model, self.data, self._opt, None, self._cam,
                            mujoco.mjtCatBit.mjCAT_ALL, self._scene)
        mujoco.mjr_render(viewport, self._scene, self._context)
        glfw.swap_buffers(self._window)
        glfw.poll_events()


    def close(self):
        # close official viewer
        try:
            from mujoco import viewer as _v
            if hasattr(self, "_viewer") and self._viewer is not None:
                try: self._viewer.close()
                except Exception: pass
                self._viewer = None
        except Exception:
            pass
        # close GLFW fallback
        if hasattr(self, "_window") and self._window is not None:
            import glfw
            try:
                mujoco.mjr_freeContext(getattr(self, "_context", None))
                mujoco.mjv_freeScene(getattr(self, "_scene", None))
            except Exception:
                pass
            glfw.destroy_window(self._window)
            glfw.terminate()
            self._window = None