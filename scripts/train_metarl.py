"""
Train Meta-RL with recurrent PPO on MuJoCo env, mapping from your PyLoCo training flow.
"""
import json
import argparse
import numpy as np
import torch
from metarl.envs.mujoco_quadruped_env import QuadrupedMujocoEnv
from metarl.policies.gru_policy import GRUPolicy
from metarl.algorithms.ppo_rnn import RecurrentPPO, PPOConfig
from metarl.utils.rollout_buffer import RecurrentRolloutBuffer

def main(args):
    with open(args.config, "r") as f:
        params = json.load(f)
    envp = params["environment_params"]
    rew = params["reward_params"]
    hyp = params["train_hyp_params"]

    env = QuadrupedMujocoEnv(
        model_xml=envp["model_xml"],
        control_frequency_hz=envp["control_frequency_hz"],
        pd_frequency_hz=envp["pd_frequency_hz"],
        episode_steps=envp["episode_steps"],
        terminate_on_fall=envp["terminate_on_fall"],
        rsi=envp["rsi"],
        reward_params=rew,
        obs_include=envp["obs_include"]
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GRUPolicy(obs_dim, act_dim, hidden_state_size=16)
    algo = RecurrentPPO(policy, PPOConfig(
        gamma=hyp["gamma"],
        learning_rate=hyp["learning_rate"],
        clip_range=hyp["clip_range"],
        entropy_coef=hyp["entropy_coef"],
        value_coef=hyp["value_coef"],
        update_epochs=hyp["update_epochs"],
        sequence_length=hyp["sequence_length"],
        meta_episode_length=hyp["meta_episode_length"],
        batch_size=hyp["batch_size"]
    ))

    steps_per_update = hyp["n_steps"]
    total_steps = hyp["time_steps"]
    buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim)
    rng = np.random.RandomState(hyp.get("random_seed",42))

    obs, _ = env.reset(seed=hyp.get("random_seed",42))
    for t in range(total_steps):
        # simple random action to fill buffer initially (replace with policy sample after wiring evaluate/logp)
        action = rng.uniform(-1,1,size=act_dim).astype(np.float32)
        next_obs, rew, term, trunc, info = env.step(action)
        # placeholders for logp, value
        lp = 0.0; v = 0.0
        buf.add(obs, action, rew, term or trunc, lp, v)
        obs = next_obs
        if (t+1) % steps_per_update == 0:
            buf.compute_returns_advantages(gamma=hyp["gamma"])
            algo.update(buf)
            buf = RecurrentRolloutBuffer(steps_per_update, obs_dim, act_dim)
            if (t+1) % hyp.get("n_steps", steps_per_update) == 0:
                print(f"Trained up to step {t+1}")

    torch.save(policy.state_dict(), "checkpoints/policy_final.pt")
    print("✅ Saved trained policy to checkpoints/policy_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/metarl_default.json")
    args = parser.parse_args()
    main(args)
