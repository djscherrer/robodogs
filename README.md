# MetaRL-MuJoCo (starter)

A minimal scaffold to reproduce the **MetaLoco** pipeline — working-memory meta-RL + motion imitation — using **MuJoCo**.

> Paper references in code comments point to MetaLoco (Zargarbashi et al., 2024).
> This repo maps your uploaded JAX/PyLoCo scripts (`main.py`, `train_metaRL.py`, `train_decoder.py`, `onnx_conversion.py`, `main_decoder.py`)
> into a PyTorch-based structure (recurrent PPO with GRU).

## Top-level ideas

- **Policy**: GRU(16) + MLP → joint position targets (12-DoF quadruped).
- **Meta-episode**: Keep GRU hidden state across K episodes (default K=5).
- **Reward**: Motion imitation (`r_I`) × regularizers (`r_R`), with RBF kernels for tracking (Eq. (2)-(3) in paper).
- **Morphology**: Procedural scaling of link dimensions/masses in U[0.5, 1.5] with cube inertia approximation (Eq. (4)).
- **Domain Randomization**: friction, gravity tilt, actuator latency, sensor noise, impulses.
- **Deployment**: At train-time use kinematic reference generator; at run-time policy + gait planner (no reference).

## Quick start

```bash
pip install -e .
python scripts/train_metarl.py --config configs/metarl_default.json
```

This is a **starter**; fill TODOs to plug in your actual MuJoCo model and assets.
