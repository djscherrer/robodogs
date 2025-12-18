# Develop meta-RL policy for varying morphologies using proxy and task training

This repository contains the code and report for our Data Science Lab (DSL) project at ETH Zurich.
We study how structured proxy tasks can bootstrap representation learning in recurrent meta-reinforcement learning for morphology-adaptive locomotion.

Our approach trains a GRU-based PPO policy in a two-phase episode structure:
(1) a dense-feedback proxy task, followed by
(2) a downstream locomotion task.
The proxy task provides morphology-sensitive interaction signals that help the recurrent policy infer embodiment-dependent dynamics without access to privileged information.

## Project Summary

Learning locomotion policies that generalize across robot morphologies remains challenging, particularly when task rewards are sparse. While recurrent meta-RL approaches can implicitly infer morphology from interaction history, they often struggle to identify relevant dynamics early in training.

We address this by introducing a proxy-task interaction phase that:

- exposes morphology-dependent dynamics,
- provides dense and structured feedback,
- and shapes the recurrent hidden state before optimizing the main locomotion objective.

We evaluate our approach primarily on randomized HalfCheetah morphologies in MuJoCo and show improved training stability and generalization compared to baselines without proxy-task interaction.

## Repository Structure

```
Experiments/
├── cartPole/ # Early experiments and prototyping
├── halfCheetah/ # Main experiments (proxy task + locomotion)
│ ├── cheetahAgent.py # Policy and agent definition
│ ├── cheetahEnv.py # Custom HalfCheetah environment
│ ├── trainCheetah.py # Training script
│ ├── evaluateCheetah.py # Evaluation script
│ └── trainingLogic.py # Shared training utilities
│
├── checkpoints/ # Saved model checkpoints (generated)
├── runs/ # Training logs (TensorBoard / W&B)
├── videos/ # Recorded rollouts and visualizations
├── wandb/ # Weights & Biases logs (if enabled)
│
├── report.pdf # Final project report (added after submission)
├── poster.pdf # Project poster (added after submission)
├── environment.yml # Conda environment specification
├── LICENSE # MIT License
└── README.md # Project documentation
```

Note:
Directories such as checkpoints/, runs/, videos/, and wandb/ are created automatically during training and are therefore not necessarily present in a fresh clone of the repository.

## Environments

- HalfCheetah (primary):
  Main experimental environment with randomized morphologies. Used to evaluate the effectiveness of proxy-task interaction for morphology adaptation.
- CartPole (secondary):
  Used for early prototyping and validation of training logic and recurrent policies before scaling to locomotion tasks.

## Installation

We recommend using Conda.

```
conda env create -f environment.yml
conda activate <env-name>
```

Make sure MuJoCo is installed and properly configured on your system.

## Running Experiments

Train a policy with proxy-task interaction:

`python Experiments/halfCheetah/trainCheetah.py`

Evaluate a trained model:

`python Experiments/halfCheetah/evaluateCheetah.py`

# Results

Our experiments show that proxy-task interaction:

- improves training stability across random seeds,
- leads to higher final locomotion returns,
- and improves generalization to unseen and asymmetric morphologies.

Detailed experimental results and analysis are provided in report.pdf.

## Reproducibility

- All experiments use fixed random seeds.
- Hyperparameters are documented in the report appendix.
- Evaluation is performed on a fixed set of held-out morphologies.

## Authors & Contributions

- Constantin Pinkl – Methodology, Software, Visualization, Investigation, Writing
- David Scherrer – Methodology, Software, Visualization, Investigation, Writing
- Fatemeh Zargarbashi – Conceptualization, Supervision
- Arnout Devos – Supervision, Methodology, Project Administration

## License

This project is released under the MIT License.
See LICENSE for details.

## Acknowledgements

This project was conducted as part of the Data Science Lab (DSL) at ETH Zurich.
We thank the ETH AI Center and the Computational Robotics Lab for their support.
