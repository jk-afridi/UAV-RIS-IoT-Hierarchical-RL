# UAV-mounted RIS-assisted IoT Networks: A Hierarchical DRL Approach

This repository contains the official implementation of our research on optimizing Age of Information (AoI) and Decode Rate in IoT networks using a UAV-mounted Reconfigurable Intelligent Surface (RIS). 

We propose a **Hierarchical Deep Reinforcement Learning (DRL)** framework consisting of a PPO-based macro-agent for UAV trajectory optimization and a SAC-based micro-agent for RIS phase alignment, accounting for realistic Phase-Dependent Amplitude (PDA) hardware impairments.

## ⚠️ Status of the results
The headline numbers in earlier versions of this README (10 % AoI reduction, 3.7x decode gain, "80 % of the
theoretical upper bound") were produced by an environment in which the UAV position did not affect the
channel, and were compared only against random RIS phases. See [`AUDIT.md`](AUDIT.md) for the audit, the fixes
on this branch, and a multi-seed benchmark of strong non-learning baselines (alternating optimisation +
AoI-greedy scheduling) that any learned method must now beat.

## 🏗️ Repository Structure
```text
├── src/
│   ├── algorithms/       # Core DRL agents (PPO, SAC, Hierarchical)
│   ├── core/             # Environment physics, AoI logic, and PDA models
│   ├── train/            # Final training loops for all baselines
│   └── experiments/      # Figure generation and statistical analysis scripts
├── results/              # Final evaluation metrics and generated figures
├── docs/                 # Research proposals and implementation roadmaps
├── tests/                # regression tests for the audit findings
└── README.md
```

*(Note: Legacy scripts and heavy historical outputs are preserved on the `archive/legacy-scripts` and `archive/historical-data` branches).*

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jk-afridi/UAV-RIS-IoT-Hierarchical-RL.git
   cd UAV-RIS-IoT-Hierarchical-RL
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Running the Code

### 1. Training the Agents
You can train the three main architectures (Proposed, PPO-Only, Random) by running the scripts in the `src/train/` directory.

```bash
# Train the proposed Hierarchical (PPO + SAC) agent
python -m src.train.train_hierarchical

# Train the PPO-Only baseline (Trajectory optimization only)
python -m src.train.train_ppo_only_baseline

# Train the Random Baseline
python -m src.train.train_random_baseline
```

### 2. Generating Publication Figures
Once training is complete and results are stored in the `results/` folder, you can generate the 6 IEEE-formatted comparison figures:

```bash
python -m src.experiments.generate_4way_figures
```
Figures will be outputted in both `.png` and `.pdf` formats in the `results/figures_4way/` directory.

### 3. Running the paper experiments (Colab, GPU)
See [`PAPER_PLAN.md`](PAPER_PLAN.md) for the plan and decision rules and `colab/paper_experiments.ipynb` (or
`bash scripts/run_colab.sh quick|full`).

### 4. Evaluating (multi-seed, per-seed layouts)
```bash
python -m src.eval.evaluate --seeds 5 --episodes 3          # non-learning baselines
python -m src.eval.evaluate_drl --ckpt_dir results/hierarchical/<run> --seeds 5   # trained PPO+SAC
```
