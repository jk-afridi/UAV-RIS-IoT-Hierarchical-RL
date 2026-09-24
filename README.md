# UAV-mounted RIS-assisted IoT Networks: A Hierarchical DRL Approach

This repository contains the official implementation of our research on optimizing Age of Information (AoI) and Decode Rate in IoT networks using a UAV-mounted Reconfigurable Intelligent Surface (RIS). 

We propose a **Hierarchical Deep Reinforcement Learning (DRL)** framework consisting of a PPO-based macro-agent for UAV trajectory optimization and a SAC-based micro-agent for RIS phase alignment, accounting for realistic Phase-Dependent Amplitude (PDA) hardware impairments.

## 🚀 Key Results
Our hierarchical approach simultaneously minimizes AoI and maximizes the successful decode rate, proving robust even under severe PDA hardware impairments:
- **10% reduction** in Average AoI compared to random baselines.
- **3.7× improvement** in Decode Rate compared to trajectory-only (PPO-only) optimization.
- Achieves **~80% of the theoretical upper bound** of ideal, flawless RIS hardware.

*(Detailed publication-ready figures can be found in the `results/figures_4way/` directory).*

## 🏗️ Repository Structure
```text
├── src/
│   ├── algorithms/       # Core DRL agents (PPO, SAC, Hierarchical)
│   ├── core/             # Environment physics, AoI logic, and PDA models
│   ├── train/            # Final training loops for all baselines
│   └── experiments/      # Figure generation and statistical analysis scripts
├── results/              # Final evaluation metrics and generated figures
├── docs/                 # Research proposals and implementation roadmaps
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
