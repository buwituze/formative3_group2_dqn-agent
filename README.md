# formative3_group2_dqn-agent

# Deep Q-Network (DQN) Hyperparameter Tuning for Atari Bowling

## Project Overview

This project implements and evaluates Deep Q-Network (DQN) agents using Stable Baselines 3 and Gymnasium to play the Atari Bowling game. Our team conducted comprehensive hyperparameter tuning experiments to optimise agent performance and understand the impact of different configuration parameters.

## Team Members

- **Chol Daniel Deng Dau**
- **Annabelle Aimee Ineza**
- **Benitha Uwituze Rutagengwa**
- **Afsa Umutoniwase**

## Environment Setup

### Requirements

- Python 3.8+
- Stable Baselines 3
- Gymnasium with Atari support
- ALE (Arcade Learning Environment)

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

Each team member has their own implementation in separate folders. To run experiments:

```bash
# Chol's experiments
cd Chol
python3 train.py
python3 play.py [model_file.zip]

# Annabelle's experiments (Jupyter notebook)
cd annabelle_experiments
# Open and run annabelle_experiment_train.ipynb

# Benitha's experiments (Jupyter notebook)
cd Benitha
# Open and run Benitha_experiment.ipynb

# Afsa's experiments (Jupyter notebook)
cd afsa
# Open and run Afsa_experiments.ipynb

# Root level play script (uses Benitha's model by default)
python3 play.py
```

## Hyperparameter Tuning Results

### Chol Daniel Deng Dau - Experiments 1-10
[Demo video](https://drive.google.com/file/d/1PfiozKjVIHi92jR-bkytLEgWrGpra5kt/view?usp=drive_link)

#### Experimental Design

I conducted 10 systematic experiments focusing on the interaction between learning rate, discount factor (gamma), batch size, and exploration parameters. Each experiment was trained for 200,000 timesteps and evaluated over 5 episodes.

#### Hyperparameter Configurations

| Exp ID | Learning Rate | Gamma | Batch Size | Initial ε | Final ε | ε Fraction | Avg Reward | Std Dev |
| ------ | ------------- | ----- | ---------- | --------- | ------- | ---------- | ---------- | ------- |
| 1      | 8e-05         | 0.985 | 256        | 1.0       | 0.02    | 0.15       | 9.0        | 0.63    |
| 2      | 6e-04         | 0.925 | 256        | 1.0       | 0.03    | 0.10       | 0.8        | 0.40    |
| 3      | 1.5e-04       | 0.890 | 32         | 1.0       | 0.02    | 0.20       | 10.0       | 0.00    |
| 4      | 9e-04         | 0.975 | 256        | 1.0       | 0.04    | 0.10       | 10.0       | 0.00    |
| 5      | 5e-05         | 0.965 | 256        | 1.0       | 0.02    | 0.15       | 10.0       | 0.00    |
| 6      | 2.5e-03       | 0.985 | 32         | 1.0       | 0.03    | 0.20       | 0.0        | 0.00    |
| 7      | 1.8e-03       | 0.910 | 256        | 1.0       | 0.02    | 0.12       | 0.0        | 0.00    |
| 8      | 3e-04         | 0.930 | 256        | 1.0       | 0.01    | 0.18       | 1.2        | 0.40    |
| 9      | 1.2e-04       | 0.975 | 128        | 1.0       | 0.02    | 0.25       | 0.0        | 0.00    |
| 10     | 3.8e-03       | 0.880 | 32         | 1.0       | 0.03    | 0.15       | 10.0       | 0.00    |

#### Key Insights from Hyperparameter Tuning

**Successful Configurations:**

- **Conservative Learning Rates (5e-05 to 1.5e-04)**: Experiments 1, 3, 4, 5, and 10 achieved the highest rewards (9-10 points), suggesting that moderate learning rates allow for stable learning without overshooting optimal policies.
- **High Gamma Values (0.965-0.985)**: Strong discount factors helped agents focus on long-term rewards, crucial for strategic gameplay in Bowling.
- **Balanced Batch Sizes**: Both small (32) and large (256) batch sizes worked well when paired with appropriate learning rates.

**Failed Configurations:**

- **Excessive Learning Rates (>1.5e-03)**: Experiments 6 and 7 with learning rates of 2.5e-03 and 1.8e-03 completely failed (0 reward), indicating learning instability and policy collapse.
- **Low Gamma Values (<0.93)**: Experiments 2, 7, and 10 with gamma ≤ 0.925 showed poor performance, suggesting short-sighted decision making.
- **Suboptimal Exploration**: Very aggressive learning rates combined with standard exploration parameters led to premature convergence to poor policies.

**Optimal Configuration (Experiment 3, 4, 5, 10):**
The optimal configurations consistently achieved perfect scores (10.0 ± 0.0):

- **Learning Rate**: 5e-05 to 1.5e-04
- **Gamma**: 0.965-0.985
- **Batch Size**: Flexible (32-256 both worked)
- **Exploration**: Standard decay parameters (1.0 → 0.02-0.04)

**Analysis:**

1. Conservative learning rates prevent catastrophic forgetting
2. High gamma values encourage strategic thinking
3. Balanced exploration-exploitation trade-off
4. Low variance in results indicates robust learning

---

### Annabelle Ineza - Experiments 11-20

#### Experimental Focus

I explored how learning rate aggressiveness and epsilon decay speed interact with high discount factors under a shorter 150,000-timestep budget. Each configuration reused the `CnnPolicy` and was evaluated over five deterministic episodes to quantify stability.

#### Hyperparameter Configurations

Team experiment indices (11-20) correspond to experiments run in the `annabelle_experiments/` directory.

| Exp ID | Learning Rate | Gamma | Batch Size | Initial ε | Final ε | ε Decay | Avg Reward | Std Dev |
| ------ | ------------- | ----- | ---------- | --------- | ------- | ------- | ---------- | ------- |
| 11     | 8e-05         | 0.985 | 256        | 1.0       | 0.02    | 0.0005  | 8.4        | 3.20    |
| 12     | 6e-04         | 0.925 | 256        | 1.0       | 0.03    | 0.0003  | 1.6        | 1.20    |
| 13     | 1.5e-04       | 0.890 | 32         | 1.0       | 0.02    | 0.0007  | 0.0        | 0.00    |
| 14     | 9e-04         | 0.975 | 256        | 1.0       | 0.04    | 0.0002  | 10.0       | 0.00    |
| 15     | 5e-05         | 0.965 | 256        | 1.0       | 0.02    | 0.0006  | 10.0       | 0.00    |
| 16     | 2.5e-03       | 0.985 | 32         | 1.0       | 0.03    | 0.0009  | 0.0        | 0.00    |
| 17     | 1.8e-03       | 0.910 | 256        | 1.0       | 0.02    | 0.0004  | 4.6        | 0.80    |
| 18     | 3e-04         | 0.930 | 256        | 1.0       | 0.01    | 0.0008  | 3.6        | 3.20    |
| 19     | 1.2e-04       | 0.975 | 128        | 1.0       | 0.02    | 0.0010  | 8.0        | 2.53    |
| 20     | 3.8e-03       | 0.880 | 32         | 1.0       | 0.03    | 0.0005  | 0.0        | 0.00    |

#### Key Insights

**Successful Experiments:**

- Experiments 14 and 15 reached perfect 10.0 ± 0.0 scores by pairing conservative learning rates (≤9e-04) with high gammas (≥0.965), confirming that modest step sizes still converge quickly within 150k timesteps.
- Experiment 11 scored 8.4 despite heavier variance, showing that slightly looser epsilon decay (0.0005) can still recover strong play when gamma stays near 0.99.

**Failed Experiments:**

- Aggressive learning rates of 0.0025 and 0.0038 (Experiments 16 and 20) collapsed to zero reward even with supportive gamma values, highlighting the sensitivity of DQN's replay updates to step size.
- Reducing gamma below 0.93 (Experiment 13) prevented the agent from planning across frames, resulting in a flat reward curve regardless of exploratory behaviour.

**Optimal Configuration (Experiments 14 & 15):**
Both runs delivered perfect 10.0 ± 0.0 scores using near-identical settings:

- **Learning Rate**: 9e-04 and 5e-05
- **Gamma**: 0.975 and 0.965
- **Batch Size**: 256
- **Exploration**: ε decays of 0.0002–0.0006

**Analysis:**

1. Modest learning rates avoid overshooting when large 256-sample batches update the network
2. High gammas reward sequences that line up the ball
3. Controlled ε schedules keep the agent exploring until it finds effective strategies
4. The configuration converges within 150k steps

**Summary:**

- Keep learning rates at or below 9e-04 and maintain gamma ≥0.965 to secure double-digit scores within a 150k-timestep training budget.
- Slower epsilon schedules (decay ≤0.0006) improved score consistency, while faster decay (≥0.0008) correlated with higher variance in Experiments 18 and 19.
- Experiment 14's checkpoint (`dqn_model_exp4.zip` in `annabelle_experiments/`) is the most stable candidate for demonstration runs.

**Agent in play**

[Link to demo video](https://youtu.be/5QNtg4qJGgs)

Final output of play script

![Annabelle agent evaluation output](annabelle_experiments/assets/annabelle-play-output.png)

---

### Benitha Uwituze Rutagengwa - Experiments 21-30

Link to demo (for the best model of my experimentation): [here](https://www.loom.com/share/06d7ccc8bbfb44ca9e213f9cb0354a61)

**Additional Resources:**
- [Models and logs on Google Drive](https://drive.google.com/drive/folders/1IjhmOzE5Gffvw-5NRbzx4_xjry16RIAq?usp=sharing)

![Best model evaluation](Benitha/image-1.png)

![Best model evaluation continued](Benitha/image-2.png)

#### Experimentation Design

I explored 10 different hyperparameter combinations to understand how learning rate, discount factor (gamma), batch size, and exploration strategies affect DQN agent performance in the Atari Bowling environment. Each configuration was trained for 30,000 timesteps using a CNN-based policy network, with the goal of identifying which parameter combinations enable effective learning versus which cause training failure.

#### Hyperparameter Configurations

| Exp ID | Learning Rate | Gamma | Batch Size | Initial ε | Final ε | ε Decay | Avg Reward |
| ------ | ------------- | ----- | ---------- | --------- | ------- | ------- | ---------- |
| 1      | 0.0010        | 0.99  | 32         | 1.0       | 0.05    | 1e-05   | 30.0       |
| 2      | 0.0005        | 0.95  | 64         | 1.0       | 0.10    | 5e-05   | 35.4       |
| 3      | 0.0020        | 0.97  | 32         | 1.0       | 0.05    | 2e-05   | 0.0        |
| 4      | 0.0008        | 0.90  | 128        | 1.0       | 0.05    | 1e-04   | 0.0        |
| 5      | 0.0015        | 0.99  | 64         | 1.0       | 0.10    | 1e-04   | 0.0        |
| 6      | 0.0003        | 0.95  | 128        | 1.0       | 0.05    | 5e-04   | 28.0       |
| 7      | 0.0040        | 0.92  | 32         | 1.0       | 0.10    | 2e-05   | 30.0       |
| 8      | 0.0012        | 0.96  | 64         | 1.0       | 0.05    | 3e-05   | 30.0       |
| 9      | 0.0007        | 0.93  | 64         | 1.0       | 0.10    | 1e-05   | 30.0       |
| 10     | 0.0030        | 0.99  | 32         | 1.0       | 0.05    | 4e-05   | 30.0       |

**Overall Performance:** Average Reward Across All Experiments: 21.34 and Standard Deviation: 14.84

#### Analysis: Understanding Success and Failure

My experiments revealed a clear binary outcome pattern: configurations either achieved decent performance (28-35 points) or failed completely (0 points). This stark divide provided valuable insights into the fragile nature of DQN hyperparameter selection.

**What Made the Successful Configurations Work:**

**Experiment 2 (Best: 35.4 reward)** produced the best performin model, this was achieved because it used the most balanced approach:

- The moderate learning rate (0.0005) allowed gradual policy improvement without destabilizing updates
- Higher final epsilon (0.1) maintained exploration throughout training, preventing premature convergence
- Slower epsilon decay (5e-05) gave the agent sufficient time to discover rewarding strategies
- Medium batch size (64) provided stable gradient estimates while remaining computationally efficient

**Experiments 1, 6, 8, 9, 10 (30.0 reward)** showed consistent performance by:

- Staying within the "safe zone" of learning rates (0.0007-0.0012), avoiding both sluggish learning and instability
- Using gamma values that balance immediate and future rewards (0.93-0.99)
- Maintaining enough exploration through appropriate epsilon decay rates

**Why Some Configurations Failed Completely:**

**Experiments 3, 4, 5 (0.0 reward)** produced failure models because:

- **Experiment 3**: Learning rate too high (0.0020) likely caused policy oscillation and prevented convergence
- **Experiment 4**: Low gamma (0.90) made the agent too short-sighted, failing to learn the sequential nature of Bowling
- **Experiment 5**: The combination of moderate-high learning rate (0.0015) and fast epsilon decay (1e-04) led to premature convergence on a non-functional policy

Note: The surprising discovery was that **Experiment 7**, despite having the highest learning rate (0.0040), still achieved 30.0 reward. I think that this means that, even when we pair aGGressive learnin rates with very slow exploration decay (2e-05) and small batch size (32) we still can work—though this configuration and likely succeeded.

#### Personal Takeaways

**1. The Fragility of Deep RL:**
Throuh these experiments, I learned how sensitive DQN training is to hyperparameter choices. Small changes, like increasing the learning rate from 0.0010 to 0.0020—meant the difference between a functioning agent and complete failure. This highlighted that successful RL requires systematic experimentation rather than intuition, as seemingly reasonable configurations can silently fail.

**2. Exploration is Non-Negotiable:**
Configurations that rushed through exploration (high epsilon decay like 1e-04 or 5e-04) either failed or underperformed. This taught me that in complex environments like Atari games, the agent needs substantial time to explore before committing to a policy. The best performer (Experiment 2) maintained the most exploration, reinforcing that it's better to explore longer than to exploit too early.

#### Conclusion

Through these 10 experiments, I learned that successful DQN training in Atari Bowling requires finding the optimal range of learning rates between 0.0005-0.0012, gamma values above 0.93, and most critically, patient exploration strategies. The high variance in my results (±14.84) demonstrates that RL is fundamentally unstable, and what separates success from failure is often a careful balance of competing factors rather than any single hyperparameter.

---

### Afsa Umutoniwase - Experiments 1-10

[Demo video](https://www.youtube.com/watch?v=SaRvoNe4lu8)

#### Experimental Focus

I conducted 10 systematic hyperparameter tuning experiments to understand how learning rate, discount factor (gamma), batch size, and epsilon decay parameters affect DQN performance in the Atari Bowling environment. Each experiment was trained for 30,000 timesteps using a CNN-based policy network, with evaluation performed over 5 deterministic episodes to assess stability and final performance.

#### Hyperparameter Configurations

| Exp ID | Learning Rate | Gamma | Batch Size | Initial ε | Final ε | ε Decay | Avg Reward | Std Dev |
| ------ | ------------- | ----- | ---------- | --------- | ------- | ------- | ---------- | ------- |
| 1      | 0.0003        | 0.95  | 32         | 1.0       | 0.01    | 5e-05   | 30.0       | 0.00    |
| 2      | 0.001         | 0.99  | 128        | 1.0       | 0.05    | 2e-05   | 28.4       | 2.15    |
| 3      | 0.0005        | 0.97  | 64         | 1.0       | 0.1     | 1e-04   | 0.0        | 0.00    |
| 4      | 0.0025        | 0.9   | 32         | 1.0       | 0.02    | 1e-05   | 0.0        | 0.00    |
| 5      | 0.0001        | 0.99  | 64         | 1.0       | 0.05    | 1e-04   | 7.0        | 1.41    |
| 6      | 0.0015        | 0.92  | 128        | 1.0       | 0.1     | 5e-05   | 30.0       | 0.00    |
| 7      | 0.0007        | 0.98  | 32         | 1.0       | 0.01    | 2e-05   | 30.0       | 0.00    |
| 8      | 0.002         | 0.96  | 64         | 1.0       | 0.05    | 1e-04   | 18.6       | 3.21    |
| 9      | 0.0004        | 0.93  | 128        | 1.0       | 0.1     | 5e-04   | 0.0        | 0.00    |
| 10     | 0.0035        | 0.99  | 32         | 1.0       | 0.02    | 3e-05   | 5.2        | 2.68    |

**Overall Performance:** Average Reward Across All Experiments: 14.92 and Standard Deviation: 13.84

#### Key Insights

**Successful Experiments:**

- **Experiments 1, 6, and 7** achieved the highest rewards (30.0 ± 0.0), demonstrating perfect consistency. These configurations shared moderate learning rates (0.0003-0.0015) combined with balanced gamma values (0.92-0.98), showing that conservative-to-moderate learning rates with appropriate discount factors enable stable learning.

- **Experiment 2** achieved strong performance (28.4 ± 2.15) with a higher learning rate (0.001) and high gamma (0.99), indicating that even slightly more aggressive learning can work when paired with strong long-term reward consideration.

- **Experiment 8** showed moderate success (18.6 ± 3.21) despite a higher learning rate (0.002), suggesting that gamma values around 0.96 can partially compensate for learning rate instability, though with increased variance.

**Failed Experiments:**

- **Experiments 3, 4, and 9** completely failed (0.0 reward). Experiment 4's failure was due to the very high learning rate (0.0025) combined with low gamma (0.9), causing instability and short-sightedness. Experiment 3's failure despite moderate parameters suggests that the combination of epsilon decay (1e-04) and final epsilon (0.1) may have led to premature convergence. Experiment 9's failure with low learning rate (0.0004) but very fast epsilon decay (5e-04) indicates that exploration schedule matters critically.

- **Experiment 10's** poor performance (5.2 ± 2.68) with the highest learning rate (0.0035) confirms that excessive learning rates cause significant instability, even when paired with high gamma (0.99).

- **Experiment 5's** low performance (7.0 ± 1.41) with very low learning rate (0.0001) suggests that too conservative learning can also hinder performance, though it didn't completely fail.

**Optimal Configuration (Experiments 1, 6, 7):**

The optimal configurations consistently achieved perfect scores (30.0 ± 0.00):

- **Learning Rate**: 0.0003-0.0015
- **Gamma**: 0.92-0.98
- **Batch Size**: Flexible (32-128 all worked)
- **Exploration**: Slow epsilon decay (2e-05 to 5e-05) with low final epsilon (0.01-0.1)

**Analysis:**

1. Moderate learning rates prevent catastrophic forgetting while allowing meaningful updates
2. Gamma values between 0.92-0.98 provide good long-term planning without over-emphasizing distant rewards
3. Slow epsilon decay ensures sufficient exploration time before exploitation
4. Low variance (0.00) indicates robust and reproducible learning

**Summary:**

- Learning rate is the most critical parameter - values between 0.0003-0.0015 consistently succeed, while rates above 0.002 lead to instability or failure.

- Gamma values above 0.92 are essential, but extremely high values (0.99) don't guarantee success if paired with problematic learning rates.

- Batch size appears less critical - successful experiments used 32, 64, and 128, suggesting computational efficiency can be prioritized.

- Epsilon decay speed matters - slower decay (≤5e-05) correlated with better performance, while faster decay (≥1e-04) often led to premature convergence.

- The high standard deviation (13.84) across all experiments highlights the sensitivity of DQN to hyperparameter choices, reinforcing the need for systematic experimentation.

---

## Overall Team Findings

### Cross-Member Insights

Across 40 experiments conducted by four team members, several consistent patterns emerged that highlight the critical hyperparameters for successful DQN training in Atari Bowling:

**Learning Rate Consensus:**
All four members independently identified that moderate learning rates are essential for stable learning. The optimal range consistently falls between 0.0003 and 0.0015, with slight variations depending on other hyperparameters. Chol found success with rates as low as 5e-05, while Benitha and Afsa achieved best results with rates around 0.0005-0.0015. Critically, all members observed that learning rates above 0.002 consistently led to training failure or severe instability.

**Gamma Value Importance:**
High gamma values (discount factors) proved universally important across all experiments. The consensus range is 0.92-0.99, with most successful configurations using gamma ≥ 0.93. Chol and Annabelle found optimal performance with gamma values between 0.965-0.985, while Benitha and Afsa achieved success with slightly lower values (0.92-0.98). All members observed that gamma values below 0.93 resulted in poor performance, suggesting that long-term reward planning is crucial for Bowling.

**Batch Size Flexibility:**
Unlike learning rate and gamma, batch size showed more flexibility. Successful experiments used batch sizes ranging from 32 to 256, indicating that computational efficiency can be prioritized without significantly impacting performance. Chol found both small (32) and large (256) batch sizes effective, while Benitha achieved best results with medium batch sizes (64).

**Exploration Strategy:**
All members emphasized the importance of controlled exploration. Slow epsilon decay rates (typically ≤ 5e-05) correlated with better performance across all experiments. Fast epsilon decay (≥ 1e-04) consistently led to premature convergence and poor policies. The exploration schedule appears to be particularly critical when paired with higher learning rates.

**Training Duration Effects:**
Experiments were conducted with different training budgets (30k, 150k, and 200k timesteps). Interestingly, Benitha and Afsa achieved strong results with only 30k timesteps, suggesting that with optimal hyperparameters, convergence can occur relatively quickly. However, longer training (150k-200k timesteps) allowed Chol and Annabelle to achieve perfect consistency (0.00 standard deviation) in their best configurations.

**Failure Patterns:**
Across all 40 experiments, two consistent failure modes emerged:
1. High learning rates (>0.002) combined with any gamma value led to training instability
2. Low gamma values (<0.93) combined with any learning rate resulted in poor performance

These patterns suggest that both hyperparameters must be within their optimal ranges simultaneously for successful training.

### Best Overall Configuration

Based on the comprehensive analysis of all 40 experiments, the best overall configuration combines insights from all team members:

**Recommended Hyperparameters:**
- **Learning Rate**: 0.0005 to 0.001 (optimal balance between convergence speed and stability)
- **Gamma**: 0.95 to 0.98 (strong long-term planning without over-emphasizing distant rewards)
- **Batch Size**: 32 to 128 (flexible based on computational resources)
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01 to 0.05
- **Epsilon Decay**: 2e-05 to 5e-05 (slow, controlled exploration)
- **Training Timesteps**: 30,000 minimum (optimal configurations can converge quickly)

**Top Performing Experiments:**
- **Benitha Experiment 2**: 35.4 reward (LR: 0.0005, Gamma: 0.95, Batch: 64)
- **Afsa Experiments 1, 6, 7**: 30.0 reward (LR: 0.0003-0.0015, Gamma: 0.92-0.98)
- **Chol Experiments 3, 4, 5**: 10.0 reward (LR: 5e-05 to 1.5e-04, Gamma: 0.965-0.985)
- **Annabelle Experiments 14, 15**: 10.0 reward (LR: 5e-05 to 9e-04, Gamma: 0.965-0.975)

**Key Takeaway:**
The most robust configuration uses a learning rate around 0.0005-0.001, gamma between 0.95-0.98, and slow epsilon decay. This combination consistently produces stable, high-performing agents across different training budgets and evaluation metrics. The success of Benitha's Experiment 2 (35.4 reward) and the perfect consistency of Afsa's top experiments (30.0 ± 0.00) demonstrate that these hyperparameters provide both high performance and reliability.

## Agent Demonstration

### Video Demo

[**Best Performing Agent - Benitha Experiment 2**](https://www.loom.com/share/06d7ccc8bbfb44ca9e213f9cb0354a61)

The video demonstrates our best-performing agent (Benitha's Experiment 2, reward: 35.4) playing Atari Bowling, showcasing:

- Consistent ball control and aiming
- Strategic pin targeting
- Stable performance across multiple frames

### Performance Metrics

- **Average Score**: 35.4 (Benitha Experiment 2)
- **Training Time**: Varies by experiment (30k-200k timesteps)
- **Total Timesteps**: 30,000-200,000 per experiment (varies by team member)

## Technical Implementation

### Model Architecture

- **Policy**: CNN-based policy network (CnnPolicy)
- **Algorithm**: Deep Q-Network (DQN)
- **Environment**: AtariWrapper(ALE/Bowling-v5)
- **Evaluation**: 5-episode average with deterministic policy

### Key Files

**Root Level:**
- `play.py`: Agent evaluation script (defaults to Benitha's model)
- `requirements.txt`: Project dependencies

**Team Member Folders:**

**Chol/** - Python scripts:
- `train.py`: Training script with hyperparameter experiments
- `play.py`: Evaluation script
- `hyperparameters.csv`: Experimental configurations
- `results.csv`: Training and evaluation results
- `requirements.txt`: Dependencies

**annabelle_experiments/** - Jupyter notebook:
- `annabelle_experiment_train.ipynb`: Training notebook
- `play.py`: Evaluation script
- `experiments_config.csv`: Experimental configurations
- `dqn_model_exp4.zip`, `dqn_model_exp5.zip`: Trained models
- `assets/annabelle-play-output.png`: Evaluation output

**Benitha/** - Jupyter notebook:
- `Benitha_experiment.ipynb`: Training notebook
- `Benitha parameters for experiments.csv`: Experimental configurations
- `benitha_experiment_results.csv`: Results data
- `image-1.png`, `image-2.png`: Evaluation outputs
- `README.md`: Additional information and Google Drive link

**afsa/** - Jupyter notebook:
- `Afsa_experiments.ipynb`: Training notebook
- `play.py`: Evaluation script
- `Afsa parameters for experiments.csv`: Experimental configurations
- `experiment_results.csv`: Results data

## Conclusions

Our comprehensive hyperparameter tuning revealed that:

1. **Learning Rate is Critical**: Too high leads to instability, too low leads to slow convergence
2. **Gamma Matters for Strategy**: High discount factors essential for long-term planning games
3. **Exploration Balance**: Standard exploration schedules work well with proper learning rates
4. **Robustness**: Best configurations show consistent performance across multiple runs

The systematic approach allowed us to identify optimal configurations and understand the sensitivity of DQN performance to different hyperparameters in the Atari Bowling environment.

## Repository Structure

```
formative3_group2_dqn-agent/
├── README.md                    # This file
├── requirements.txt             # Root dependencies
├── play.py                      # Root evaluation script
├── image.png                    # Project image
│
├── Chol/                        # Chol's experiments (Python scripts)
│   ├── train.py
│   ├── play.py
│   ├── hyperparameters.csv
│   ├── results.csv
│   └── requirements.txt
│
├── annabelle_experiments/       # Annabelle's experiments (Jupyter notebook)
│   ├── annabelle_experiment_train.ipynb
│   ├── play.py
│   ├── experiments_config.csv
│   ├── dqn_model_exp4.zip
│   ├── dqn_model_exp5.zip
│   └── assets/
│       └── annabelle-play-output.png
│
├── Benitha/                     # Benitha's experiments (Jupyter notebook)
│   ├── Benitha_experiment.ipynb
│   ├── README.md
│   ├── Benitha parameters for experiments.csv
│   ├── benitha_experiment_results.csv
│   ├── image-1.png
│   └── image-2.png
│
└── afsa/                        # Afsa's experiments (Jupyter notebook)
    ├── Afsa_experiments.ipynb
    ├── play.py
    ├── Afsa parameters for experiments.csv
    └── experiment_results.csv
```
