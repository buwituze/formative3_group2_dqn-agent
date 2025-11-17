import gymnasium as gym
from gymnasium import error as gym_error
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import pandas as pd
import numpy as np
import ale_py

def train_dqn(exp_num, params, timesteps=150_000):
    print(f"\n===== Training Experiment {exp_num} =====")
    print(params)

    def make_env(render_mode=None):
        try:
            env_local = gym.make("ALE/Bowling-v5", render_mode=render_mode)
        except gym_error.NamespaceNotFound as err:
            raise RuntimeError(
                "Gym can't find the ALE namespace. Install Atari support via ",
                "`pip install \"gymnasium[atari]\" autorom[accept-rom-license]` ",
                "and run `AutoROM --accept-license`."
            ) from err
        except gym_error.Error as err:
            raise RuntimeError(
                "Failed to create ALE/Bowling-v5. Double-check Atari packages and ROMs."
            ) from err
        return AtariWrapper(env_local)

    env = make_env(render_mode=None)

    exploration_fraction = params.get("eps_fraction", None)
    if exploration_fraction is None:
        eps_decay = params.get("eps_decay", None)
        if eps_decay is not None and eps_decay > 0:
            exploration_fraction = (params.get("eps_start", 1.0) - params.get("eps_end", 0.02)) / (eps_decay * timesteps)
            exploration_fraction = max(0.001, min(1.0, exploration_fraction))
        else:
            exploration_fraction = 0.1

    policy_name = params.get("policy", "CnnPolicy")

    model = DQN(
        policy_name,
        env,
        learning_rate=params["lr"],
        gamma=params["gamma"],
        batch_size=params["batch"],
        exploration_initial_eps=params["eps_start"],
        exploration_final_eps=params["eps_end"],
        exploration_fraction=exploration_fraction,
        verbose=1,
        tensorboard_log="./logs/"
    )

    model.learn(total_timesteps=timesteps)
    model.save(f"dqn_model_exp{exp_num}.zip")

    eval_env = make_env(render_mode=None)
    rewards = []
    for _ in range(5):
        done = False
        obs, _ = eval_env.reset()
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)

    avg_reward = float(np.mean(rewards))
    reward_std = float(np.std(rewards))
    print(f"Experiment {exp_num} Avg Reward = {avg_reward:.2f} ± {reward_std:.2f}")

    return {
        "avg_reward": avg_reward,
        "reward_std": reward_std,
        "reward_history": rewards,
        "policy": policy_name,
        "total_timesteps": timesteps,
    }

# Load and run experiments
df = pd.read_csv('/Users/ram/Development/bowling_dqn/hyperparameters.csv')
results = []

for _, row in df.iterrows():
    params = {
        'lr': row['learning_rate'],
        'gamma': row['gamma'],
        'batch': row['batch_size'],
        'eps_start': row['exploration_initial_eps'],
        'eps_end': row['exploration_final_eps'],
        'eps_fraction': row['exploration_fraction'],
        'policy': row['policy']
    }
    result = train_dqn(int(row['exp_id']), params, timesteps=200_000)
    results.append({**row.to_dict(), **result})

results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)
print("\n✓ All experiments complete!")