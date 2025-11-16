import gymnasium as gym
from stable_baselines3 import DQN
import time
import ale_py

gym.register_envs(ale_py)

# change this path if you need to run a different model
model_path = "benitha_dqn_model.zip"
print(f"Loading model from: {model_path}")

env = gym.make("ALE/Bowling-v5", render_mode="human")

model = DQN.load(model_path, env=env)
print("Model loaded successfully!")

num_episodes = 5

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    step_count = 0
    
    print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        time.sleep(0.02)
    
    print(f"Episode {episode + 1} finished!")
    print(f"  Steps: {step_count}")
    print(f"  Total Reward: {episode_reward}")

env.close()
print("\nDemo complete!")