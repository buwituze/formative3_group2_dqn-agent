import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import sys

gym.register_envs(ale_py)

def play_game(model_path, num_episodes=10):
    env = AtariWrapper(gym.make('ALE/Bowling-v5', render_mode='human'))
    
    print("Loading model...")
    model = DQN.load(model_path, custom_objects={"buffer_size": 1, "learning_starts": 0})
    print("Model loaded successfully!\n")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}:")
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"  Steps: {steps}, Reward: {total_reward:.1f}", end='\r')
        
        print(f"  Final - Reward: {total_reward:.1f}, Steps: {steps}     ")
    
    env.close()

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'dqn_model_exp1.zip'
    print(f"Loading model from {model_path}")
    play_game(model_path)
