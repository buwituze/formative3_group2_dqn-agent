"""
Play script for DQN Bowling Agent
Uses the trained model from Experiment 1 to play the Bowling Atari game.
"""

import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import argparse
import os

# Import ale_py to register ALE namespace
import ale_py

def make_env(env_id="ALE/Bowling-v5", render_mode=None):
    """Create and wrap the environment"""
    env = gym.make(env_id, render_mode=render_mode)
    return env

def play(model_path, num_episodes=5, render=True):
    """
    Play using a trained DQN model
    
    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to play
        render: Whether to render the game (show visual output)
    """
    print("="*80)
    print("DQN Bowling Agent - Play Mode")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Rendering: {render}")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at: {model_path}")
        print(f"\nAvailable models in models/exp_1_CnnPolicy/:")
        model_dir = "models/exp_1_CnnPolicy"
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    print(f"  - {os.path.join(model_dir, file)}")
        return
    
    # Create environment
    print("\nLoading environment...")
    env = make_env(render_mode="human" if render else None)
    print(f"Environment loaded: {env}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        model = DQN.load(model_path, env=env)
        print("[OK] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        env.close()
        return
    
    # Set to deterministic policy (no exploration)
    model.exploration_rate = 0.0
    print("[OK] Set to deterministic policy (no exploration)")
    
    # Play episodes
    print(f"\n{'='*80}")
    print("Starting gameplay...")
    print(f"{'='*80}\n")
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        
        print(f"Episode {episode + 1}/{num_episodes} - ", end="", flush=True)
        
        while not (done or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Render if enabled
            if render:
                env.render()
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        print(f"Reward: {episode_reward:.2f}, Steps: {episode_steps}")
    
    env.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("GAMEPLAY SUMMARY")
    print(f"{'='*80}")
    print(f"Episodes played: {num_episodes}")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Best reward: {np.max(total_rewards):.2f}")
    print(f"Worst reward: {np.min(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"\nIndividual episode rewards:")
    for i, reward in enumerate(total_rewards, 1):
        print(f"  Episode {i}: {reward:.2f}")
    print(f"{'='*80}\n")
    
    return total_rewards

def main():
    parser = argparse.ArgumentParser(description="Play Bowling using trained DQN model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/exp_1_CnnPolicy/best_model.zip",
        help="Path to the trained model (default: models/exp_1_CnnPolicy/best_model.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (faster, no visual output)"
    )
    
    args = parser.parse_args()
    
    # Play the game
    play(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )

if __name__ == "__main__":
    main()

