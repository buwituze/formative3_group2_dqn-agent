"""Play an Atari Bowling game with a trained DQN agent using a greedy policy.

Usage example:
    python play.py --model-path dqn_model_exp4.zip --episodes 3

Ensure Atari ROMs are installed (e.g., run `AutoROM --accept-license`).
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
from gymnasium import error as gym_error
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Import registers the ALE namespace so Gymnasium can find the Atari env.
import ale_py  # noqa: F401  pylint: disable=unused-import


class GreedyQPolicy:
    """Select the action with the highest predicted value via deterministic DQN."""

    def select_action(self, model: DQN, observation) -> int:
        action, _ = model.predict(observation, deterministic=True)
        return int(action)


def make_env(render_mode: str | None) -> tuple[gym.Env, gym.Env]:
    """Create the wrapped Atari Bowling env plus the underlying env for rendering."""
    try:
        base_env = gym.make("ALE/Bowling-v5", render_mode=render_mode)
    except gym_error.NamespaceNotFound as err:
        raise RuntimeError(
            "Gym can't find the ALE namespace. Install Atari support via "
            "`pip install \"gymnasium[atari]\" autorom[accept-rom-license]` "
            "and run `AutoROM --accept-license`."
        ) from err
    wrapped_env = AtariWrapper(base_env)
    return wrapped_env, base_env


def play_episode(model: DQN, policy: GreedyQPolicy, *, render: bool = True, frame_delay: float = 0.02) -> float:
    """Run a single greedy episode and return the cumulative reward."""
    env, base_env = make_env(render_mode="human" if render else None)

    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = policy.select_action(model, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if render:
            base_env.render()
            if frame_delay > 0.0:
                time.sleep(frame_delay)

    env.close()
    return float(total_reward)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained DQN Bowling agent with a greedy policy.")
    parser.add_argument(
        "--model-path",
        default="dqn_model_exp4.zip",
        help="Path to the saved Stable-Baselines3 DQN model (zip file).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable GUI rendering (useful for headless evaluation).",
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.02,
        help="Delay in seconds between rendered frames for smoother playback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading DQN model from {args.model_path} ...")
    # Shrink replay buffer footprint during load; only the policy is required for play.
    custom_overrides = {"buffer_size": 1_000}
    model = DQN.load(args.model_path, custom_objects=custom_overrides, device="cpu")

    policy = GreedyQPolicy()
    render = not args.no_render

    rewards = []
    for episode in range(1, args.episodes + 1):
        reward = play_episode(model, policy, render=render, frame_delay=args.frame_delay)
        rewards.append(reward)
        print(f"Episode {episode}: reward = {reward:.2f}")

    if rewards:
        average_reward = sum(rewards) / len(rewards)
        print(f"Average reward over {len(rewards)} episode(s): {average_reward:.2f}")


if __name__ == "__main__":
    main()
