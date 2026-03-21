"""
Watch the trained PPO agent land the rocket.

Usage:
    uv run python enjoy.py

Loads rocket_ppo.zip and runs 5 episodes with render_mode="human"
so you can see each step printed to the console.
"""

from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv


def main():
    env = RocketLandingEnv(render_mode="human")
    model = PPO.load("rocket_ppo")

    for ep in range(1, 6):
        print("=" * 50)
        print(f"Episode {ep}")
        print("=" * 50)
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f">>> Episode {ep} finished  |  total reward = {total_reward:.2f}\n")

    env.close()


if __name__ == "__main__":
    main()
