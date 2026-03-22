"""
Sanity-check script for RocketLandingEnv.

1. Runs Gymnasium's built-in env_checker (validates API compliance).
2. Runs a few episodes with random actions so you can watch the physics.
"""

from bootstrap import bootstrap_autorl_paths

bootstrap_autorl_paths()

from gymnasium.utils.env_checker import check_env
from tasks.rocket.env import RocketLandingEnv


def run_env_checker():
    print("=" * 50)
    print("Running Gymnasium env_checker ...")
    print("=" * 50)
    env = RocketLandingEnv()
    check_env(env.unwrapped)
    print("env_checker passed!\n")


def run_random_episodes(n_episodes=3):
    env = RocketLandingEnv(render_mode="human")

    for ep in range(1, n_episodes + 1):
        print("=" * 50)
        print(f"Episode {ep}")
        print("=" * 50)
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = env.action_space.sample()  # random: 0 or 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f">>> Episode {ep} finished  |  total reward = {total_reward:.2f}")
        print()

    env.close()


if __name__ == "__main__":
    run_env_checker()
    run_random_episodes()
