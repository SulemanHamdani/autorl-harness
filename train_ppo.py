"""
Train a PPO agent to land the rocket.

Usage:
    uv run python train_ppo.py

This will:
    1. Train for 100k timesteps (~2-3 minutes)
    2. Print reward stats every 1k steps so you can watch it improve
    3. Save the trained model to rocket_ppo.zip
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from rocket_env import RocketLandingEnv


def main():
    # --- create environments ---
    # Training env: the agent explores here
    train_env = Monitor(RocketLandingEnv())

    # Eval env: tested every 5k steps to measure real performance
    eval_env = Monitor(RocketLandingEnv())

    # --- set up evaluation callback ---
    # Every 5000 steps, run 10 test episodes and print the mean reward.
    # Saves the best model automatically.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_v2/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # --- create PPO agent ---
    model = PPO(
        "MlpPolicy",           # neural network with 2 hidden layers (64, 64)
        train_env,
        verbose=1,             # print training stats
        learning_rate=3e-4,    # how fast the network learns
        n_steps=2048,          # collect this many steps before each update
        batch_size=64,         # mini-batch size for gradient descent
        n_epochs=10,           # how many passes over the data per update
        gamma=0.99,            # discount factor (how much to value future rewards)
        device="cuda",         # use GPU for training
    )

    print("Starting training...")
    print("You should see eval reward climb from ~-100 toward +100\n")

    # --- train ---
    model.learn(total_timesteps=500_000, callback=eval_callback)

    # --- save ---
    model.save("rocket_ppo_v2")
    print("\nTraining complete! Model saved to rocket_ppo_v2.zip")


if __name__ == "__main__":
    main()
