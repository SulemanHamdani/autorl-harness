#!/usr/bin/env python
"""Rocket landing training script.

This is the editable training entry point. The outer loop runs this script
and reads metrics.json from the task directory. Everything about training,
evaluation, and metrics lives here.

The LLM agent edits this file (and reward.py) to improve soft_landing_rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env import RocketLandingEnv

# ---- configuration ----

TOTAL_TIMESTEPS = 500_000
N_EVAL_EPISODES = 10
SEED = 0

HYPERPARAMETERS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
}

TASK_DIR = Path(__file__).resolve().parent


def train() -> dict:
    """Train a PPO agent and return evaluation metrics."""

    set_random_seed(SEED)
    env = Monitor(RocketLandingEnv())
    env.reset(seed=SEED)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        seed=SEED,
        learning_rate=float(HYPERPARAMETERS["learning_rate"]),
        n_steps=int(HYPERPARAMETERS["n_steps"]),
        batch_size=int(HYPERPARAMETERS["batch_size"]),
        n_epochs=int(HYPERPARAMETERS["n_epochs"]),
        gamma=float(HYPERPARAMETERS["gamma"]),
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(str(TASK_DIR / "model.zip"))
    env.close()

    return evaluate(model)


def evaluate(model) -> dict:
    """Run deterministic evaluation episodes and compute metrics."""

    results = []
    for i in range(N_EVAL_EPISODES):
        env = RocketLandingEnv()
        obs, _ = env.reset(seed=SEED * 10_000 + i)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

        altitude, velocity, fuel, _ = obs.tolist()
        touched_down = terminated and altitude <= 0.0
        success = touched_down and abs(velocity) < env.safe_velocity
        crash = touched_down and not success

        results.append({
            "return": total_reward,
            "length": steps,
            "success": success,
            "crash": crash,
            "timeout": bool(truncated),
            "out_of_bounds": terminated and altitude > 0.0,
            "touchdown_velocity": abs(velocity) if touched_down else None,
            "remaining_fuel": float(fuel),
        })
        env.close()

    successes = [r["success"] for r in results]
    crashes = [r["crash"] for r in results]
    timeouts = [r["timeout"] for r in results]
    oob = [r["out_of_bounds"] for r in results]
    td_vels = [r["touchdown_velocity"] for r in results if r["touchdown_velocity"] is not None]

    return {
        "soft_landing_rate": float(np.mean(successes)),
        "crash_rate": float(np.mean(crashes)),
        "timeout_rate": float(np.mean(timeouts)),
        "out_of_bounds_rate": float(np.mean(oob)),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_episode_length": float(np.mean([r["length"] for r in results])),
        "mean_touchdown_velocity": float(np.mean(td_vels)) if td_vels else 0.0,
        "mean_remaining_fuel": float(np.mean([r["remaining_fuel"] for r in results])),
    }


def main() -> None:
    start = perf_counter()
    metrics = train()
    wall_clock = perf_counter() - start
    metrics["wall_clock_seconds"] = round(wall_clock, 1)

    # Write structured metrics for the loop to parse
    metrics_path = TASK_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Print Karpathy-style summary to stdout
    print("---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
