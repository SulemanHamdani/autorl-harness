#!/usr/bin/env python
"""Rocket landing training script.

This is the editable training entry point. The outer loop runs this script
and reads metrics.json from the task directory. Everything about training,
evaluation, and metrics lives here.

The LLM agent edits this file (and reward.py / env.py) to improve
soft_landing_rate across a fixed evaluation suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env import RocketLandingEnv

# ---- configuration ----

TOTAL_TIMESTEPS = 1_250_000
SEED = 0

HYPERPARAMETERS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
}

TASK_DIR = Path(__file__).resolve().parent

# ---- fixed evaluation scenarios ----
# Each scenario tests a specific regime. The agent must generalize across all.

EVAL_SCENARIOS = {
    # easy: low altitude, gentle descent, plenty of fuel
    "easy_low":       {"altitude": 50.0,  "velocity": -2.0,  "fuel": 20.0},
    "easy_mid":       {"altitude": 150.0, "velocity": -5.0,  "fuel": 20.0},

    # medium: moderate altitude and velocity
    "med_descent":    {"altitude": 250.0, "velocity": -10.0, "fuel": 20.0},
    "med_fast":       {"altitude": 200.0, "velocity": -20.0, "fuel": 20.0},
    "med_rising":     {"altitude": 100.0, "velocity": 5.0,   "fuel": 15.0},

    # hard: high altitude, fast descent
    "hard_high":      {"altitude": 450.0, "velocity": -15.0, "fuel": 25.0},
    "hard_fast":      {"altitude": 300.0, "velocity": -30.0, "fuel": 25.0},
    "hard_very_fast": {"altitude": 200.0, "velocity": -30.0, "fuel": 20.0},

    # low fuel: must be fuel-efficient
    "lowfuel_low":    {"altitude": 80.0,  "velocity": -5.0,  "fuel": 5.0},
    "lowfuel_mid":    {"altitude": 200.0, "velocity": -10.0, "fuel": 8.0},
    "lowfuel_high":   {"altitude": 350.0, "velocity": -10.0, "fuel": 10.0},

    # edge cases
    "edge_barely":    {"altitude": 15.0,  "velocity": -4.0,  "fuel": 5.0},
    "edge_high_v":    {"altitude": 100.0, "velocity": -25.0, "fuel": 15.0},
    "edge_rising_hi": {"altitude": 300.0, "velocity": 10.0,  "fuel": 20.0},
    "edge_max_alt":   {"altitude": 490.0, "velocity": -5.0,  "fuel": 30.0},
}


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
    """Run evaluation on the fixed scenario suite."""

    scenario_results = {}
    all_successes = []
    all_crashes = []
    all_timeouts = []
    all_oob = []
    all_td_vels = []

    for name, scenario in EVAL_SCENARIOS.items():
        env = RocketLandingEnv(scenario=scenario)
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

        # Evaluation metrics should use physical state, not normalized observations.
        altitude = env.z
        velocity = env.v
        fuel = env.fuel
        touched_down = terminated and altitude <= 0.0
        success = touched_down and abs(velocity) < env.safe_velocity
        crash = touched_down and not success

        scenario_results[name] = {
            "success": success,
            "crash": crash,
            "timeout": bool(truncated),
            "out_of_bounds": terminated and altitude > 0.0,
            "touchdown_velocity": round(abs(velocity), 2) if touched_down else None,
            "remaining_fuel": round(float(fuel), 2),
            "return": round(total_reward, 2),
            "steps": steps,
        }

        all_successes.append(success)
        all_crashes.append(crash)
        all_timeouts.append(bool(truncated))
        all_oob.append(terminated and altitude > 0.0)
        if touched_down:
            all_td_vels.append(abs(velocity))

        env.close()

    return {
        # primary score
        "soft_landing_rate": float(np.mean(all_successes)),
        # aggregate metrics
        "crash_rate": float(np.mean(all_crashes)),
        "timeout_rate": float(np.mean(all_timeouts)),
        "out_of_bounds_rate": float(np.mean(all_oob)),
        "mean_touchdown_velocity": round(float(np.mean(all_td_vels)), 2) if all_td_vels else 0.0,
        # per-scenario breakdown (so the LLM can see where it fails)
        "scenarios": scenario_results,
    }


def main() -> None:
    start = perf_counter()
    metrics = train()
    wall_clock = perf_counter() - start
    metrics["wall_clock_seconds"] = round(wall_clock, 1)

    # Write structured metrics for the loop to parse
    metrics_path = TASK_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Print summary to stdout
    print("---")
    print(f"soft_landing_rate:       {metrics['soft_landing_rate']:.3f}")
    print(f"crash_rate:              {metrics['crash_rate']:.3f}")
    print(f"timeout_rate:            {metrics['timeout_rate']:.3f}")
    print(f"out_of_bounds_rate:      {metrics['out_of_bounds_rate']:.3f}")
    print(f"mean_touchdown_velocity: {metrics['mean_touchdown_velocity']:.2f}")
    print(f"wall_clock_seconds:      {metrics['wall_clock_seconds']}")
    print()
    print("Per-scenario results:")
    for name, r in metrics["scenarios"].items():
        status = "LAND" if r["success"] else ("CRASH" if r["crash"] else ("OOB" if r["out_of_bounds"] else "TIMEOUT"))
        vel_str = f"v={r['touchdown_velocity']:.1f}" if r["touchdown_velocity"] is not None else "v=N/A"
        print(f"  {name:20s}  {status:7s}  {vel_str}  fuel={r['remaining_fuel']:.1f}  steps={r['steps']}")


if __name__ == "__main__":
    main()
