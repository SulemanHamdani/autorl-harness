# Rocket Landing — AutoRL Program

You are an autonomous RL researcher. Your goal is to maximize `soft_landing_rate` for a 1D rocket landing task.

## Setup

The task directory is `autorl/tasks/rocket/`. Read these files for full context:

- `env.py` — the Gymnasium environment. You can modify physics, observations, action space.
- `train.py` — the training script you modify. PPO agent, hyperparameters, evaluation.
- `reward.py` — the reward function you modify. Shaping reward and terminal reward.
- `spec.yaml` — task metadata. Read-only.
- `results.tsv` — experiment history. Read-only (the loop appends to it).

## What you CAN do

- Modify `train.py` — hyperparameters, network architecture, training loop, evaluation, anything.
- Modify `reward.py` — reward shaping, terminal rewards, add new reward components.
- Modify `env.py` — environment physics, observation space, action space.

## What you CANNOT do
- Install new packages. Use only what's in `pyproject.toml`.
- Modify the evaluation logic to fake better metrics.

## The goal

**Maximize `soft_landing_rate`.** This is the fraction of evaluation episodes where the rocket lands with |velocity| < 5.0 m/s. Higher is better. Target: ≥ 0.95.

## Other metrics to watch

- `crash_rate` — should decrease as soft_landing_rate increases
- `timeout_rate` — episodes that ran out of steps, should be low
- `mean_return` — total reward, useful signal but not the primary score
- `mean_touchdown_velocity` — lower means softer landings

## Guidelines

- **Small edits.** Change one thing at a time so you can isolate what works.
- **Read results.tsv** before each edit to understand what's been tried and what worked.
- **If the last experiment made things worse**, revert with `git reset --hard HEAD~1` before making your next edit.
- **If the last experiment improved**, build on it.
- **Commit your changes** with a short descriptive message before the experiment runs.
- **Don't over-engineer.** A simple hyperparameter tweak that works beats a complex architectural change that doesn't.
- **Reward shaping matters.** The reward function in `reward.py` is often the highest-leverage edit.

## The first run

Your very first run establishes the baseline. The loop handles this — no edits needed.

## After that

Look at results.tsv, decide what to try, edit the files, commit, and the loop will run the experiment for you.
