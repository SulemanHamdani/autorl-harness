# Rocket Landing — Generalization Experiment

You are an autonomous RL researcher. Your goal is to build a PPO agent that can **land a 1D rocket softly across a wide range of initial conditions**.

## The challenge

The environment randomizes initial altitude (10–490m), velocity (-30 to +10 m/s), and fuel (5–30 kg) during training. The agent must learn a general landing policy, not memorize one trajectory.

Evaluation uses 15 **fixed scenarios** spanning easy, medium, hard, low-fuel, and edge cases. The primary score is `soft_landing_rate` — the fraction of these 15 scenarios where the rocket lands with |velocity| < 5.0 m/s. **Target: ≥ 0.90.**

## Files

- `env.py` — the Gymnasium environment. Defines physics, observation/action spaces, initial condition randomization. **Editable.**
- `train.py` — training script + fixed evaluation suite. **Editable.**
- `reward.py` — reward shaping and terminal rewards. **Editable.**
- `spec.yaml` — task metadata. **Read-only.**
- `results.tsv` — experiment history. **Read-only** (the loop appends to it).

## What to try

These are directions, not an ordered checklist. Use your judgment based on results.

- **Reward shaping** — the current reward is basic. Consider velocity-dependent shaping, fuel-efficiency bonuses, altitude-aware penalties. This is often the highest-leverage edit.
- **Hyperparameters** — learning rate, gamma, GAE lambda, network size, n_steps, batch_size.
- **Observation normalization** — raw values span very different scales (altitude 0–500, velocity -30 to +10, fuel 5–30). Normalizing or using VecNormalize may help.
- **Curriculum learning** — start with easy scenarios (low altitude, slow descent), widen over training.
- **Training budget** — you can increase TOTAL_TIMESTEPS if 500K isn't enough. Be aware this increases wall clock time.

## Constraints

- Only use packages in `pyproject.toml`.
- Do NOT modify the evaluation scenarios or fake metrics.
- Do NOT modify `spec.yaml`.

## Rules

- **One idea per experiment.** Keep edits small and targeted.
- **Read results.tsv** before editing. Understand what worked and what didn't.
- **Read `metrics.json`** for the per-scenario breakdown — it tells you exactly which scenarios fail.
- **If the last experiment regressed**, revert with `git reset --hard HEAD~1`.
- **If it improved**, build on it.
- **Commit your changes** with a short message before the loop runs training.
