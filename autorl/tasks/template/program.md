# Task Name

Describe what this task is and what the agent is trying to learn.

## Goal

Maximize `your_metric_name` across a fixed evaluation suite.
Target: >= 0.X

## Files

- `train.py` — training script and evaluation. **Editable.**
- `reward.py` — reward shaping. **Editable.** (if applicable)
- `env.py` — environment definition. **Editable.** (if applicable)

## What to try

List directions for the LLM to explore. Be specific about what levers exist.

- **Reward shaping** — ...
- **Hyperparameters** — learning rate, gamma, batch size, ...
- **Curriculum learning** — ...
- **Architecture** — network size, activation functions, ...

## Constraints

- Do not modify the evaluation scenarios or fake metrics.
- Only use packages already in `pyproject.toml`.

## Rules

- One idea per experiment. Keep edits small and targeted.
- Read `results.tsv` before editing to understand what has been tried.
- Read `metrics.json` for the per-scenario breakdown.
- If the last experiment regressed, revert with `git reset --hard HEAD~1`.
