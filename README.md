# AutoRL Harness

A minimal autonomous RL experiment loop. Each iteration, an LLM agent reads your task description, experiment history, and current code, edits files to improve a target metric, and exits. The loop handles training, git commits, and logging. Repeat for N iterations.

Inspired by Karpathy's auto-research work. The main difference is that each LLM call is a fresh session rather than one long conversation. This matters for RL because training runs can take 5-20+ minutes per iteration — long sessions eventually timeout or lose context. Here, the full context (file contents, results history, last run log) is injected fresh each call.

The harness is task-agnostic. It reads a `spec.yaml` to know what command to run, which files the LLM can edit, and which metric to track. The LLM never runs training itself — it only edits code.

---

## How it works

```
for each iteration:
  1. build a prompt from: task description + experiment history + current file contents
  2. call the LLM CLI (claude or codex) with that prompt via stdin
  3. commit any file changes the LLM made
  4. run the training script
  5. read metrics.json, append a row to results.tsv
```

The LLM sees the full `results.tsv` history every iteration, so it can reason about what has been tried and what direction to go next. It can also revert a bad experiment with `git reset --hard HEAD~1` before making new changes.

---

## Quick start

```bash
git clone <this repo>
cd autorl-harness
uv sync

# run the included rocket landing example for 20 iterations
./autorl/loop.sh rocket 20 claude
```

Requires [uv](https://github.com/astral-sh/uv) and either the [Claude Code CLI](https://github.com/anthropics/claude-code) or [OpenAI Codex CLI](https://github.com/openai/codex).

---

## Adding your own task

Create a folder under `autorl/tasks/` with three things:

```
autorl/tasks/mytask/
  spec.yaml      # tells the loop how to run your task
  program.md     # tells the LLM what the goal is
  train.py       # your training/evaluation script
```

A skeleton is in `autorl/tasks/template/`.

### spec.yaml

```yaml
task_id: mytask
description: one line description
run_command: "uv run python autorl/tasks/mytask/train.py"
editable_files:
  - autorl/tasks/mytask/train.py
  - autorl/tasks/mytask/reward.py   # add as many as you want
primary_score: your_metric_name
```

`run_command` can be any shell command. The editable files can live anywhere in the repo — just list their paths relative to the repo root.

### The metrics.json contract

Your training script must write this file to `autorl/tasks/mytask/metrics.json` when it finishes:

```json
{
  "your_metric_name": 0.85,
  "other_fields": "..."
}
```

The harness only reads `primary_score`. Everything else in the file gets passed to the LLM as context via `metrics.json`, so include whatever breakdown is useful for debugging (per-scenario results, loss curves, etc.).

### program.md

A plain text description of the task written for the LLM. Include:

- what the goal is and what metric to optimize
- which files are editable and what each one does
- what directions to explore (reward shaping, hyperparameters, curriculum, etc.)
- hard constraints (do not modify evaluation code, do not fake metrics)

The rocket task's `program.md` is a good reference.

---

## Results format

Each run appends a row to `results.tsv`:

```
commit   soft_landing_rate   status   description
a1b2c3   0.33                keep     baseline
b2c3d4   0.93                keep     experiment 3
c3d4e5   1.00                keep     experiment 6
```

The LLM reads this every iteration. It can see the trajectory of experiments and decide whether to keep building on the last change or revert it.

---

## Example: rocket landing

The included `tasks/rocket/` task trains a PPO agent (via stable-baselines3) to land a 1D rocket softly across 15 evaluation scenarios. Initial conditions vary across altitude (15-490m), velocity (up to -30 m/s), and fuel (as low as 5kg). The agent must generalize across all of them — not just the easy cases.

Starting from a 33% soft landing rate, the LLM reached 100% across all 15 scenarios in 20 experiments. The changes it made across those experiments: observation normalization, curriculum learning with a gradually widening training distribution, a low-fuel training bias so the agent sees constrained scenarios more often, and altitude-aware reward shaping with a flare window penalty near touchdown. None of that was in the initial code.

---

## Requirements

- bash, git
- [uv](https://github.com/astral-sh/uv)
- [Claude Code CLI](https://github.com/anthropics/claude-code) (`claude`) or [OpenAI Codex CLI](https://github.com/openai/codex) (`codex`)
- Python 3.11+

Task-specific Python dependencies go in `pyproject.toml`. The rocket task uses `gymnasium` and `stable-baselines3`.
