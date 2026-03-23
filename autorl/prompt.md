# AutoRL Experiment Iteration

You are an autonomous RL researcher running experiments in a loop.
Read the context below, then edit the editable files to improve the primary score.

## Task Program

{{PROGRAM_MD}}

## Task Spec

{{SPEC_YAML}}

## Experiment History

{{RESULTS_TSV}}

## Current Editable Files

{{EDITABLE_FILES}}

## Previous Run Log

The full output from the last experiment is at `{{TASK_DIR}}/run.log`. Read it if you need to understand training progress, crashes, or errors.

The per-scenario evaluation breakdown is at `{{TASK_DIR}}/metrics.json`. Read it to see exactly which scenarios pass and fail.

## Instructions

1. Read the experiment history above. Understand what has been tried and what worked.
2. Read `{{TASK_DIR}}/metrics.json` for per-scenario results from the last run.
3. If needed, read `{{TASK_DIR}}/run.log` for training details.
4. If the last experiment made things worse, run: `git reset --hard HEAD~1`
5. Edit the editable files to try something new that should improve `{{PRIMARY_SCORE}}`.
6. Keep edits small and targeted — one idea per experiment.
7. Do NOT commit — the loop handles git commits automatically after you exit.
8. Do NOT run the training script — the loop handles that after you exit.
9. Do NOT modify any files outside the editable files list.
