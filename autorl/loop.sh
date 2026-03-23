#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# autorl loop — Karpathy-style autonomous experiment loop
#
# Usage: ./autorl/loop.sh <task> <iterations> [claude|codex]
#
# Example: ./autorl/loop.sh rocket 50 claude
# ============================================================================

TASK="${1:?Usage: loop.sh <task> <iterations> [claude|codex]}"
ITERATIONS="${2:?Usage: loop.sh <task> <iterations> [claude|codex]}"
PROVIDER="${3:-claude}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_DIR="$SCRIPT_DIR/tasks/$TASK"

# ---- validate ----

if [ ! -f "$TASK_DIR/spec.yaml" ]; then
    echo "Error: No spec.yaml found at $TASK_DIR/spec.yaml"
    exit 1
fi

# ---- parse spec.yaml with uv (pyyaml is in the venv) ----

read_spec() {
    uv run python3 -c "
import yaml
spec = yaml.safe_load(open('$TASK_DIR/spec.yaml'))
print(spec['$1'])
"
}

RUN_COMMAND=$(read_spec run_command)
PRIMARY_SCORE=$(read_spec primary_score)

EDITABLE_FILES=$(uv run python3 -c "
import yaml
spec = yaml.safe_load(open('$TASK_DIR/spec.yaml'))
for f in spec['editable_files']:
    print(f)
")

echo "=== AutoRL Loop ==="
echo "Task:       $TASK"
echo "Iterations: $ITERATIONS"
echo "Provider:   $PROVIDER"
echo "Score:      $PRIMARY_SCORE"
echo "Run cmd:    $RUN_COMMAND"
echo "==================="

# ---- initialize results.tsv ----

RESULTS_FILE="$TASK_DIR/results.tsv"
if [ ! -f "$RESULTS_FILE" ]; then
    printf "commit\t%s\tcrash_rate\tstatus\tdescription\n" "$PRIMARY_SCORE" > "$RESULTS_FILE"
fi

cd "$REPO_ROOT"

# ---- helper: run experiment and log results ----

run_and_log() {
    local status_override="${1:-}"
    local desc_override="${2:-}"

    echo "  Running: $RUN_COMMAND"
    # Run in a subshell so `cd` in RUN_COMMAND doesn't change our cwd
    if (cd "$REPO_ROOT" && eval "$RUN_COMMAND") > "$TASK_DIR/run.log" 2>&1; then
        local score crash_rate
        score=$(uv run python3 -c "import json; print(json.load(open('$TASK_DIR/metrics.json'))['$PRIMARY_SCORE'])")
        crash_rate=$(uv run python3 -c "import json; print(json.load(open('$TASK_DIR/metrics.json')).get('crash_rate', 0.0))")
        local commit desc status
        commit=$(git rev-parse --short HEAD)
        desc="${desc_override:-$(git log -1 --format=%s)}"
        status="${status_override:-keep}"
        printf "%s\t%s\t%s\t%s\t%s\n" "$commit" "$score" "$crash_rate" "$status" "$desc" >> "$RESULTS_FILE"
        echo "  Score: $PRIMARY_SCORE=$score  crash_rate=$crash_rate  [$status]"
    else
        local commit desc
        commit=$(git rev-parse --short HEAD)
        desc="${desc_override:-$(git log -1 --format=%s)}"
        printf "%s\t0.0\t0.0\tcrash\t%s\n" "$commit" "$desc" >> "$RESULTS_FILE"
        echo "  CRASHED — see $TASK_DIR/run.log"
    fi
}

# ---- helper: build prompt (done in Python to avoid bash escaping hell) ----

build_prompt() {
    uv run python3 -c "
import yaml
from pathlib import Path

task_dir = Path('$TASK_DIR')
repo_root = Path('$REPO_ROOT')
template = Path('$SCRIPT_DIR/prompt.md').read_text()

spec_yaml = task_dir / 'spec.yaml'
program_md = task_dir / 'program.md'
results_file = task_dir / 'results.tsv'

spec = yaml.safe_load(spec_yaml.read_text())

# build editable files section
files_section = ''
for fpath in spec['editable_files']:
    full = repo_root / fpath
    if full.exists():
        files_section += f'### {fpath}\n\`\`\`python\n{full.read_text()}\`\`\`\n\n'

replacements = {
    '{{PROGRAM_MD}}': program_md.read_text() if program_md.exists() else '',
    '{{SPEC_YAML}}': spec_yaml.read_text(),
    '{{RESULTS_TSV}}': results_file.read_text() if results_file.exists() else 'No experiments yet.',
    '{{EDITABLE_FILES}}': files_section,
    '{{PRIMARY_SCORE}}': spec['primary_score'],
    '{{TASK_DIR}}': str(task_dir),
}

prompt = template
for key, value in replacements.items():
    prompt = prompt.replace(key, value)

print(prompt)
"
}

# ---- helper: call LLM ----

call_llm() {
    local prompt_file
    prompt_file=$(mktemp)
    build_prompt > "$prompt_file"

    if [ "$PROVIDER" = "claude" ]; then
        claude -p --allowedTools "Edit,Read,Bash(git:*)" < "$prompt_file"
    elif [ "$PROVIDER" = "codex" ]; then
        codex exec "$(cat "$prompt_file")"
    else
        echo "Error: Unknown provider '$PROVIDER'. Use 'claude' or 'codex'."
        rm -f "$prompt_file"
        exit 1
    fi

    rm -f "$prompt_file"
}

# ---- iteration 0: baseline ----

echo ""
echo "[0/$ITERATIONS] Baseline run (no edits)"
run_and_log "keep" "baseline"

# ---- main loop ----

for i in $(seq 1 "$ITERATIONS"); do
    echo ""
    echo "[$i/$ITERATIONS] Calling $PROVIDER for experiment..."

    call_llm

    # The LLM edits files but may not be able to commit (sandbox restrictions).
    # We handle the commit here if there are uncommitted changes to editable files.
    if ! git diff --quiet -- $EDITABLE_FILES 2>/dev/null; then
        local desc
        desc=$(git log -1 --format=%s 2>/dev/null || echo "experiment $i")
        git add $EDITABLE_FILES
        git commit -m "experiment $i" --allow-empty-message 2>/dev/null || true
    fi

    echo "  Running experiment..."
    run_and_log
done

echo ""
echo "=== Done: $ITERATIONS iterations ==="
echo "Results: $RESULTS_FILE"
cat "$RESULTS_FILE"
