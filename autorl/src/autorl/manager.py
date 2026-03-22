"""Core coordination contracts for the AutoResearch-RL scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from autorl.evaluator import ExperimentEvaluator
from autorl.planner import ExperimentPlanner
from autorl.registry import TaskAdapter, get_task
from autorl.runner import ExperimentRunner
from autorl.schemas import ExperimentSpec, TaskSpec


@dataclass(slots=True)
class ExperimentManager:
    """Small orchestration shell around the main scaffold interfaces."""

    planner: ExperimentPlanner
    runner: ExperimentRunner
    evaluator: ExperimentEvaluator

    def get_task_adapter(self, task_id: str) -> TaskAdapter:
        """Return the adapter for a registered task."""

        return get_task(task_id)

    def get_task_spec(self, task_id: str) -> TaskSpec:
        """Return the normalized spec for a registered task."""

        return self.get_task_adapter(task_id).get_task_spec()

    def validate_experiment_spec(self, spec: ExperimentSpec) -> None:
        """Ensure the spec only references editable files allowed by the task."""

        task_spec = self.get_task_spec(spec.task.task_id)
        disallowed = set(spec.editable_files_allowlist) - set(task_spec.editable_files)
        if disallowed:
            files = ", ".join(sorted(disallowed))
            raise ValueError(f"Experiment spec references disallowed editable files: {files}")
