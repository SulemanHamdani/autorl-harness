"""Task adapter registry for the AutoResearch-RL scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from autorl.schemas import JsonDict, MetricMap, TaskSpec


@dataclass(frozen=True, slots=True)
class TaskInfo:
    """Small summary object suitable for CLI output."""

    task_id: str
    module: str
    description: str
    editable_files: tuple[str, ...]
    primary_score_name: str


class TaskAdapter(ABC):
    """Contract that every task package must implement."""

    @property
    @abstractmethod
    def module_name(self) -> str:
        """Import path for the task package."""

    @abstractmethod
    def get_task_spec(self) -> TaskSpec:
        """Return the normalized task specification."""

    @abstractmethod
    def build_train_env(self, **kwargs: Any):
        """Create the training environment instance."""

    @abstractmethod
    def build_eval_env(self, **kwargs: Any):
        """Create the evaluation environment instance."""

    @abstractmethod
    def get_default_experiment_config(self) -> JsonDict:
        """Return the task's default experiment config."""

    @abstractmethod
    def get_allowed_interventions(self) -> tuple[str, ...]:
        """Return allowed intervention categories for the planner."""

    @abstractmethod
    def get_primary_score_definition(self) -> tuple[str, tuple[Any, ...]]:
        """Return the task's primary score and guardrail definitions."""

    @abstractmethod
    def compute_summary_metrics(self, raw_metrics: MetricMap) -> MetricMap:
        """Compute task-specific summary metrics from normalized raw metrics."""

    @abstractmethod
    def get_success_criteria(self) -> JsonDict:
        """Return task-specific success criteria for later evaluation."""

    def get_task_info(self) -> TaskInfo:
        """Return a CLI-friendly task summary."""

        spec = self.get_task_spec()
        return TaskInfo(
            task_id=spec.task_id,
            module=self.module_name,
            description=spec.description,
            editable_files=spec.editable_files,
            primary_score_name=spec.primary_score_name,
        )


_TASKS: dict[str, TaskAdapter] = {}
_BUILTINS_LOADED = False


def _load_builtin_tasks() -> None:
    """Import built-in task packages once so they can self-register."""

    global _BUILTINS_LOADED

    if _BUILTINS_LOADED:
        return

    import_module("tasks.rocket")
    _BUILTINS_LOADED = True


def register_task(adapter: TaskAdapter) -> None:
    """Register a task adapter by task id."""

    task_id = adapter.get_task_spec().task_id
    _TASKS[task_id] = adapter


def get_task(task_id: str) -> TaskAdapter:
    """Return the registered task adapter or raise a helpful error."""

    _load_builtin_tasks()

    try:
        return _TASKS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(_TASKS)) or "<none>"
        raise KeyError(f"Unknown task '{task_id}'. Registered tasks: {available}") from exc


def list_tasks() -> list[TaskInfo]:
    """Return all known task packages."""

    _load_builtin_tasks()
    return [adapter.get_task_info() for _, adapter in sorted(_TASKS.items())]
