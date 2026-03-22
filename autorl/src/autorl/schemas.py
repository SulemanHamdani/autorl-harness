"""Typed schemas for the AutoResearch-RL experiment scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


JsonDict = dict[str, Any]
MetricValue = float | int | bool
MetricMap = dict[str, MetricValue]
RunStatus = Literal["pending", "running", "completed", "failed", "discarded"]
GuardrailMode = Literal["max", "min", "boolean"]


@dataclass(frozen=True, slots=True)
class GuardrailSpec:
    """Constraint that can veto a run even when the primary score improves."""

    name: str
    mode: GuardrailMode
    threshold: MetricValue | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Task-level contract exposed by a task adapter."""

    task_id: str
    description: str
    backend: str
    supported_algorithms: tuple[str, ...]
    default_algorithm: str
    default_budget: JsonDict
    editable_files: tuple[str, ...]
    primary_score_name: str
    guardrails: tuple[GuardrailSpec, ...] = ()
    allowed_interventions: tuple[str, ...] = ()
    success_criteria: JsonDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Planner output for one candidate experiment."""

    task: TaskSpec
    parent_id: str | None
    seed_set: tuple[int, ...]
    intervention_config: JsonDict
    editable_files_allowlist: tuple[str, ...]
    rationale: str
    expected_risk: str
    run_budget: JsonDict
    algorithm: str


@dataclass(frozen=True, slots=True)
class RunResult:
    """Normalized outputs from one experiment execution."""

    run_id: str
    task_id: str
    status: RunStatus
    primary_score_name: str
    primary_score_value: float | None
    train_metrics: MetricMap = field(default_factory=dict)
    eval_metrics: MetricMap = field(default_factory=dict)
    summary_paths: tuple[str, ...] = ()
    wall_clock_seconds: float | None = None
    seed_outcomes: tuple[MetricMap, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Keep/discard record written after evaluating a run."""

    run_id: str
    task_id: str
    comparison_target: str | None
    keep: bool
    primary_score_name: str
    primary_score_delta: float | None
    guardrail_status: MetricMap = field(default_factory=dict)
    rationale: str = ""
