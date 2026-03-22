"""Planner contract for the AutoResearch-RL scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod

from autorl.schemas import ExperimentSpec, JsonDict, TaskSpec


class ExperimentPlanner(ABC):
    """Produces validated experiment specs from program intent and history."""

    @abstractmethod
    def plan(
        self,
        *,
        task: TaskSpec,
        program_text: str,
        experiment_history: tuple[JsonDict, ...],
        latest_summary: JsonDict | None,
        allowed_interventions: tuple[str, ...],
    ) -> ExperimentSpec:
        """Return the next experiment spec for a task."""
