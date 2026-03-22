"""Runner contracts for the AutoResearch-RL scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod

from autorl.schemas import ExperimentSpec, RunResult


class ExperimentRunner(ABC):
    """Executes a validated experiment spec and returns normalized outputs."""

    @abstractmethod
    def run(self, spec: ExperimentSpec) -> RunResult:
        """Execute one experiment."""
