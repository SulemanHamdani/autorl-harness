"""Evaluator contracts for the AutoResearch-RL scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod

from autorl.schemas import DecisionRecord, RunResult


class ExperimentEvaluator(ABC):
    """Compares a candidate run against the current kept baseline."""

    @abstractmethod
    def evaluate(self, candidate: RunResult, baseline: RunResult | None) -> DecisionRecord:
        """Return the keep/discard decision for a candidate run."""
