"""Metrics aggregation module.

:class:`EvaluationMetrics` collects the :class:`~agentic_eval.evaluators.base.EvaluationResult`
objects produced by multiple evaluators and computes aggregate statistics.
"""

from __future__ import annotations

import statistics
from typing import Any

from agentic_eval.evaluators.base import EvaluationResult


class EvaluationMetrics:
    """Aggregate and summarise a collection of :class:`EvaluationResult` objects."""

    def __init__(self, results: list[EvaluationResult]) -> None:
        self.results = results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def overall_score(self) -> float:
        """Mean normalised score across all evaluators."""
        if not self.results:
            return 0.0
        return statistics.mean(r.normalised_score for r in self.results)

    @property
    def scores_by_evaluator(self) -> dict[str, float]:
        """Mapping of evaluator name → normalised score."""
        return {r.evaluator: r.normalised_score for r in self.results}

    @property
    def passed(self) -> bool:
        """Return True if the overall score is ≥ 0.5."""
        return self.overall_score >= 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a structured summary dictionary."""
        return {
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "evaluators": {
                r.evaluator: {
                    "score": round(r.normalised_score, 4),
                    "errors": r.errors,
                }
                for r in self.results
            },
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EvaluationMetrics(overall={self.overall_score:.4f}, "
            f"passed={self.passed})"
        )
