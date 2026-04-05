"""EvaluationSuite – the top-level orchestrator.

Usage example::

    from agentic_eval import EvaluationSuite

    suite = EvaluationSuite()
    report = suite.evaluate(sample)
    report.print_summary()
    print(report.to_json())
"""

from __future__ import annotations

from typing import Any

from agentic_eval.evaluators.reasoning import ReasoningEvaluator
from agentic_eval.evaluators.tool_usage import ToolUsageEvaluator
from agentic_eval.evaluators.workflow import WorkflowEvaluator
from agentic_eval.evaluators.task_success import TaskSuccessEvaluator
from agentic_eval.evaluators.base import BaseEvaluator, EvaluationResult
from agentic_eval.metrics.metrics import EvaluationMetrics
from agentic_eval.reporters.report import EvaluationReport


class EvaluationSuite:
    """Orchestrates all evaluators and produces a unified :class:`EvaluationReport`.

    Parameters
    ----------
    evaluators:
        Optional list of :class:`~agentic_eval.evaluators.base.BaseEvaluator`
        instances.  Defaults to all four built-in evaluators.
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator] | None = None,
    ) -> None:
        if evaluators is None:
            self._evaluators: list[BaseEvaluator] = [
                ReasoningEvaluator(),
                ToolUsageEvaluator(),
                WorkflowEvaluator(),
                TaskSuccessEvaluator(),
            ]
        else:
            self._evaluators = list(evaluators)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sample: dict[str, Any],
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Run all evaluators on *sample* and return a full report.

        Parameters
        ----------
        sample:
            Dictionary containing the agent interaction data.
        run_id:
            Optional identifier for this evaluation run.
        metadata:
            Optional key-value pairs attached to the report (e.g. model name).
        """
        results: list[EvaluationResult] = [
            ev.evaluate(sample) for ev in self._evaluators
        ]
        metrics = EvaluationMetrics(results)
        return EvaluationReport(metrics, run_id=run_id, metadata=metadata)

    def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[EvaluationReport]:
        """Run evaluation on every sample in *samples*.

        Parameters
        ----------
        samples:
            List of sample dictionaries.
        metadata:
            Optional key-value pairs attached to every report.

        Returns
        -------
        list[EvaluationReport]
            One report per sample in the same order as *samples*.
        """
        return [
            self.evaluate(s, metadata=metadata) for s in samples
        ]

    def aggregate_batch(
        self,
        samples: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Evaluate a batch and aggregate results into a single report.

        The aggregated score for each dimension is the mean across all samples.

        Parameters
        ----------
        samples:
            List of sample dictionaries.
        metadata:
            Optional key-value pairs attached to the aggregated report.

        Returns
        -------
        EvaluationReport
            A single report whose scores are averages over the batch.
        """
        reports = self.evaluate_batch(samples, metadata=metadata)
        if not reports:
            return EvaluationReport(EvaluationMetrics([]), metadata=metadata)

        evaluator_names = [ev.name for ev in self._evaluators]
        aggregated: list[EvaluationResult] = []
        for name in evaluator_names:
            individual_scores = [
                next(
                    r.normalised_score
                    for r in report.metrics.results
                    if r.evaluator == name
                )
                for report in reports
            ]
            import statistics
            mean_score = statistics.mean(individual_scores)
            aggregated.append(
                EvaluationResult(
                    evaluator=name,
                    score=mean_score,
                    details={"aggregated_over": len(samples)},
                )
            )

        return EvaluationReport(
            EvaluationMetrics(aggregated),
            metadata={**(metadata or {}), "batch_size": len(samples)},
        )
