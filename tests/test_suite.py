"""Integration tests for EvaluationSuite."""

import json
import pytest

from agentic_eval.suite import EvaluationSuite
from agentic_eval.reporters.report import EvaluationReport


PERFECT_SAMPLE = {
    # Reasoning
    "reasoning_steps": ["identify facts", "apply logic", "derive answer"],
    "expected_reasoning_steps": ["identify facts", "apply logic", "derive answer"],
    "conclusion": "42",
    "expected_conclusion": "42",
    # Tool usage
    "tool_calls": [
        {"tool": "calculator", "parameters": {"expression": "6*7"}, "status": "success"}
    ],
    "expected_tool_calls": [
        {"tool": "calculator", "parameters": {"expression": "6*7"}}
    ],
    # Workflow
    "steps": [
        {"id": 1, "status": "completed", "retries": 0, "state": {"result": 42}},
    ],
    "total_steps": 1,
    "expected_states": {1: {"result": 42}},
    # Task success
    "status": "success",
    "completion_score": 1.0,
    "goals": [{"id": "main", "achieved": True}],
    "steps_taken": 3,
    "step_budget": 5,
}

FAILING_SAMPLE = {
    "reasoning_steps": [],
    "expected_steps": ["step a", "step b"],
    "conclusion": "wrong",
    "expected_conclusion": "right",
    "tool_calls": [{"tool": "bad_tool", "status": "error"}],
    "expected_tool_calls": [{"tool": "search"}],
    "steps": [{"id": 1, "status": "failed", "recovered": False}],
    "total_steps": 5,
    "status": "failure",
    "goals": [{"id": "g1", "achieved": False}],
}


class TestEvaluateSingle:
    def test_returns_report(self):
        suite = EvaluationSuite()
        report = suite.evaluate(PERFECT_SAMPLE)
        assert isinstance(report, EvaluationReport)

    def test_perfect_sample_high_score(self):
        suite = EvaluationSuite()
        report = suite.evaluate(PERFECT_SAMPLE)
        assert report.metrics.overall_score == pytest.approx(1.0)

    def test_failing_sample_low_score(self):
        suite = EvaluationSuite()
        report = suite.evaluate(FAILING_SAMPLE)
        assert report.metrics.overall_score < 0.5

    def test_report_has_four_evaluators(self):
        suite = EvaluationSuite()
        report = suite.evaluate(PERFECT_SAMPLE)
        names = {r.evaluator for r in report.metrics.results}
        assert names == {"reasoning", "tool_usage", "workflow", "task_success"}

    def test_run_id_attached(self):
        suite = EvaluationSuite()
        report = suite.evaluate(PERFECT_SAMPLE, run_id="my-run-001")
        assert report.run_id == "my-run-001"

    def test_metadata_attached(self):
        suite = EvaluationSuite()
        report = suite.evaluate(PERFECT_SAMPLE, metadata={"model": "gpt-test"})
        assert report.metadata["model"] == "gpt-test"


class TestEvaluateBatch:
    def test_returns_list_of_reports(self):
        suite = EvaluationSuite()
        reports = suite.evaluate_batch([PERFECT_SAMPLE, FAILING_SAMPLE])
        assert len(reports) == 2
        assert all(isinstance(r, EvaluationReport) for r in reports)

    def test_empty_batch(self):
        suite = EvaluationSuite()
        reports = suite.evaluate_batch([])
        assert reports == []


class TestAggregateBatch:
    def test_aggregate_returns_single_report(self):
        suite = EvaluationSuite()
        report = suite.aggregate_batch([PERFECT_SAMPLE, FAILING_SAMPLE])
        assert isinstance(report, EvaluationReport)

    def test_aggregate_score_is_between_individual(self):
        suite = EvaluationSuite()
        perfect = suite.evaluate(PERFECT_SAMPLE).metrics.overall_score
        failing = suite.evaluate(FAILING_SAMPLE).metrics.overall_score
        aggregated = suite.aggregate_batch([PERFECT_SAMPLE, FAILING_SAMPLE])
        agg_score = aggregated.metrics.overall_score
        assert failing <= agg_score <= perfect

    def test_batch_size_in_metadata(self):
        suite = EvaluationSuite()
        report = suite.aggregate_batch([PERFECT_SAMPLE, FAILING_SAMPLE])
        assert report.metadata["batch_size"] == 2

    def test_empty_aggregate(self):
        suite = EvaluationSuite()
        report = suite.aggregate_batch([])
        assert isinstance(report, EvaluationReport)


class TestJsonSerialisability:
    def test_to_dict(self):
        suite = EvaluationSuite()
        report = suite.evaluate(PERFECT_SAMPLE)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "summary" in d
        assert "details" in d

    def test_to_json(self):
        suite = EvaluationSuite()
        report = suite.evaluate(PERFECT_SAMPLE)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert "summary" in parsed


class TestCustomEvaluators:
    def test_single_custom_evaluator(self):
        from agentic_eval.evaluators.reasoning import ReasoningEvaluator

        suite = EvaluationSuite(evaluators=[ReasoningEvaluator()])
        report = suite.evaluate(PERFECT_SAMPLE)
        assert len(report.metrics.results) == 1
        assert report.metrics.results[0].evaluator == "reasoning"


class TestErrorHandling:
    def test_exception_in_sample_returns_zero_score(self):
        """An evaluator receiving a malformed value should not propagate an exception."""
        from agentic_eval.evaluators.base import BaseEvaluator, EvaluationResult

        class BrokenEvaluator(BaseEvaluator):
            name = "broken"

            def _evaluate(self, sample):
                raise ValueError("intentional test error")

        suite = EvaluationSuite(evaluators=[BrokenEvaluator()])
        report = suite.evaluate({})
        result = report.metrics.results[0]
        assert result.score == pytest.approx(0.0)
        assert len(result.errors) > 0
