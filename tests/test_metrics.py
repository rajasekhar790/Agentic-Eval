"""Tests for EvaluationMetrics."""

import pytest
from agentic_eval.evaluators.base import EvaluationResult
from agentic_eval.metrics.metrics import EvaluationMetrics


def _make_result(name: str, score: float) -> EvaluationResult:
    return EvaluationResult(evaluator=name, score=score)


class TestOverallScore:
    def test_mean_of_scores(self):
        results = [
            _make_result("reasoning", 0.8),
            _make_result("tool_usage", 0.6),
        ]
        metrics = EvaluationMetrics(results)
        assert metrics.overall_score == pytest.approx(0.7)

    def test_empty_results(self):
        metrics = EvaluationMetrics([])
        assert metrics.overall_score == pytest.approx(0.0)

    def test_single_result(self):
        metrics = EvaluationMetrics([_make_result("reasoning", 0.9)])
        assert metrics.overall_score == pytest.approx(0.9)


class TestPassFail:
    def test_pass_threshold(self):
        metrics = EvaluationMetrics([_make_result("r", 0.5)])
        assert metrics.passed is True

    def test_fail_threshold(self):
        metrics = EvaluationMetrics([_make_result("r", 0.49)])
        assert metrics.passed is False


class TestScoresByEvaluator:
    def test_mapping(self):
        results = [
            _make_result("reasoning", 0.8),
            _make_result("tool_usage", 0.6),
        ]
        metrics = EvaluationMetrics(results)
        assert metrics.scores_by_evaluator == {
            "reasoning": pytest.approx(0.8),
            "tool_usage": pytest.approx(0.6),
        }


class TestSummary:
    def test_summary_keys(self):
        results = [_make_result("reasoning", 0.8)]
        summary = EvaluationMetrics(results).summary()
        assert "overall_score" in summary
        assert "passed" in summary
        assert "evaluators" in summary

    def test_summary_values(self):
        results = [_make_result("reasoning", 1.0)]
        summary = EvaluationMetrics(results).summary()
        assert summary["overall_score"] == pytest.approx(1.0)
        assert summary["passed"] is True
