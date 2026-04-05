"""Tests for ToolUsageEvaluator."""

import pytest
from agentic_eval.evaluators.tool_usage import ToolUsageEvaluator


@pytest.fixture
def evaluator():
    return ToolUsageEvaluator()


class TestSelectionAccuracy:
    def test_perfect_tool_selection(self, evaluator):
        sample = {
            "tool_calls": [
                {"tool": "search", "parameters": {"query": "python"}},
                {"tool": "calculator", "parameters": {"expression": "2+2"}},
            ],
            "expected_tool_calls": [
                {"tool": "search"},
                {"tool": "calculator"},
            ],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["selection_accuracy"] == pytest.approx(1.0)

    def test_wrong_tool_selection(self, evaluator):
        sample = {
            "tool_calls": [{"tool": "wrong_tool"}],
            "expected_tool_calls": [{"tool": "search"}],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["selection_accuracy"] == pytest.approx(0.0)

    def test_no_expected_calls(self, evaluator):
        sample = {"tool_calls": [{"tool": "search", "status": "success"}]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["selection_accuracy"] == pytest.approx(1.0)


class TestParameterCorrectness:
    def test_correct_parameters(self, evaluator):
        sample = {
            "tool_calls": [{"tool": "search", "parameters": {"query": "ai"}}],
            "expected_tool_calls": [{"tool": "search", "parameters": {"query": "ai"}}],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["parameter_correctness"] == pytest.approx(1.0)

    def test_wrong_parameters(self, evaluator):
        sample = {
            "tool_calls": [{"tool": "search", "parameters": {"query": "wrong"}}],
            "expected_tool_calls": [{"tool": "search", "parameters": {"query": "ai"}}],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["parameter_correctness"] == pytest.approx(0.0)

    def test_extra_parameters_allowed(self, evaluator):
        """Extra params in the call beyond expected are fine."""
        sample = {
            "tool_calls": [{"tool": "search", "parameters": {"query": "ai", "top_k": 5}}],
            "expected_tool_calls": [{"tool": "search", "parameters": {"query": "ai"}}],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["parameter_correctness"] == pytest.approx(1.0)


class TestRedundancy:
    def test_no_extra_calls(self, evaluator):
        sample = {
            "tool_calls": [{"tool": "search", "status": "success"}],
            "expected_tool_calls": [{"tool": "search"}],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["redundancy"] == pytest.approx(1.0)

    def test_one_extra_call(self, evaluator):
        sample = {
            "tool_calls": [
                {"tool": "search", "status": "success"},
                {"tool": "search", "status": "success"},  # extra
            ],
            "expected_tool_calls": [{"tool": "search"}],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["redundancy"] == pytest.approx(0.9)

    def test_ten_extra_calls_capped(self, evaluator):
        sample = {
            "tool_calls": [{"tool": "t", "status": "success"}] * 11,
            "expected_tool_calls": [{"tool": "t"}],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["redundancy"] == pytest.approx(0.0)


class TestSuccessRate:
    def test_all_successful(self, evaluator):
        sample = {
            "tool_calls": [
                {"tool": "a", "status": "success"},
                {"tool": "b", "status": "success"},
            ]
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["success_rate"] == pytest.approx(1.0)

    def test_half_successful(self, evaluator):
        sample = {
            "tool_calls": [
                {"tool": "a", "status": "success"},
                {"tool": "b", "status": "error"},
            ]
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["success_rate"] == pytest.approx(0.5)

    def test_no_status_defaults_to_success(self, evaluator):
        sample = {"tool_calls": [{"tool": "a"}]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["success_rate"] == pytest.approx(1.0)

    def test_empty_calls(self, evaluator):
        sample = {"tool_calls": []}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["success_rate"] == pytest.approx(1.0)


class TestAggregateScore:
    def test_perfect_sample(self, evaluator):
        sample = {
            "tool_calls": [{"tool": "search", "parameters": {"q": "x"}, "status": "success"}],
            "expected_tool_calls": [{"tool": "search", "parameters": {"q": "x"}}],
        }
        result = evaluator.evaluate(sample)
        assert result.normalised_score == pytest.approx(1.0)

    def test_evaluator_name(self, evaluator):
        result = evaluator.evaluate({})
        assert result.evaluator == "tool_usage"
