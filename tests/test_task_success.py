"""Tests for TaskSuccessEvaluator."""

import pytest
from agentic_eval.evaluators.task_success import TaskSuccessEvaluator, _budget_score


@pytest.fixture
def evaluator():
    return TaskSuccessEvaluator()


class TestBinarySuccess:
    def test_success_status(self, evaluator):
        result = evaluator.evaluate({"status": "success"})
        assert result.details["sub_scores"]["binary_success"] == pytest.approx(1.0)

    def test_partial_status(self, evaluator):
        result = evaluator.evaluate({"status": "partial"})
        assert result.details["sub_scores"]["binary_success"] == pytest.approx(0.5)

    def test_failure_status(self, evaluator):
        result = evaluator.evaluate({"status": "failure"})
        assert result.details["sub_scores"]["binary_success"] == pytest.approx(0.0)

    def test_unknown_status(self, evaluator):
        result = evaluator.evaluate({"status": "unknown"})
        assert result.details["sub_scores"]["binary_success"] == pytest.approx(0.0)

    def test_missing_status_defaults_to_failure(self, evaluator):
        result = evaluator.evaluate({})
        assert result.details["sub_scores"]["binary_success"] == pytest.approx(0.0)


class TestPartialCompletion:
    def test_explicit_completion_score(self, evaluator):
        result = evaluator.evaluate({"status": "partial", "completion_score": 0.7})
        assert result.details["sub_scores"]["partial_completion"] == pytest.approx(0.7)

    def test_clamp_above_one(self, evaluator):
        result = evaluator.evaluate({"status": "success", "completion_score": 1.5})
        assert result.details["sub_scores"]["partial_completion"] == pytest.approx(1.0)

    def test_clamp_below_zero(self, evaluator):
        result = evaluator.evaluate({"status": "failure", "completion_score": -0.5})
        assert result.details["sub_scores"]["partial_completion"] == pytest.approx(0.0)

    def test_falls_back_to_binary_when_absent(self, evaluator):
        result = evaluator.evaluate({"status": "success"})
        assert result.details["sub_scores"]["partial_completion"] == pytest.approx(1.0)


class TestGoalAchievement:
    def test_all_goals_achieved(self, evaluator):
        sample = {
            "status": "success",
            "goals": [
                {"id": "g1", "achieved": True},
                {"id": "g2", "achieved": True},
            ],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["goal_achievement"] == pytest.approx(1.0)

    def test_no_goals_achieved(self, evaluator):
        sample = {
            "status": "failure",
            "goals": [
                {"id": "g1", "achieved": False},
                {"id": "g2", "achieved": False},
            ],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["goal_achievement"] == pytest.approx(0.0)

    def test_half_goals_achieved(self, evaluator):
        sample = {
            "status": "partial",
            "goals": [
                {"id": "g1", "achieved": True},
                {"id": "g2", "achieved": False},
            ],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["goal_achievement"] == pytest.approx(0.5)

    def test_no_goals_defined_uses_binary(self, evaluator):
        result = evaluator.evaluate({"status": "success"})
        assert result.details["sub_scores"]["goal_achievement"] == pytest.approx(1.0)


class TestEfficiency:
    def test_within_step_budget(self, evaluator):
        result = evaluator.evaluate({"status": "success", "steps_taken": 5, "step_budget": 10})
        assert result.details["sub_scores"]["efficiency"] == pytest.approx(1.0)

    def test_at_step_budget(self, evaluator):
        result = evaluator.evaluate({"status": "success", "steps_taken": 10, "step_budget": 10})
        assert result.details["sub_scores"]["efficiency"] == pytest.approx(1.0)

    def test_slightly_over_budget(self, evaluator):
        result = evaluator.evaluate({"status": "success", "steps_taken": 15, "step_budget": 10})
        # ratio = 1.5 → score = 1 - (1.5 - 1.0) = 0.5
        assert result.details["sub_scores"]["efficiency"] == pytest.approx(0.5)

    def test_double_budget_is_zero(self, evaluator):
        result = evaluator.evaluate({"status": "success", "steps_taken": 20, "step_budget": 10})
        assert result.details["sub_scores"]["efficiency"] == pytest.approx(0.0)

    def test_time_budget_fallback(self, evaluator):
        result = evaluator.evaluate({"status": "success", "time_taken": 30.0, "time_budget": 60.0})
        assert result.details["sub_scores"]["efficiency"] == pytest.approx(1.0)

    def test_no_budget_info(self, evaluator):
        result = evaluator.evaluate({"status": "success"})
        assert result.details["sub_scores"]["efficiency"] == pytest.approx(1.0)


class TestBudgetScoreHelper:
    def test_within_budget(self):
        assert _budget_score(0.5) == pytest.approx(1.0)
        assert _budget_score(1.0) == pytest.approx(1.0)

    def test_over_budget(self):
        assert _budget_score(1.5) == pytest.approx(0.5)

    def test_double_budget(self):
        assert _budget_score(2.0) == pytest.approx(0.0)

    def test_beyond_double(self):
        assert _budget_score(3.0) == pytest.approx(0.0)


class TestAggregateScore:
    def test_perfect_sample(self, evaluator):
        sample = {
            "status": "success",
            "completion_score": 1.0,
            "goals": [{"id": "g1", "achieved": True}],
            "steps_taken": 5,
            "step_budget": 10,
        }
        result = evaluator.evaluate(sample)
        assert result.normalised_score == pytest.approx(1.0)

    def test_evaluator_name(self, evaluator):
        result = evaluator.evaluate({})
        assert result.evaluator == "task_success"
