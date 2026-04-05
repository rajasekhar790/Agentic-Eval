"""Tests for ReasoningEvaluator."""

import pytest
from agentic_eval.evaluators.reasoning import ReasoningEvaluator


@pytest.fixture
def evaluator():
    return ReasoningEvaluator()


class TestChainOfThought:
    def test_perfect_cot_match(self, evaluator):
        sample = {
            "reasoning_steps": ["identify the variables", "apply the formula", "calculate the result"],
            "expected_reasoning_steps": ["identify the variables", "apply the formula", "calculate the result"],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["chain_of_thought"] == pytest.approx(1.0)

    def test_partial_cot_match(self, evaluator):
        sample = {
            "reasoning_steps": ["identify the variables", "calculate the result"],
            "expected_reasoning_steps": ["identify the variables", "apply the formula", "calculate the result"],
        }
        result = evaluator.evaluate(sample)
        # 2 out of 3 expected steps matched
        assert result.details["sub_scores"]["chain_of_thought"] == pytest.approx(2 / 3)

    def test_no_expected_steps_with_reasoning(self, evaluator):
        sample = {"reasoning_steps": ["step 1", "step 2"]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["chain_of_thought"] == pytest.approx(1.0)

    def test_no_reasoning_steps(self, evaluator):
        sample = {}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["chain_of_thought"] == pytest.approx(0.0)

    def test_cot_case_insensitive(self, evaluator):
        sample = {
            "reasoning_steps": ["IDENTIFY THE VARIABLES"],
            "expected_reasoning_steps": ["identify the variables"],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["chain_of_thought"] == pytest.approx(1.0)


class TestLogicalConsistency:
    def test_no_contradictions(self, evaluator):
        sample = {"reasoning_steps": ["step 1"]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["logical_consistency"] == pytest.approx(1.0)

    def test_one_contradiction(self, evaluator):
        sample = {
            "reasoning_steps": ["step 1"],
            "contradictions": ["conclusion contradicts premise"],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["logical_consistency"] == pytest.approx(0.75)

    def test_four_contradictions_capped_at_zero(self, evaluator):
        sample = {
            "reasoning_steps": ["step 1"],
            "contradictions": ["c1", "c2", "c3", "c4"],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["logical_consistency"] == pytest.approx(0.0)


class TestIntermediateSteps:
    def test_all_intermediate_correct(self, evaluator):
        sample = {
            "intermediate_answers": ["4", "12"],
            "expected_intermediate_answers": ["4", "12"],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["intermediate_steps"] == pytest.approx(1.0)

    def test_none_intermediate_correct(self, evaluator):
        sample = {
            "intermediate_answers": ["3", "11"],
            "expected_intermediate_answers": ["4", "12"],
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["intermediate_steps"] == pytest.approx(0.0)

    def test_no_expected_intermediate(self, evaluator):
        sample = {"intermediate_answers": ["any"]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["intermediate_steps"] == pytest.approx(1.0)


class TestConclusionAccuracy:
    def test_correct_conclusion(self, evaluator):
        sample = {"conclusion": "Paris", "expected_conclusion": "paris"}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["conclusion_accuracy"] == pytest.approx(1.0)

    def test_wrong_conclusion(self, evaluator):
        sample = {"conclusion": "London", "expected_conclusion": "Paris"}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["conclusion_accuracy"] == pytest.approx(0.0)

    def test_no_expected_conclusion(self, evaluator):
        sample = {"conclusion": "anything"}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["conclusion_accuracy"] == pytest.approx(1.0)


class TestAggregateScore:
    def test_perfect_sample(self, evaluator):
        sample = {
            "reasoning_steps": ["step a", "step b"],
            "expected_reasoning_steps": ["step a", "step b"],
            "conclusion": "42",
            "expected_conclusion": "42",
        }
        result = evaluator.evaluate(sample)
        assert result.normalised_score == pytest.approx(1.0)

    def test_score_in_range(self, evaluator):
        sample = {
            "reasoning_steps": ["step a"],
            "expected_reasoning_steps": ["step a", "step b", "step c"],
            "conclusion": "wrong",
            "expected_conclusion": "right",
            "contradictions": ["c1"],
        }
        result = evaluator.evaluate(sample)
        assert 0.0 <= result.normalised_score <= 1.0

    def test_evaluator_name(self, evaluator):
        result = evaluator.evaluate({})
        assert result.evaluator == "reasoning"
