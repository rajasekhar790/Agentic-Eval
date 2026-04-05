"""Tests for WorkflowEvaluator."""

import pytest
from agentic_eval.evaluators.workflow import WorkflowEvaluator


@pytest.fixture
def evaluator():
    return WorkflowEvaluator()


class TestStepCompletion:
    def test_all_steps_completed(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "completed"},
            ],
            "total_steps": 2,
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["step_completion"] == pytest.approx(1.0)

    def test_half_steps_completed(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "failed"},
            ],
            "total_steps": 2,
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["step_completion"] == pytest.approx(0.5)

    def test_agent_stopped_early(self, evaluator):
        """total_steps > len(steps): agent did not attempt all steps."""
        sample = {
            "steps": [{"id": 1, "status": "completed"}],
            "total_steps": 4,
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["step_completion"] == pytest.approx(0.25)

    def test_no_steps(self, evaluator):
        sample = {"steps": [], "total_steps": 0}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["step_completion"] == pytest.approx(1.0)


class TestErrorRecovery:
    def test_no_failures(self, evaluator):
        sample = {"steps": [{"id": 1, "status": "completed"}]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["error_recovery"] == pytest.approx(1.0)

    def test_recovered_from_all_failures(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "failed", "recovered": True},
                {"id": 2, "status": "failed", "recovered": True},
            ]
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["error_recovery"] == pytest.approx(1.0)

    def test_no_recovery(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "failed", "recovered": False},
            ]
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["error_recovery"] == pytest.approx(0.0)

    def test_partial_recovery(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "failed", "recovered": True},
                {"id": 2, "status": "failed", "recovered": False},
            ]
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["error_recovery"] == pytest.approx(0.5)


class TestStateConsistency:
    def test_consistent_states(self, evaluator):
        sample = {
            "steps": [
                {"id": "s1", "status": "completed", "state": {"count": 1}},
                {"id": "s2", "status": "completed", "state": {"count": 2}},
            ],
            "expected_states": {
                "s1": {"count": 1},
                "s2": {"count": 2},
            },
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["state_consistency"] == pytest.approx(1.0)

    def test_inconsistent_state(self, evaluator):
        sample = {
            "steps": [
                {"id": "s1", "status": "completed", "state": {"count": 99}},
            ],
            "expected_states": {"s1": {"count": 1}},
        }
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["state_consistency"] == pytest.approx(0.0)

    def test_no_expected_states(self, evaluator):
        sample = {"steps": [{"id": 1, "status": "completed", "state": {"x": 1}}]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["state_consistency"] == pytest.approx(1.0)


class TestRetryEfficiency:
    def test_no_retries(self, evaluator):
        sample = {"steps": [{"id": 1, "status": "completed", "retries": 0}]}
        result = evaluator.evaluate(sample)
        assert result.details["sub_scores"]["retry_efficiency"] == pytest.approx(1.0)

    def test_one_retry_on_failed_step_is_efficient(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "failed", "retries": 1},
            ]
        }
        result = evaluator.evaluate(sample)
        # 1 retry for 1 failed step → no excessive retries
        assert result.details["sub_scores"]["retry_efficiency"] == pytest.approx(1.0)

    def test_excessive_retries(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "failed", "retries": 5},
            ]
        }
        result = evaluator.evaluate(sample)
        # 5 retries, 1 failed step → 4 excessive → score = 1 - 0.1*4 = 0.6
        assert result.details["sub_scores"]["retry_efficiency"] == pytest.approx(0.6)


class TestAggregateScore:
    def test_perfect_workflow(self, evaluator):
        sample = {
            "steps": [
                {"id": 1, "status": "completed", "retries": 0, "state": {"done": True}},
                {"id": 2, "status": "completed", "retries": 0, "state": {"done": True}},
            ],
            "total_steps": 2,
            "expected_states": {1: {"done": True}, 2: {"done": True}},
        }
        result = evaluator.evaluate(sample)
        assert result.normalised_score == pytest.approx(1.0)

    def test_evaluator_name(self, evaluator):
        result = evaluator.evaluate({})
        assert result.evaluator == "workflow"
