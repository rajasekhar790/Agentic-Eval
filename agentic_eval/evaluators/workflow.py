"""Workflow reliability evaluator.

Measures how reliably an agent completes multi-step workflows across four
sub-dimensions:

1. **Step completion rate** – fraction of workflow steps that were actually
   completed.
2. **Error recovery** – fraction of encountered errors from which the agent
   successfully recovered.
3. **State consistency** – whether the agent's reported state at each
   completed step is consistent with the expected state.
4. **Retry efficiency** – penalises excessive retry attempts (more than one
   retry per failed step is considered inefficient).

Expected sample keys
--------------------
``steps`` : list[dict]
    Workflow steps attempted by the agent.  Each element should contain:

    * ``id`` (str | int) – unique step identifier.
    * ``status`` (str) – ``"completed"`` | ``"failed"`` | ``"skipped"``.
    * ``state`` (dict, optional) – agent state after the step.
    * ``retries`` (int, optional) – number of retry attempts for this step.
    * ``error`` (str, optional) – error message if the step failed.
    * ``recovered`` (bool, optional) – whether the agent recovered from the
      error (default ``False``).

``expected_states`` : dict[str | int, dict], optional
    Mapping from step id to expected state dictionary.
``total_steps`` : int, optional
    Total number of steps in the workflow (used when not all steps appear
    in ``steps`` because the agent stopped early).
"""

from __future__ import annotations

from typing import Any

from agentic_eval.evaluators.base import BaseEvaluator, EvaluationResult


class WorkflowEvaluator(BaseEvaluator):
    """Evaluate the reliability of an agent's multi-step workflow execution."""

    name = "workflow"

    _WEIGHTS = {
        "step_completion": 0.35,
        "error_recovery": 0.25,
        "state_consistency": 0.25,
        "retry_efficiency": 0.15,
    }

    def _evaluate(self, sample: dict[str, Any]) -> EvaluationResult:
        steps: list[dict[str, Any]] = sample.get("steps") or []
        expected_states: dict[Any, dict] = sample.get("expected_states") or {}
        total_steps: int = sample.get("total_steps") or len(steps)

        scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        # 1. Step completion rate
        completed = sum(
            1 for s in steps if s.get("status") == "completed"
        )
        if total_steps > 0:
            completion_score = completed / total_steps
        else:
            completion_score = 1.0

        scores["step_completion"] = completion_score
        details["step_completion"] = {
            "completed": completed,
            "total": total_steps,
            "score": completion_score,
        }

        # 2. Error recovery rate
        failed_steps = [s for s in steps if s.get("status") == "failed"]
        if failed_steps:
            recovered = sum(
                1 for s in failed_steps if s.get("recovered", False)
            )
            recovery_score = recovered / len(failed_steps)
        else:
            recovery_score = 1.0  # no failures → perfect

        scores["error_recovery"] = recovery_score
        details["error_recovery"] = {
            "failed_steps": len(failed_steps),
            "recovered": int(recovery_score * len(failed_steps)) if failed_steps else 0,
            "score": recovery_score,
        }

        # 3. State consistency
        if expected_states:
            checked = 0
            consistent = 0
            for step in steps:
                step_id = step.get("id")
                if step_id in expected_states and step.get("state") is not None:
                    checked += 1
                    if _states_consistent(step["state"], expected_states[step_id]):
                        consistent += 1
            state_score = consistent / checked if checked > 0 else 1.0
        else:
            state_score = 1.0

        scores["state_consistency"] = state_score
        details["state_consistency"] = {
            "states_checked": len(expected_states),
            "score": state_score,
        }

        # 4. Retry efficiency
        total_retries = sum(s.get("retries", 0) for s in steps)
        failed_count = len(failed_steps)
        # Allow one retry per failed step; everything beyond is penalised
        excessive_retries = max(0, total_retries - failed_count)
        retry_score = max(0.0, 1.0 - 0.1 * excessive_retries)
        scores["retry_efficiency"] = retry_score
        details["retry_efficiency"] = {
            "total_retries": total_retries,
            "excessive_retries": excessive_retries,
            "score": retry_score,
        }

        aggregate = sum(scores[k] * w for k, w in self._WEIGHTS.items())

        return EvaluationResult(
            evaluator=self.name,
            score=aggregate,
            details={"sub_scores": scores, **details},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _states_consistent(got: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when every key in *expected* matches in *got*."""
    for key, exp_val in expected.items():
        if key not in got:
            return False
        if got[key] != exp_val:
            return False
    return True
