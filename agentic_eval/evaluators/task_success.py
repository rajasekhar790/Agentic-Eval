"""Task-success evaluator.

Measures overall task completion across four sub-dimensions:

1. **Binary success** – whether the task was fully completed (``status ==
   "success"``).
2. **Partial completion** – a 0–1 score of how much of the task was achieved
   even when full success was not reached.
3. **Goal achievement** – fraction of defined sub-goals that were reached.
4. **Efficiency** – bonus/penalty based on whether the agent completed the
   task within the expected number of steps (or time budget).

Expected sample keys
--------------------
``status`` : str
    ``"success"`` | ``"partial"`` | ``"failure"``.
``completion_score`` : float, optional
    Explicit partial-completion score in [0, 1].  If absent, inferred from
    ``status``.
``goals`` : list[dict], optional
    Sub-goals for the task.  Each element should contain:

    * ``id`` (str | int) – goal identifier.
    * ``achieved`` (bool) – whether the sub-goal was reached.

``step_budget`` : int, optional
    Expected / budget number of steps.
``time_taken`` : float, optional
    Wall-clock seconds the agent used.
``time_budget`` : float, optional
    Allowed time budget in seconds.
"""

from __future__ import annotations

from typing import Any

from agentic_eval.evaluators.base import BaseEvaluator, EvaluationResult


class TaskSuccessEvaluator(BaseEvaluator):
    """Evaluate the overall task success of an agentic system."""

    name = "task_success"

    _WEIGHTS = {
        "binary_success": 0.30,
        "partial_completion": 0.25,
        "goal_achievement": 0.30,
        "efficiency": 0.15,
    }

    _STATUS_SCORES = {
        "success": 1.0,
        "partial": 0.5,
        "failure": 0.0,
    }

    def _evaluate(self, sample: dict[str, Any]) -> EvaluationResult:
        scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        status: str = (sample.get("status") or "failure").lower()

        # 1. Binary success
        binary_score = self._STATUS_SCORES.get(status, 0.0)
        scores["binary_success"] = binary_score
        details["binary_success"] = {"status": status, "score": binary_score}

        # 2. Partial completion
        if "completion_score" in sample:
            partial_score = float(sample["completion_score"])
            partial_score = max(0.0, min(1.0, partial_score))
        else:
            partial_score = binary_score  # fall back to binary

        scores["partial_completion"] = partial_score
        details["partial_completion"] = {"score": partial_score}

        # 3. Goal achievement
        goals: list[dict[str, Any]] = sample.get("goals") or []
        if goals:
            achieved = sum(1 for g in goals if g.get("achieved", False))
            goal_score = achieved / len(goals)
        else:
            goal_score = binary_score  # no sub-goals defined: use binary

        scores["goal_achievement"] = goal_score
        details["goal_achievement"] = {
            "total_goals": len(goals),
            "achieved_goals": int(goal_score * len(goals)) if goals else 0,
            "score": goal_score,
        }

        # 4. Efficiency
        efficiency_score = self._compute_efficiency(sample)
        scores["efficiency"] = efficiency_score
        details["efficiency"] = {
            "steps_taken": sample.get("steps_taken"),
            "step_budget": sample.get("step_budget"),
            "time_taken": sample.get("time_taken"),
            "time_budget": sample.get("time_budget"),
            "score": efficiency_score,
        }

        aggregate = sum(scores[k] * w for k, w in self._WEIGHTS.items())

        return EvaluationResult(
            evaluator=self.name,
            score=aggregate,
            details={"sub_scores": scores, **details},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_efficiency(self, sample: dict[str, Any]) -> float:
        """Return an efficiency score in [0, 1].

        Uses step budget first; falls back to time budget.
        """
        steps_taken = sample.get("steps_taken")
        expected_steps = sample.get("step_budget")
        if steps_taken is not None and expected_steps and expected_steps > 0:
            ratio = steps_taken / expected_steps
            return _budget_score(ratio)

        time_taken = sample.get("time_taken")
        time_budget = sample.get("time_budget")
        if time_taken is not None and time_budget and time_budget > 0:
            ratio = time_taken / time_budget
            return _budget_score(ratio)

        return 1.0  # no budget information provided


def _budget_score(ratio: float) -> float:
    """Convert a usage ratio to a score.

    * ratio ≤ 1.0 (within budget) → 1.0
    * ratio up to 2.0 → linear decay from 1.0 to 0.0
    * ratio > 2.0 → 0.0
    """
    if ratio <= 1.0:
        return 1.0
    if ratio >= 2.0:
        return 0.0
    return 1.0 - (ratio - 1.0)
