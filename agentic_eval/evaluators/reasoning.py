"""Reasoning evaluator.

Measures the quality of an agent's reasoning traces across four sub-dimensions:

1. **Chain-of-thought completeness** – the proportion of expected reasoning
   steps that appear in the agent's trace.
2. **Logical consistency** – whether the agent's conclusion is consistent with
   its stated reasoning (no contradictions flagged explicitly).
3. **Intermediate-step correctness** – fraction of intermediate steps whose
   answer matches the expected answer (when reference answers are provided).
4. **Conclusion accuracy** – whether the final answer/conclusion matches the
   ground-truth answer.

Expected sample keys
--------------------
``reasoning_steps`` : list[str]
    The reasoning steps produced by the agent.
``expected_reasoning_steps`` : list[str], optional
    Reference reasoning steps used to check chain-of-thought completeness.
``intermediate_answers`` : list[Any], optional
    Answers produced at intermediate steps.
``expected_intermediate_answers`` : list[Any], optional
    Ground-truth answers for each intermediate step.
``conclusion`` : Any, optional
    The agent's final answer / conclusion.
``expected_conclusion`` : Any, optional
    Ground-truth final answer.
``contradictions`` : list[str], optional
    Explicit contradiction flags already detected upstream. If absent,
    logical consistency is assumed perfect.
"""

from __future__ import annotations

from typing import Any

from agentic_eval.evaluators.base import BaseEvaluator, EvaluationResult


class ReasoningEvaluator(BaseEvaluator):
    """Evaluate the reasoning quality of an agentic system."""

    name = "reasoning"

    # Weights for the four sub-dimensions (must sum to 1.0)
    _WEIGHTS = {
        "chain_of_thought": 0.30,
        "logical_consistency": 0.25,
        "intermediate_steps": 0.20,
        "conclusion_accuracy": 0.25,
    }

    def _evaluate(self, sample: dict[str, Any]) -> EvaluationResult:
        scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        # 1. Chain-of-thought completeness
        reasoning_steps: list[str] = sample.get("reasoning_steps") or []
        expected_steps: list[str] = sample.get("expected_reasoning_steps") or []
        if expected_steps:
            matched = sum(
                1
                for exp in expected_steps
                if any(exp.lower() in step.lower() for step in reasoning_steps)
            )
            cot_score = matched / len(expected_steps)
        else:
            # No reference steps: award full marks if at least one step present
            cot_score = 1.0 if reasoning_steps else 0.0

        scores["chain_of_thought"] = cot_score
        details["chain_of_thought"] = {
            "steps_produced": len(reasoning_steps),
            "steps_expected": len(expected_steps),
            "score": cot_score,
        }

        # 2. Logical consistency
        contradictions: list[str] = sample.get("contradictions") or []
        consistency_score = max(0.0, 1.0 - 0.25 * len(contradictions))
        scores["logical_consistency"] = consistency_score
        details["logical_consistency"] = {
            "contradictions_found": len(contradictions),
            "score": consistency_score,
        }

        # 3. Intermediate-step correctness
        intermediate_answers: list[Any] = sample.get("intermediate_answers") or []
        expected_intermediate: list[Any] = (
            sample.get("expected_intermediate_answers") or []
        )
        if expected_intermediate:
            pairs = zip(intermediate_answers, expected_intermediate)
            correct = sum(1 for got, exp in pairs if _answers_match(got, exp))
            step_score = correct / len(expected_intermediate)
        else:
            step_score = 1.0  # not evaluated

        scores["intermediate_steps"] = step_score
        details["intermediate_steps"] = {
            "steps_checked": len(expected_intermediate),
            "score": step_score,
        }

        # 4. Conclusion accuracy
        conclusion = sample.get("conclusion")
        expected_conclusion = sample.get("expected_conclusion")
        if expected_conclusion is not None:
            conclusion_score = 1.0 if _answers_match(conclusion, expected_conclusion) else 0.0
        else:
            conclusion_score = 1.0  # not evaluated

        scores["conclusion_accuracy"] = conclusion_score
        details["conclusion_accuracy"] = {
            "conclusion": conclusion,
            "expected": expected_conclusion,
            "score": conclusion_score,
        }

        # Weighted aggregate
        aggregate = sum(
            scores[k] * w for k, w in self._WEIGHTS.items()
        )

        return EvaluationResult(
            evaluator=self.name,
            score=aggregate,
            details={"sub_scores": scores, **details},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answers_match(got: Any, expected: Any) -> bool:
    """Return True when *got* is considered equivalent to *expected*."""
    if got is None and expected is None:
        return True
    if got is None or expected is None:
        return False
    # String comparison: normalise whitespace and case
    if isinstance(got, str) and isinstance(expected, str):
        return got.strip().lower() == expected.strip().lower()
    return got == expected
