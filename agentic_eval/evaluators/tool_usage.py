"""Tool-usage evaluator.

Measures how accurately an agent selects and invokes tools across four
sub-dimensions:

1. **Selection accuracy** – fraction of agent tool calls where the *correct*
   tool was chosen (matched against a reference sequence when provided).
2. **Parameter correctness** – fraction of tool calls where the parameters
   match the expected parameters.
3. **Redundancy** – penalty for unnecessary/duplicate tool calls beyond what
   was expected.
4. **Success rate** – fraction of tool calls that actually succeeded
   (``status == "success"`` or the call has no ``status`` field at all).

Expected sample keys
--------------------
``tool_calls`` : list[dict]
    Tool calls made by the agent.  Each element should contain:

    * ``tool`` (str) – name of the tool invoked.
    * ``parameters`` (dict, optional) – keyword arguments supplied.
    * ``status`` (str, optional) – ``"success"`` | ``"error"`` | ``"timeout"``.
    * ``result`` (Any, optional) – the value returned by the tool.

``expected_tool_calls`` : list[dict], optional
    Reference sequence of tool calls.  Each element should contain at least:

    * ``tool`` (str) – the expected tool name.
    * ``parameters`` (dict, optional) – expected parameters.
"""

from __future__ import annotations

from typing import Any

from agentic_eval.evaluators.base import BaseEvaluator, EvaluationResult


class ToolUsageEvaluator(BaseEvaluator):
    """Evaluate an agent's tool-usage behaviour."""

    name = "tool_usage"

    _WEIGHTS = {
        "selection_accuracy": 0.35,
        "parameter_correctness": 0.30,
        "redundancy": 0.15,
        "success_rate": 0.20,
    }

    def _evaluate(self, sample: dict[str, Any]) -> EvaluationResult:
        tool_calls: list[dict[str, Any]] = sample.get("tool_calls") or []
        expected_calls: list[dict[str, Any]] = sample.get("expected_tool_calls") or []

        scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        # 1. Tool-selection accuracy
        if expected_calls:
            correct_tool = sum(
                1
                for i, exp in enumerate(expected_calls)
                if i < len(tool_calls)
                and tool_calls[i].get("tool") == exp.get("tool")
            )
            selection_score = correct_tool / len(expected_calls)
        else:
            selection_score = 1.0 if tool_calls else 1.0  # not penalised

        scores["selection_accuracy"] = selection_score
        details["selection_accuracy"] = {
            "expected_calls": len(expected_calls),
            "correct_tool_selections": (
                int(selection_score * len(expected_calls)) if expected_calls else None
            ),
            "score": selection_score,
        }

        # 2. Parameter correctness
        if expected_calls:
            correct_params = sum(
                1
                for i, exp in enumerate(expected_calls)
                if i < len(tool_calls)
                and tool_calls[i].get("tool") == exp.get("tool")
                and _params_match(
                    tool_calls[i].get("parameters") or {},
                    exp.get("parameters") or {},
                )
            )
            param_score = correct_params / len(expected_calls)
        else:
            param_score = 1.0

        scores["parameter_correctness"] = param_score
        details["parameter_correctness"] = {
            "score": param_score,
        }

        # 3. Redundancy (extra calls beyond expected)
        if expected_calls:
            extra = max(0, len(tool_calls) - len(expected_calls))
            # Each extra call deducts 0.1 (capped at 1.0 penalty)
            redundancy_score = max(0.0, 1.0 - 0.1 * extra)
        else:
            redundancy_score = 1.0

        scores["redundancy"] = redundancy_score
        details["redundancy"] = {
            "calls_made": len(tool_calls),
            "calls_expected": len(expected_calls) if expected_calls else None,
            "extra_calls": max(0, len(tool_calls) - len(expected_calls)) if expected_calls else 0,
            "score": redundancy_score,
        }

        # 4. Success rate
        if tool_calls:
            successes = sum(
                1
                for call in tool_calls
                if call.get("status", "success") == "success"
            )
            success_score = successes / len(tool_calls)
        else:
            success_score = 1.0

        scores["success_rate"] = success_score
        details["success_rate"] = {
            "total_calls": len(tool_calls),
            "successful_calls": (
                int(success_score * len(tool_calls)) if tool_calls else 0
            ),
            "score": success_score,
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

def _params_match(got: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when every expected parameter key-value pair is present in *got*."""
    if not expected:
        return True
    for key, exp_val in expected.items():
        if key not in got:
            return False
        got_val = got[key]
        # String: case-insensitive, strip whitespace
        if isinstance(exp_val, str) and isinstance(got_val, str):
            if exp_val.strip().lower() != got_val.strip().lower():
                return False
        elif got_val != exp_val:
            return False
    return True
