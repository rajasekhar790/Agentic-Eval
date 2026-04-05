"""Evaluation report module.

:class:`EvaluationReport` wraps an :class:`~agentic_eval.metrics.EvaluationMetrics`
object and provides human-readable console output as well as a JSON-serialisable
representation of the full evaluation run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agentic_eval.metrics.metrics import EvaluationMetrics


class EvaluationReport:
    """Format and render evaluation results."""

    def __init__(
        self,
        metrics: EvaluationMetrics,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.metrics = metrics
        self.run_id = run_id or _generate_run_id()
        self.metadata = metadata or {}
        self.timestamp = datetime.now(tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as a JSON-serialisable dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "summary": self.metrics.summary(),
            "details": [r.to_dict() for r in self.metrics.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    # Console rendering
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        summary = self.metrics.summary()
        width = 60
        print("=" * width)
        print(f"  Agentic-Eval Report  |  run: {self.run_id}")
        print(f"  {self.timestamp}")
        print("=" * width)
        for name, info in summary["evaluators"].items():
            bar = _score_bar(info["score"])
            status = "✓" if info["score"] >= 0.5 else "✗"
            print(f"  {status} {name:<22} {bar}  {info['score']:.4f}")
            if info["errors"]:
                for err in info["errors"]:
                    print(f"      ⚠  {err}")
        print("-" * width)
        overall = summary["overall_score"]
        verdict = "PASS ✓" if summary["passed"] else "FAIL ✗"
        print(f"  Overall score: {overall:.4f}   →  {verdict}")
        print("=" * width)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_run_id() -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    import random
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    return f"run-{ts}-{suffix}"


def _score_bar(score: float, width: int = 20) -> str:
    """Return a simple ASCII progress bar for *score* in [0, 1]."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"
