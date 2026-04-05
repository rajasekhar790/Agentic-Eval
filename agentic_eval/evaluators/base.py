"""Base evaluator interface shared by all dimension evaluators."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Container for the result produced by one evaluator on one sample."""

    evaluator: str
    score: float  # 0.0 – 1.0 normalised score
    max_score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def normalised_score(self) -> float:
        """Score normalised to [0, 1]."""
        if self.max_score == 0:
            return 0.0
        return max(0.0, min(1.0, self.score / self.max_score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "score": self.score,
            "max_score": self.max_score,
            "normalised_score": self.normalised_score,
            "details": self.details,
            "errors": self.errors,
        }


class BaseEvaluator(abc.ABC):
    """Abstract base class that every dimension evaluator must subclass."""

    #: Human-readable name for this evaluator dimension.
    name: str = "base"

    def evaluate(self, sample: dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a single sample and return an :class:`EvaluationResult`.

        Parameters
        ----------
        sample:
            A dictionary describing the agent interaction to be evaluated.
            The keys expected depend on the concrete evaluator; see each
            subclass for full documentation.

        Returns
        -------
        EvaluationResult
        """
        errors: list[str] = []
        try:
            result = self._evaluate(sample)
        except Exception as exc:  # noqa: BLE001
            result = EvaluationResult(
                evaluator=self.name,
                score=0.0,
                errors=[str(exc)],
            )
        return result

    @abc.abstractmethod
    def _evaluate(self, sample: dict[str, Any]) -> EvaluationResult:
        """Subclasses implement the actual evaluation logic here."""
