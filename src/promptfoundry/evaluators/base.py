"""Base evaluator class for objective functions.

This module provides the abstract base class that all evaluators
must extend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Abstract base class for evaluators (objective functions).

    Evaluators score LLM outputs against expected outputs.
    Scores should be in the range [0.0, 1.0] where 1.0 is perfect.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            case_sensitive: Whether comparison is case-sensitive.
            strip_whitespace: Whether to strip leading/trailing whitespace.
        """
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def _preprocess(self, text: str) -> str:
        """Preprocess text before comparison.

        Args:
            text: Text to preprocess.

        Returns:
            Preprocessed text.
        """
        if self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    @abstractmethod
    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Score a single prediction.

        Args:
            predicted: The LLM's output.
            expected: The expected/desired output.
            metadata: Optional additional context.

        Returns:
            Score between 0.0 (worst) and 1.0 (best).
        """
        pass

    def evaluate_batch(
        self,
        predictions: list[str],
        expected: list[str],
    ) -> list[float]:
        """Score multiple predictions.

        Default implementation calls evaluate() for each pair.
        Subclasses may override for batch optimization.

        Args:
            predictions: List of LLM outputs.
            expected: List of expected outputs.

        Returns:
            List of scores, one per prediction.
        """
        if len(predictions) != len(expected):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(expected)} expected"
            )

        return [self.evaluate(pred, exp) for pred, exp in zip(predictions, expected, strict=True)]

    def aggregate(self, scores: list[float]) -> float:
        """Aggregate multiple scores into a single fitness value.

        Default implementation returns the mean.

        Args:
            scores: List of individual scores.

        Returns:
            Aggregated score.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return information about this evaluator.

        Returns:
            Dictionary with evaluator metadata.
        """
        return {
            "name": self.__class__.__name__,
            "case_sensitive": self.case_sensitive,
            "strip_whitespace": self.strip_whitespace,
        }
