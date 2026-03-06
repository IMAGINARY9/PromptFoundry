"""Custom function evaluator.

This module provides an evaluator that wraps user-defined functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class EvaluatorFunction(Protocol):
    """Protocol for custom evaluation functions.

    Custom functions should accept predicted and expected strings
    and return a score between 0.0 and 1.0.
    """

    def __call__(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate a prediction.

        Args:
            predicted: The LLM's output.
            expected: The expected output.
            metadata: Optional additional context.

        Returns:
            Score between 0.0 and 1.0.
        """
        ...


class CustomFunctionEvaluator:
    """Evaluator that wraps a user-defined function.

    Allows custom evaluation logic without subclassing.

    Example:
        >>> def my_evaluator(pred: str, exp: str, meta: dict | None = None) -> float:
        ...     return 1.0 if "keyword" in pred.lower() else 0.0
        >>> evaluator = CustomFunctionEvaluator(my_evaluator)
        >>> evaluator.evaluate("Contains keyword here", "expected")
        1.0
    """

    def __init__(
        self,
        func: Callable[[str, str, dict[str, Any] | None], float],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize with a custom evaluation function.

        Args:
            func: Custom evaluation function.
            name: Optional name for the evaluator.
            description: Optional description.
        """
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or ""

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate using the custom function.

        Args:
            predicted: The LLM's output.
            expected: The expected output.
            metadata: Optional additional context.

        Returns:
            Score from the custom function, clamped to [0.0, 1.0].
        """
        score = self._func(predicted, expected, metadata)
        # Ensure score is in valid range
        return max(0.0, min(1.0, float(score)))

    def evaluate_batch(
        self,
        predictions: list[str],
        expected: list[str],
    ) -> list[float]:
        """Score multiple predictions.

        Args:
            predictions: List of LLM outputs.
            expected: List of expected outputs.

        Returns:
            List of scores.
        """
        if len(predictions) != len(expected):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(expected)} expected"
            )
        return [self.evaluate(pred, exp) for pred, exp in zip(predictions, expected, strict=True)]

    def aggregate(self, scores: list[float]) -> float:
        """Aggregate scores into a single value.

        Args:
            scores: List of individual scores.

        Returns:
            Mean score.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        return {
            "name": self._name,
            "type": "custom_function",
            "description": self._description,
        }

    @property
    def name(self) -> str:
        """Return evaluator name."""
        return self._name


class CompositeEvaluator:
    """Evaluator that combines multiple evaluators with weights.

    Useful for multi-objective optimization where different
    aspects of the output need to be evaluated.

    Example:
        >>> from promptfoundry.evaluators import ExactMatchEvaluator, ContainsEvaluator
        >>> composite = CompositeEvaluator([
        ...     (ExactMatchEvaluator(), 0.7),
        ...     (ContainsEvaluator(), 0.3),
        ... ])
        >>> composite.evaluate("hello world", "hello world")
        1.0
    """

    def __init__(
        self,
        evaluators: list[tuple[Any, float]],
        normalize_weights: bool = True,
    ) -> None:
        """Initialize composite evaluator.

        Args:
            evaluators: List of (evaluator, weight) tuples.
            normalize_weights: If True, normalize weights to sum to 1.0.
        """
        self._evaluators = evaluators

        # Extract weights and optionally normalize
        weights = [w for _, w in evaluators]
        if normalize_weights:
            total = sum(weights)
            if total > 0:
                self._weights = [w / total for w in weights]
            else:
                self._weights = [1.0 / len(weights)] * len(weights)
        else:
            self._weights = weights

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate using weighted combination of evaluators.

        Args:
            predicted: The LLM's output.
            expected: The expected output.
            metadata: Optional additional context.

        Returns:
            Weighted average of all evaluator scores.
        """
        total_score = 0.0

        for (evaluator, _), weight in zip(self._evaluators, self._weights, strict=True):
            score = evaluator.evaluate(predicted, expected, metadata)
            total_score += score * weight

        return total_score

    def evaluate_batch(
        self,
        predictions: list[str],
        expected: list[str],
    ) -> list[float]:
        """Score multiple predictions.

        Args:
            predictions: List of LLM outputs.
            expected: List of expected outputs.

        Returns:
            List of weighted scores.
        """
        return [self.evaluate(pred, exp) for pred, exp in zip(predictions, expected, strict=True)]

    def aggregate(self, scores: list[float]) -> float:
        """Aggregate scores into a single value."""
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        components = []
        for (evaluator, _), weight in zip(self._evaluators, self._weights, strict=True):
            info = evaluator.get_evaluator_info()
            info["weight"] = weight
            components.append(info)

        return {
            "name": "composite",
            "type": "composite",
            "components": components,
        }
