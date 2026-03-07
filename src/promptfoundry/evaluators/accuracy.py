"""Accuracy-based evaluators.

This module provides evaluators that measure accuracy based on
string matching (exact or fuzzy).
"""

from __future__ import annotations

from typing import Any

from promptfoundry.evaluators.base import BaseEvaluator
from promptfoundry.evaluators.normalization import normalize_for_exact_match


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator that checks for exact string match.

    Returns 1.0 if predicted equals expected, 0.0 otherwise.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        normalize_output: bool = True,
    ) -> None:
        """Initialize exact-match evaluation behavior."""
        super().__init__(case_sensitive, strip_whitespace)
        self.normalize_output = normalize_output

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Check for exact match.

        Args:
            predicted: The LLM's output.
            expected: The expected output.
            metadata: Unused.

        Returns:
            1.0 if exact match, 0.0 otherwise.
        """
        pred = self._preprocess(predicted)
        exp = self._preprocess(expected)
        if self.normalize_output and pred != exp:
            pred = self._preprocess(
                normalize_for_exact_match(
                    predicted,
                    expected,
                    case_sensitive=self.case_sensitive,
                )
            )
        return 1.0 if pred == exp else 0.0

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator metadata including normalization behavior."""
        info = super().get_evaluator_info()
        info["normalize_output"] = self.normalize_output
        return info


class FuzzyMatchEvaluator(BaseEvaluator):
    """Evaluator that uses fuzzy string matching.

    Uses Levenshtein distance normalized by string length.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize the fuzzy match evaluator.

        Args:
            threshold: Minimum similarity to consider a match (0.0-1.0).
            case_sensitive: Whether comparison is case-sensitive.
            strip_whitespace: Whether to strip whitespace.
        """
        super().__init__(case_sensitive, strip_whitespace)
        self.threshold = threshold

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Calculate fuzzy similarity.

        Args:
            predicted: The LLM's output.
            expected: The expected output.
            metadata: Unused.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        pred = self._preprocess(predicted)
        exp = self._preprocess(expected)

        if pred == exp:
            return 1.0

        if not pred or not exp:
            return 0.0

        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(pred, exp)
        max_len = max(len(pred), len(exp))
        similarity = 1.0 - (distance / max_len)

        return similarity

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Edit distance.
        """
        if len(s1) < len(s2):
            return FuzzyMatchEvaluator._levenshtein_distance(s2, s1)

        if not s2:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information including threshold."""
        info = super().get_evaluator_info()
        info["threshold"] = self.threshold
        return info
