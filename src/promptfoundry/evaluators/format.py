"""Format-based evaluators.

This module provides evaluators that check output format
compliance using regex patterns or JSON schemas.
"""

from __future__ import annotations

import re
from typing import Any

from promptfoundry.evaluators.base import BaseEvaluator


class RegexEvaluator(BaseEvaluator):
    """Evaluator that checks if output matches a regex pattern.

    Can use the expected output as the pattern or a fixed pattern.
    """

    def __init__(
        self,
        pattern: str | None = None,
        use_expected_as_pattern: bool = False,
        full_match: bool = False,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize the regex evaluator.

        Args:
            pattern: Fixed regex pattern to match against.
            use_expected_as_pattern: If True, treat expected as regex pattern.
            full_match: If True, require full string match instead of search.
            case_sensitive: Whether pattern is case-sensitive.
            strip_whitespace: Whether to strip whitespace from predicted.
        """
        super().__init__(case_sensitive, strip_whitespace)
        self.pattern = pattern
        self.use_expected_as_pattern = use_expected_as_pattern
        self.full_match = full_match

        # Pre-compile fixed pattern if provided
        self._compiled_pattern: re.Pattern[str] | None = None
        if pattern:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._compiled_pattern = re.compile(pattern, flags)

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Check if predicted matches the pattern.

        Args:
            predicted: The LLM's output.
            expected: Expected output (used as pattern if use_expected_as_pattern).
            metadata: Unused.

        Returns:
            1.0 if matches, 0.0 otherwise.
        """
        pred = self._preprocess(predicted)

        # Determine which pattern to use
        if self.use_expected_as_pattern:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            pattern = re.compile(expected, flags)
        elif self._compiled_pattern:
            pattern = self._compiled_pattern
        else:
            # No pattern specified, fall back to exact match
            exp = self._preprocess(expected)
            return 1.0 if pred == exp else 0.0

        # Match
        if self.full_match:
            match = pattern.fullmatch(pred)
        else:
            match = pattern.search(pred)

        return 1.0 if match else 0.0

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        info = super().get_evaluator_info()
        info.update(
            {
                "pattern": self.pattern,
                "use_expected_as_pattern": self.use_expected_as_pattern,
                "full_match": self.full_match,
            }
        )
        return info


class ContainsEvaluator(BaseEvaluator):
    """Evaluator that checks if expected text is contained in output."""

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Check if predicted contains expected.

        Args:
            predicted: The LLM's output.
            expected: Text that should appear in output.
            metadata: Unused.

        Returns:
            1.0 if expected is in predicted, 0.0 otherwise.
        """
        pred = self._preprocess(predicted)
        exp = self._preprocess(expected)
        return 1.0 if exp in pred else 0.0
