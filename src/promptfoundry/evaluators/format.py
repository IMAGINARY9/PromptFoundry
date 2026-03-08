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
    Fixed patterns may also contain an ``{expected}`` placeholder which is
    replaced with an escaped copy of the expected output at evaluation time.
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

        # Pre-compile fixed patterns that do not depend on the expected value.
        self._compiled_pattern: re.Pattern[str] | None = None
        if pattern and "{expected}" not in pattern:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._compiled_pattern = re.compile(pattern, flags)

    def _build_pattern(self, expected: str) -> re.Pattern[str] | None:
        """Resolve the effective regex pattern for this evaluation."""
        flags = 0 if self.case_sensitive else re.IGNORECASE

        if self.use_expected_as_pattern:
            return re.compile(expected, flags)

        if self._compiled_pattern is not None:
            return self._compiled_pattern

        if self.pattern is None:
            return None

        if "{expected}" in self.pattern:
            resolved_pattern = self.pattern.replace("{expected}", re.escape(expected))
            return re.compile(resolved_pattern, flags)

        return re.compile(self.pattern, flags)

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
        exp = self._preprocess(expected)

        # Determine which pattern to use
        pattern = self._build_pattern(exp)
        if pattern is None:
            # No pattern specified, fall back to exact match
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
