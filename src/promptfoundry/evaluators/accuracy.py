"""Accuracy-based evaluators.

This module provides evaluators that measure accuracy based on
string matching (exact or fuzzy).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
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
        allow_terminal_punctuation: bool = True,
    ) -> None:
        """Initialize exact-match evaluation behavior."""
        super().__init__(case_sensitive, strip_whitespace)
        self.normalize_output = normalize_output
        self.allow_terminal_punctuation = allow_terminal_punctuation

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
        elif self.allow_terminal_punctuation and pred != exp:
            pred = self._preprocess(self._strip_terminal_punctuation(predicted))
            exp = self._preprocess(self._strip_terminal_punctuation(expected))
        return 1.0 if pred == exp else 0.0

    @staticmethod
    def _strip_terminal_punctuation(text: str) -> str:
        """Strip trivial trailing punctuation without recovering explanations."""
        return re.sub(r"[\s.!?,;:]+$", "", text.strip())

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator metadata including normalization behavior."""
        info = super().get_evaluator_info()
        info["normalize_output"] = self.normalize_output
        info["allow_terminal_punctuation"] = self.allow_terminal_punctuation
        return info


class NumericAnswerEvaluator(BaseEvaluator):
    """Evaluator for strict numeric-answer tasks with partial format credit.

    A perfect score is reserved for bare numeric answers only. Outputs that
    contain the correct number inside additional prose receive partial credit
    so the optimizer retains useful signal without over-scoring verbose answers.
    """

    _BARE_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
    _EMBEDDED_NUMBER_PATTERN = re.compile(r"[-+]?(?:\$)?(?:\d[\d,]*\.?\d*|\.\d+)")

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        allow_terminal_punctuation: bool = True,
        prose_partial_credit: float = 0.6,
        last_number_partial_credit: float = 0.4,
        embedded_number_partial_credit: float = 0.25,
    ) -> None:
        """Initialize the numeric-answer evaluator.

        Args:
            case_sensitive: Unused, kept for evaluator interface consistency.
            strip_whitespace: Whether to trim leading/trailing whitespace.
            allow_terminal_punctuation: Whether to accept bare answers like `42.`.
            prose_partial_credit: Score when the only number is correct but wrapped in prose.
            last_number_partial_credit: Score when the final number is correct among many numbers.
            embedded_number_partial_credit: Score when the correct number appears but is not final.
        """
        super().__init__(case_sensitive=case_sensitive, strip_whitespace=strip_whitespace)
        self.allow_terminal_punctuation = allow_terminal_punctuation
        self.prose_partial_credit = prose_partial_credit
        self.last_number_partial_credit = last_number_partial_credit
        self.embedded_number_partial_credit = embedded_number_partial_credit

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Score numeric answers while keeping numeric-only outputs strict."""
        del metadata
        expected_number = self._parse_number(expected)
        if expected_number is None:
            return 0.0

        bare_prediction = self._extract_bare_number(predicted)
        if bare_prediction is not None and bare_prediction == expected_number:
            return 1.0

        extracted_numbers = self._extract_numbers(predicted)
        if not extracted_numbers:
            return 0.0

        matching_indexes = [index for index, number in enumerate(extracted_numbers) if number == expected_number]
        if not matching_indexes:
            return 0.0

        if len(extracted_numbers) == 1:
            return self.prose_partial_credit
        if matching_indexes[-1] == len(extracted_numbers) - 1:
            return self.last_number_partial_credit
        return self.embedded_number_partial_credit

    def _extract_bare_number(self, text: str) -> Decimal | None:
        candidate = text.strip() if self.strip_whitespace else text
        if self.allow_terminal_punctuation:
            candidate = ExactMatchEvaluator._strip_terminal_punctuation(candidate)
        if not self._BARE_NUMBER_PATTERN.fullmatch(candidate):
            return None
        return self._parse_number(candidate)

    def _extract_numbers(self, text: str) -> list[Decimal]:
        numbers: list[Decimal] = []
        for match in self._EMBEDDED_NUMBER_PATTERN.finditer(text):
            parsed = self._parse_number(match.group(0))
            if parsed is not None:
                numbers.append(parsed)
        return numbers

    @staticmethod
    def _parse_number(text: str) -> Decimal | None:
        cleaned = text.strip()
        cleaned = re.sub(r"^[^\d+\-.]+", "", cleaned)
        cleaned = re.sub(r"[^\d.\-+,]+$", "", cleaned)
        cleaned = cleaned.replace(",", "")
        if not cleaned:
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information including partial-credit behavior."""
        info = super().get_evaluator_info()
        info.update({
            "allow_terminal_punctuation": self.allow_terminal_punctuation,
            "prose_partial_credit": self.prose_partial_credit,
            "last_number_partial_credit": self.last_number_partial_credit,
            "embedded_number_partial_credit": self.embedded_number_partial_credit,
        })
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
