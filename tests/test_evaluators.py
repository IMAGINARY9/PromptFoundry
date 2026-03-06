"""Tests for evaluators."""

from __future__ import annotations

import pytest

from promptfoundry.evaluators.accuracy import ExactMatchEvaluator, FuzzyMatchEvaluator
from promptfoundry.evaluators.format import RegexEvaluator, ContainsEvaluator


class TestExactMatchEvaluator:
    """Tests for ExactMatchEvaluator."""

    def test_exact_match_same(self) -> None:
        """Test exact match with identical strings."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate("positive", "positive")
        assert score == 1.0

    def test_exact_match_different(self) -> None:
        """Test exact match with different strings."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate("positive", "negative")
        assert score == 0.0

    def test_exact_match_case_insensitive(self) -> None:
        """Test case-insensitive matching (default)."""
        evaluator = ExactMatchEvaluator(case_sensitive=False)
        score = evaluator.evaluate("POSITIVE", "positive")
        assert score == 1.0

    def test_exact_match_case_sensitive(self) -> None:
        """Test case-sensitive matching."""
        evaluator = ExactMatchEvaluator(case_sensitive=True)
        score = evaluator.evaluate("POSITIVE", "positive")
        assert score == 0.0

    def test_exact_match_whitespace_strip(self) -> None:
        """Test whitespace stripping."""
        evaluator = ExactMatchEvaluator(strip_whitespace=True)
        score = evaluator.evaluate("  positive  ", "positive")
        assert score == 1.0

    def test_exact_match_batch(self) -> None:
        """Test batch evaluation."""
        evaluator = ExactMatchEvaluator()
        scores = evaluator.evaluate_batch(
            ["positive", "negative", "positive"],
            ["positive", "positive", "negative"],
        )
        assert scores == [1.0, 0.0, 0.0]

    def test_exact_match_aggregate(self) -> None:
        """Test score aggregation."""
        evaluator = ExactMatchEvaluator()
        agg = evaluator.aggregate([1.0, 0.0, 1.0, 0.0])
        assert agg == 0.5


class TestFuzzyMatchEvaluator:
    """Tests for FuzzyMatchEvaluator."""

    def test_fuzzy_exact_match(self) -> None:
        """Test fuzzy match with identical strings."""
        evaluator = FuzzyMatchEvaluator()
        score = evaluator.evaluate("positive", "positive")
        assert score == 1.0

    def test_fuzzy_similar_strings(self) -> None:
        """Test fuzzy match with similar strings."""
        evaluator = FuzzyMatchEvaluator()
        score = evaluator.evaluate("positiv", "positive")
        assert 0.8 < score < 1.0  # Should be close

    def test_fuzzy_different_strings(self) -> None:
        """Test fuzzy match with different strings."""
        evaluator = FuzzyMatchEvaluator()
        score = evaluator.evaluate("abc", "xyz")
        assert score < 0.5

    def test_fuzzy_empty_strings(self) -> None:
        """Test fuzzy match with empty strings."""
        evaluator = FuzzyMatchEvaluator()
        assert evaluator.evaluate("", "") == 1.0
        assert evaluator.evaluate("test", "") == 0.0
        assert evaluator.evaluate("", "test") == 0.0


class TestRegexEvaluator:
    """Tests for RegexEvaluator."""

    def test_regex_fixed_pattern_match(self) -> None:
        """Test matching with fixed pattern."""
        evaluator = RegexEvaluator(pattern=r"positive|negative|neutral")
        assert evaluator.evaluate("positive", "") == 1.0
        assert evaluator.evaluate("maybe", "") == 0.0

    def test_regex_full_match(self) -> None:
        """Test full string match requirement."""
        evaluator = RegexEvaluator(pattern=r"\d+", full_match=False)
        assert evaluator.evaluate("has 123 numbers", "") == 1.0

        evaluator_full = RegexEvaluator(pattern=r"\d+", full_match=True)
        assert evaluator_full.evaluate("has 123 numbers", "") == 0.0
        assert evaluator_full.evaluate("123", "") == 1.0

    def test_regex_use_expected_as_pattern(self) -> None:
        """Test using expected output as regex pattern."""
        evaluator = RegexEvaluator(use_expected_as_pattern=True)
        assert evaluator.evaluate("error code 404", r"\d+") == 1.0


class TestContainsEvaluator:
    """Tests for ContainsEvaluator."""

    def test_contains_substring(self) -> None:
        """Test substring containment."""
        evaluator = ContainsEvaluator()
        assert evaluator.evaluate("The answer is positive.", "positive") == 1.0
        assert evaluator.evaluate("The answer is negative.", "positive") == 0.0

    def test_contains_case_insensitive(self) -> None:
        """Test case-insensitive containment."""
        evaluator = ContainsEvaluator(case_sensitive=False)
        assert evaluator.evaluate("POSITIVE result", "positive") == 1.0
