"""Tests for evaluators."""

from __future__ import annotations

from typing import Any

import pytest

from promptfoundry.evaluators.accuracy import ExactMatchEvaluator, FuzzyMatchEvaluator
from promptfoundry.evaluators.custom import CompositeEvaluator, CustomFunctionEvaluator
from promptfoundry.evaluators.format import ContainsEvaluator, RegexEvaluator


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

    def test_exact_match_normalizes_short_label_answers(self) -> None:
        """Verbose completions should normalize down to the expected label."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate(
            "Based on the review, the sentiment is positive.",
            "positive",
        )
        assert score == 1.0

    def test_exact_match_normalizes_numeric_answers(self) -> None:
        """Numeric answers should be extracted from short explanatory completions."""
        evaluator = ExactMatchEvaluator()
        score = evaluator.evaluate("The final answer is 42.", "42")
        assert score == 1.0

    def test_exact_match_can_disable_normalization(self) -> None:
        """Normalization should remain configurable for strict comparisons."""
        evaluator = ExactMatchEvaluator(normalize_output=False)
        score = evaluator.evaluate("The final answer is 42.", "42")
        assert score == 0.0


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

    def test_regex_substitutes_expected_placeholder(self) -> None:
        """Fixed regex patterns may interpolate the expected answer safely."""
        evaluator = RegexEvaluator(pattern=r"\b{expected}\b", full_match=True)
        assert evaluator.evaluate("42", "42") == 1.0

    def test_regex_placeholder_full_match_rejects_verbose_answer(self) -> None:
        """Strict regex tasks should not score explanatory answers as exact hits."""
        evaluator = RegexEvaluator(pattern=r"\b{expected}\b", full_match=True)
        assert evaluator.evaluate("The answer is 42.", "42") == 0.0


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


class TestCustomFunctionEvaluator:
    """Tests for CustomFunctionEvaluator."""

    def test_custom_function_basic(self) -> None:
        """Test basic custom function evaluation."""

        def keyword_checker(
            pred: str, exp: str, meta: dict[str, Any] | None = None
        ) -> float:
            return 1.0 if "keyword" in pred.lower() else 0.0

        evaluator = CustomFunctionEvaluator(keyword_checker)
        assert evaluator.evaluate("Contains keyword here", "anything") == 1.0
        assert evaluator.evaluate("No match", "anything") == 0.0

    def test_custom_function_with_metadata(self) -> None:
        """Test custom function that uses metadata."""

        def weighted_match(
            pred: str, exp: str, meta: dict[str, Any] | None = None
        ) -> float:
            base = 1.0 if pred == exp else 0.0
            weight = (meta or {}).get("weight", 1.0)
            return base * weight

        evaluator = CustomFunctionEvaluator(weighted_match)
        assert evaluator.evaluate("test", "test", {"weight": 0.5}) == 0.5

    def test_custom_function_score_clamping(self) -> None:
        """Test that scores are clamped to [0.0, 1.0]."""

        def bad_scorer(
            pred: str, exp: str, meta: dict[str, Any] | None = None
        ) -> float:
            return 2.0  # Invalid score

        evaluator = CustomFunctionEvaluator(bad_scorer)
        assert evaluator.evaluate("a", "b") == 1.0  # Clamped to max

    def test_custom_function_batch(self) -> None:
        """Test batch evaluation."""

        def length_match(
            pred: str, exp: str, meta: dict[str, Any] | None = None
        ) -> float:
            return 1.0 if len(pred) == len(exp) else 0.0

        evaluator = CustomFunctionEvaluator(length_match)
        scores = evaluator.evaluate_batch(["abc", "ab", "abcd"], ["xyz", "x", "wxyz"])
        assert scores == [1.0, 0.0, 1.0]

    def test_custom_function_info(self) -> None:
        """Test evaluator info retrieval."""

        def my_func(
            pred: str, exp: str, meta: dict[str, Any] | None = None
        ) -> float:
            """My custom evaluator function."""
            return 0.5

        evaluator = CustomFunctionEvaluator(my_func, name="Custom", description="Test")
        info = evaluator.get_evaluator_info()
        assert info["name"] == "Custom"
        assert info["type"] == "custom_function"


class TestCompositeEvaluator:
    """Tests for CompositeEvaluator."""

    def test_composite_weighted_average(self) -> None:
        """Test weighted average of multiple evaluators."""
        exact = ExactMatchEvaluator()
        contains = ContainsEvaluator()

        composite = CompositeEvaluator([
            (exact, 0.7),
            (contains, 0.3),
        ])

        # Exact match and contains both succeed
        score = composite.evaluate("positive", "positive")
        assert score == 1.0

        # Only contains succeeds
        score = composite.evaluate("I am positive", "positive")
        assert score == pytest.approx(0.3)

    def test_composite_equal_weights(self) -> None:
        """Test equal weights normalization."""
        exact = ExactMatchEvaluator()
        contains = ContainsEvaluator()

        composite = CompositeEvaluator([
            (exact, 1.0),
            (contains, 1.0),
        ])

        # Both succeed
        score = composite.evaluate("positive", "positive")
        assert score == 1.0

        # Only contains
        score = composite.evaluate("I feel positive today", "positive")
        assert score == pytest.approx(0.5)

    def test_composite_batch(self) -> None:
        """Test batch evaluation with composite."""
        exact = ExactMatchEvaluator()
        composite = CompositeEvaluator([(exact, 1.0)])

        scores = composite.evaluate_batch(
            ["yes", "no", "yes"],
            ["yes", "yes", "no"],
        )
        assert scores == [1.0, 0.0, 0.0]

    def test_composite_info(self) -> None:
        """Test composite evaluator info."""
        exact = ExactMatchEvaluator()
        contains = ContainsEvaluator()

        composite = CompositeEvaluator([
            (exact, 0.6),
            (contains, 0.4),
        ])

        info = composite.get_evaluator_info()
        assert info["type"] == "composite"
        assert len(info["components"]) == 2
