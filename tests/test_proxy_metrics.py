"""Tests for cheap proxy metric evaluators.

These tests validate the proxy metric evaluators that provide
useful signal without requiring exact match evaluation.
"""

import pytest

from promptfoundry.evaluators.proxy_metrics import (
    FieldCoverageEvaluator,
    JsonParseEvaluator,
    JsonSchemaEvaluator,
    KeywordPresenceEvaluator,
    LengthConstraintEvaluator,
    OutputShapeEvaluator,
)


class TestJsonParseEvaluator:
    """Tests for JSON parse validation."""

    def test_valid_json_object(self) -> None:
        """Test valid JSON object returns 1.0."""
        evaluator = JsonParseEvaluator()
        assert evaluator.evaluate('{"key": "value"}', '') == 1.0

    def test_valid_json_array(self) -> None:
        """Test valid JSON array returns 1.0."""
        evaluator = JsonParseEvaluator()
        assert evaluator.evaluate('[1, 2, 3]', '') == 1.0

    def test_invalid_json(self) -> None:
        """Test invalid JSON returns 0.0."""
        evaluator = JsonParseEvaluator()
        assert evaluator.evaluate('not json at all', '') == 0.0

    def test_extract_json_from_markdown(self) -> None:
        """Test extraction from markdown code block."""
        evaluator = JsonParseEvaluator(extract_json=True)
        text = '''Here is the result:
```json
{"name": "Alice", "age": 30}
```
'''
        assert evaluator.evaluate(text, '') == 1.0

    def test_extract_json_from_text(self) -> None:
        """Test extraction from surrounding text."""
        evaluator = JsonParseEvaluator(extract_json=True)
        text = 'The answer is {"result": 42} as expected.'
        assert evaluator.evaluate(text, '') == 1.0

    def test_no_extraction(self) -> None:
        """Test that extraction can be disabled."""
        evaluator = JsonParseEvaluator(extract_json=False)
        text = 'The answer is {"result": 42} as expected.'
        assert evaluator.evaluate(text, '') == 0.0

    def test_json_type_constraint_object(self) -> None:
        """Test type constraint for objects."""
        evaluator = JsonParseEvaluator()
        assert evaluator.evaluate('{"key": "value"}', '', {"json_type": "object"}) == 1.0
        assert evaluator.evaluate('[1, 2, 3]', '', {"json_type": "object"}) == 0.0

    def test_json_type_constraint_array(self) -> None:
        """Test type constraint for arrays."""
        evaluator = JsonParseEvaluator()
        assert evaluator.evaluate('[1, 2, 3]', '', {"json_type": "array"}) == 1.0
        assert evaluator.evaluate('{"key": "value"}', '', {"json_type": "array"}) == 0.0


class TestJsonSchemaEvaluator:
    """Tests for JSON schema validation with partial credit."""

    def test_all_required_keys_present(self) -> None:
        """Test all required keys gives full score."""
        evaluator = JsonSchemaEvaluator(required_keys=["name", "age"])
        assert evaluator.evaluate('{"name": "Alice", "age": 30}', '') == 1.0

    def test_some_required_keys_missing(self) -> None:
        """Test partial credit for missing keys."""
        evaluator = JsonSchemaEvaluator(required_keys=["name", "age", "location"])
        score = evaluator.evaluate('{"name": "Alice", "age": 30}', '')
        assert score == pytest.approx(2/3)

    def test_no_required_keys_present(self) -> None:
        """Test no keys returns 0."""
        evaluator = JsonSchemaEvaluator(required_keys=["name", "age"])
        assert evaluator.evaluate('{"foo": "bar"}', '') == 0.0

    def test_type_checking(self) -> None:
        """Test type validation."""
        evaluator = JsonSchemaEvaluator(
            required_keys=["name"],
            key_types={"name": str, "age": int}
        )
        # Both key present and correct type
        score = evaluator.evaluate('{"name": "Alice", "age": 30}', '')
        assert score == 1.0  # 1 required key + 2 type checks = 3/3

        # Wrong type for age
        score = evaluator.evaluate('{"name": "Alice", "age": "thirty"}', '')
        assert score == pytest.approx(2/3)  # name key + name type, but not age type

    def test_invalid_json(self) -> None:
        """Test invalid JSON returns 0."""
        evaluator = JsonSchemaEvaluator(required_keys=["name"])
        assert evaluator.evaluate('not json', '') == 0.0

    def test_non_object_json(self) -> None:
        """Test non-object JSON returns 0 when checking keys."""
        evaluator = JsonSchemaEvaluator(required_keys=["name"])
        assert evaluator.evaluate('[1, 2, 3]', '') == 0.0

    def test_extract_from_markdown(self) -> None:
        """Test extraction from markdown."""
        evaluator = JsonSchemaEvaluator(required_keys=["result"])
        text = '```json\n{"result": 42}\n```'
        assert evaluator.evaluate(text, '') == 1.0


class TestFieldCoverageEvaluator:
    """Tests for field/pattern coverage evaluation."""

    def test_all_patterns_found(self) -> None:
        """Test all patterns found gives full score."""
        evaluator = FieldCoverageEvaluator(
            required_patterns=["Name:", "Age:", "Location:"]
        )
        output = "Name: Alice\nAge: 30\nLocation: NYC"
        assert evaluator.evaluate(output, '') == 1.0

    def test_partial_patterns_found(self) -> None:
        """Test partial credit for some patterns."""
        evaluator = FieldCoverageEvaluator(
            required_patterns=["Name:", "Age:", "Location:"]
        )
        output = "Name: Alice\nAge: 30"
        assert evaluator.evaluate(output, '') == pytest.approx(2/3)

    def test_no_patterns_found(self) -> None:
        """Test no patterns returns 0."""
        evaluator = FieldCoverageEvaluator(
            required_patterns=["Name:", "Age:"]
        )
        assert evaluator.evaluate("Random text", '') == 0.0

    def test_no_partial_credit(self) -> None:
        """Test binary mode (no partial credit)."""
        evaluator = FieldCoverageEvaluator(
            required_patterns=["Name:", "Age:"],
            partial_credit=False
        )
        assert evaluator.evaluate("Name: Alice", '') == 0.0
        assert evaluator.evaluate("Name: Alice\nAge: 30", '') == 1.0

    def test_regex_patterns(self) -> None:
        """Test regex pattern matching."""
        evaluator = FieldCoverageEvaluator(
            required_patterns=[r"Step \d+:", r"Total: \$?\d+"],
            use_regex=True
        )
        output = "Step 1: Do this\nStep 2: Do that\nTotal: $100"
        assert evaluator.evaluate(output, '') == 1.0

    def test_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        evaluator = FieldCoverageEvaluator(
            required_patterns=["ERROR", "WARNING"],
            case_sensitive=False
        )
        output = "error: something failed\nwarning: check this"
        assert evaluator.evaluate(output, '') == 1.0


class TestKeywordPresenceEvaluator:
    """Tests for keyword presence evaluation."""

    def test_all_keywords_present(self) -> None:
        """Test all keywords found."""
        evaluator = KeywordPresenceEvaluator(
            required_keywords=["positive", "sentiment", "good"]
        )
        output = "This is a positive sentiment analysis. The result is good."
        assert evaluator.evaluate(output, '') == 1.0

    def test_some_keywords_missing(self) -> None:
        """Test partial credit."""
        evaluator = KeywordPresenceEvaluator(
            required_keywords=["positive", "sentiment", "excellent"]
        )
        output = "This has positive sentiment."
        assert evaluator.evaluate(output, '') == pytest.approx(2/3)

    def test_weighted_keywords(self) -> None:
        """Test weighted importance."""
        evaluator = KeywordPresenceEvaluator(
            required_keywords=["critical", "minor"],
            weights={"critical": 3.0, "minor": 1.0}
        )
        # critical (weight 3) found, minor (weight 1) missing
        output = "This is a critical issue."
        # Score: 3 / (3+1) = 0.75
        assert evaluator.evaluate(output, '') == pytest.approx(0.75)

    def test_forbidden_keywords(self) -> None:
        """Test forbidden keywords deduct points."""
        evaluator = KeywordPresenceEvaluator(
            required_keywords=["answer"],
            forbidden_keywords=["error"]
        )
        # Has answer, no error
        assert evaluator.evaluate("Here is the answer.", '') == 1.0
        # Has answer and error - should deduct
        score = evaluator.evaluate("Here is the answer. Error!", '')
        assert score < 1.0

    def test_word_boundary(self) -> None:
        """Test word boundary matching."""
        evaluator = KeywordPresenceEvaluator(
            required_keywords=["test"],
            word_boundary=True
        )
        assert evaluator.evaluate("This is a test.", '') == 1.0
        assert evaluator.evaluate("Testing things.", '') == 0.0  # "test" not at boundary

    def test_no_word_boundary(self) -> None:
        """Test matching without word boundary."""
        evaluator = KeywordPresenceEvaluator(
            required_keywords=["test"],
            word_boundary=False
        )
        assert evaluator.evaluate("Testing things.", '') == 1.0


class TestLengthConstraintEvaluator:
    """Tests for length constraint evaluation."""

    def test_within_char_range(self) -> None:
        """Test output within character range."""
        evaluator = LengthConstraintEvaluator(min_chars=10, max_chars=100)
        output = "This is a medium length response."  # ~35 chars
        assert evaluator.evaluate(output, '') == 1.0

    def test_below_min_chars_linear(self) -> None:
        """Test linear penalty for too short."""
        evaluator = LengthConstraintEvaluator(min_chars=100, penalty_mode="linear")
        output = "Short"  # 5 chars
        score = evaluator.evaluate(output, '')
        assert score == pytest.approx(5 / 100)

    def test_above_max_chars_linear(self) -> None:
        """Test linear penalty for too long."""
        evaluator = LengthConstraintEvaluator(max_chars=10, penalty_mode="linear")
        output = "This is way too long"  # 20 chars
        score = evaluator.evaluate(output, '')
        assert score < 1.0
        assert score > 0.0

    def test_binary_mode(self) -> None:
        """Test binary (all or nothing) mode."""
        evaluator = LengthConstraintEvaluator(
            min_chars=10, max_chars=50, penalty_mode="binary"
        )
        assert evaluator.evaluate("Short", '') == 0.0
        assert evaluator.evaluate("This is adequate length text.", '') == 1.0

    def test_word_constraints(self) -> None:
        """Test word count constraints."""
        evaluator = LengthConstraintEvaluator(min_words=5, max_words=20)
        assert evaluator.evaluate("One two three four five six.", '') == 1.0
        assert evaluator.evaluate("Short.", '') < 1.0

    def test_line_constraints(self) -> None:
        """Test line count constraints."""
        evaluator = LengthConstraintEvaluator(min_lines=2, max_lines=5)
        output = "Line 1\nLine 2\nLine 3"
        assert evaluator.evaluate(output, '') == 1.0

    def test_combined_constraints(self) -> None:
        """Test multiple constraints (uses min score)."""
        evaluator = LengthConstraintEvaluator(
            min_chars=10, min_words=3
        )
        # Passes chars but not words
        output = "Aa bb"  # 5 chars, 2 words
        assert evaluator.evaluate(output, '') < 1.0


class TestOutputShapeEvaluator:
    """Tests for structural shape validation."""

    def test_starts_with(self) -> None:
        """Test prefix validation."""
        evaluator = OutputShapeEvaluator(starts_with="Answer:")
        assert evaluator.evaluate("Answer: 42", '') == 1.0
        assert evaluator.evaluate("The answer is 42", '') == 0.0

    def test_ends_with(self) -> None:
        """Test suffix validation."""
        evaluator = OutputShapeEvaluator(ends_with=".")
        assert evaluator.evaluate("This is a sentence.", '') == 1.0
        assert evaluator.evaluate("This is a sentence", '') == 0.0

    def test_contains_all(self) -> None:
        """Test all markers present."""
        evaluator = OutputShapeEvaluator(
            contains_all=["Step 1:", "Step 2:", "Conclusion:"]
        )
        output = "Step 1: First\nStep 2: Second\nConclusion: Done"
        assert evaluator.evaluate(output, '') == 1.0

        # Missing one
        output = "Step 1: First\nConclusion: Done"
        assert evaluator.evaluate(output, '') == pytest.approx(2/3)

    def test_contains_any(self) -> None:
        """Test at least one marker present."""
        evaluator = OutputShapeEvaluator(
            contains_any=["Yes", "Affirmative", "Confirmed"]
        )
        assert evaluator.evaluate("Yes, definitely", '') == 1.0
        assert evaluator.evaluate("I am unsure about this.", '') == 0.0

    def test_not_contains(self) -> None:
        """Test forbidden content."""
        evaluator = OutputShapeEvaluator(
            not_contains=["TODO", "FIXME", "ERROR"]
        )
        assert evaluator.evaluate("This is complete.", '') == 1.0
        assert evaluator.evaluate("TODO: finish this", '') == pytest.approx(2/3)

    def test_combined_shape_checks(self) -> None:
        """Test multiple shape checks."""
        evaluator = OutputShapeEvaluator(
            starts_with="Result:",
            ends_with=".",
            contains_all=["calculated"]
        )
        # All pass
        output = "Result: The value was calculated correctly."
        assert evaluator.evaluate(output, '') == 1.0

        # Missing suffix
        output = "Result: The value was calculated correctly"
        assert evaluator.evaluate(output, '') == pytest.approx(2/3)

    def test_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        evaluator = OutputShapeEvaluator(
            starts_with="ANSWER:",
            case_sensitive=False
        )
        assert evaluator.evaluate("answer: 42", '') == 1.0
        assert evaluator.evaluate("Answer: 42", '') == 1.0


class TestProxyMetricsIntegration:
    """Integration tests for proxy metric combinations."""

    def test_json_task_evaluation(self) -> None:
        """Test realistic JSON task evaluation."""
        # Simulate a JSON extraction task
        parse_evaluator = JsonParseEvaluator()
        schema_evaluator = JsonSchemaEvaluator(
            required_keys=["name", "email", "phone"]
        )

        good_output = '{"name": "Alice", "email": "alice@example.com", "phone": "123-456"}'
        partial_output = '{"name": "Alice", "email": "alice@example.com"}'
        bad_output = 'I found the contact: Alice, email: alice@example.com'

        # Good output passes both
        assert parse_evaluator.evaluate(good_output, '') == 1.0
        assert schema_evaluator.evaluate(good_output, '') == 1.0

        # Partial output passes parse but not full schema
        assert parse_evaluator.evaluate(partial_output, '') == 1.0
        assert schema_evaluator.evaluate(partial_output, '') == pytest.approx(2/3)

        # Bad output fails parse
        assert parse_evaluator.evaluate(bad_output, '') == 0.0

    def test_structured_output_evaluation(self) -> None:
        """Test structured output with shape and field checks."""
        shape_evaluator = OutputShapeEvaluator(
            starts_with="Analysis:",
            contains_all=["Findings:", "Conclusion:"]
        )
        field_evaluator = FieldCoverageEvaluator(
            required_patterns=["Risk Level:", "Recommendation:"]
        )

        good_output = """Analysis: Market Review
Findings: Strong growth potential
Risk Level: Medium
Recommendation: Hold
Conclusion: Positive outlook"""

        # Both evaluators should give positive scores
        assert shape_evaluator.evaluate(good_output, '') > 0.5
        assert field_evaluator.evaluate(good_output, '') == 1.0
