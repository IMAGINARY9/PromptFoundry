"""Cheap proxy metric evaluators.

This module provides evaluators that give useful signal without requiring
exact match evaluation. These are "cheap" because they use structural
validation rather than semantic comparison.

Use cases:
- JSON/structure validation when exact values vary
- Format compliance checking (headers, markers, sections)
- Required field/keyword coverage
- Length/token constraints
- Parser success validation

These evaluators are designed to provide partial credit and work well
in multi-stage evaluation pipelines where cheap pre-filtering can
eliminate clearly invalid candidates before expensive evaluation.
"""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any, Callable

from promptfoundry.evaluators.base import BaseEvaluator


class JsonParseEvaluator(BaseEvaluator):
    """Evaluator that checks if output is valid JSON.
    
    This is one of the cheapest evaluators - just checks parse success.
    Useful as a first-stage filter for JSON-producing tasks.
    
    Example:
        >>> evaluator = JsonParseEvaluator()
        >>> evaluator.evaluate('{"key": "value"}', '')  # Returns 1.0
        >>> evaluator.evaluate('not json', '')  # Returns 0.0
    """

    def __init__(
        self,
        extract_json: bool = True,
        strip_whitespace: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        """Initialize the JSON parse evaluator.
        
        Args:
            extract_json: If True, try to extract JSON from mixed content.
            strip_whitespace: Whether to strip whitespace before parsing.
            case_sensitive: Unused, kept for interface consistency.
        """
        super().__init__(case_sensitive=case_sensitive, strip_whitespace=strip_whitespace)
        self.extract_json = extract_json

    def _extract_json_block(self, text: str) -> str | None:
        """Try to extract a JSON block from text.
        
        Handles common patterns like:
        - Plain JSON
        - ```json ... ``` blocks
        - JSON surrounded by explanation text
        """
        text = text.strip()
        
        # Try direct parse first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Try to find JSON object or array
        for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            match = re.search(pattern, text)
            if match:
                try:
                    json.loads(match.group(0))
                    return match.group(0)
                except json.JSONDecodeError:
                    continue
        
        return None

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Check if predicted output is valid JSON.
        
        Args:
            predicted: The LLM's output.
            expected: Unused.
            metadata: Optional. Can contain 'json_type' to require object or array.
        
        Returns:
            1.0 if valid JSON, 0.0 otherwise.
        """
        if self.strip_whitespace:
            predicted = predicted.strip()
        
        if self.extract_json:
            json_text = self._extract_json_block(predicted)
            if json_text is None:
                return 0.0
            predicted = json_text
        
        try:
            parsed = json.loads(predicted)
            
            # Check type constraint if specified
            if metadata:
                required_type = metadata.get('json_type')
                if required_type == 'object' and not isinstance(parsed, dict):
                    return 0.0
                if required_type == 'array' and not isinstance(parsed, list):
                    return 0.0
            
            return 1.0
        except json.JSONDecodeError:
            return 0.0

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        info = super().get_evaluator_info()
        info["extract_json"] = self.extract_json
        return info


class JsonSchemaEvaluator(BaseEvaluator):
    """Evaluator that validates JSON against a schema.
    
    Provides partial credit based on how much of the schema is satisfied.
    
    Example:
        >>> evaluator = JsonSchemaEvaluator(
        ...     required_keys=["name", "age"],
        ...     key_types={"name": str, "age": int}
        ... )
        >>> evaluator.evaluate('{"name": "Alice", "age": 30}', '')  # Returns 1.0
        >>> evaluator.evaluate('{"name": "Alice"}', '')  # Returns 0.5 (partial)
    """

    def __init__(
        self,
        required_keys: list[str] | None = None,
        optional_keys: list[str] | None = None,
        key_types: dict[str, type] | None = None,
        extract_json: bool = True,
        strip_whitespace: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        """Initialize the JSON schema evaluator.
        
        Args:
            required_keys: Keys that must be present.
            optional_keys: Keys that are allowed but not required.
            key_types: Expected types for keys (e.g., {"age": int}).
            extract_json: If True, try to extract JSON from mixed content.
            strip_whitespace: Whether to strip whitespace before parsing.
            case_sensitive: Whether key names are case-sensitive.
        """
        super().__init__(case_sensitive=case_sensitive, strip_whitespace=strip_whitespace)
        self.required_keys = required_keys or []
        self.optional_keys = optional_keys or []
        self.key_types = key_types or {}
        self.extract_json = extract_json
        self._json_parser = JsonParseEvaluator(extract_json=extract_json)

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Validate JSON against schema with partial credit.
        
        Scoring:
        - 0.0 if not valid JSON
        - Partial credit for each required key present
        - Additional credit for correct types
        
        Args:
            predicted: The LLM's output.
            expected: Unused (schema defined in constructor).
            metadata: Optional. Can override required_keys.
        
        Returns:
            Score from 0.0 to 1.0 based on schema compliance.
        """
        # First check if it's valid JSON
        if self.strip_whitespace:
            predicted = predicted.strip()
        
        json_text = self._json_parser._extract_json_block(predicted) if self.extract_json else predicted
        if json_text is None:
            return 0.0
        
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return 0.0
        
        if not isinstance(parsed, dict):
            return 0.0
        
        # Calculate score based on schema compliance
        requirements = metadata.get('required_keys', self.required_keys) if metadata else self.required_keys
        type_requirements = metadata.get('key_types', self.key_types) if metadata else self.key_types
        
        if not requirements and not type_requirements:
            # No schema defined, just JSON validity
            return 1.0
        
        total_checks = len(requirements) + len(type_requirements)
        if total_checks == 0:
            return 1.0
        
        passed_checks = 0
        
        # Check required keys
        for key in requirements:
            check_key = key if self.case_sensitive else key.lower()
            parsed_keys = parsed.keys() if self.case_sensitive else [k.lower() for k in parsed.keys()]
            if check_key in ([k for k in parsed_keys] if self.case_sensitive else parsed_keys):
                passed_checks += 1
        
        # Check types
        for key, expected_type in type_requirements.items():
            if key in parsed:
                if isinstance(parsed[key], expected_type):
                    passed_checks += 1
        
        return passed_checks / total_checks

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        info = super().get_evaluator_info()
        info.update({
            "required_keys": self.required_keys,
            "optional_keys": self.optional_keys,
            "key_types": {k: v.__name__ for k, v in self.key_types.items()},
        })
        return info


class FieldCoverageEvaluator(BaseEvaluator):
    """Evaluator that checks for required fields or sections in output.
    
    Useful for tasks that should produce structured output with specific
    sections, headers, or labeled fields.
    
    Example:
        >>> evaluator = FieldCoverageEvaluator(
        ...     required_patterns=["Name:", "Age:", "Location:"]
        ... )
        >>> output = "Name: Alice\\nAge: 30\\nLocation: NYC"
        >>> evaluator.evaluate(output, '')  # Returns 1.0
    """

    def __init__(
        self,
        required_patterns: list[str] | None = None,
        use_regex: bool = False,
        partial_credit: bool = True,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize field coverage evaluator.
        
        Args:
            required_patterns: List of patterns/strings that should appear.
            use_regex: If True, treat patterns as regex.
            partial_credit: If True, return fraction of patterns found.
            case_sensitive: Whether patterns are case-sensitive.
            strip_whitespace: Whether to strip whitespace from output.
        """
        super().__init__(case_sensitive=case_sensitive, strip_whitespace=strip_whitespace)
        self.required_patterns = required_patterns or []
        self.use_regex = use_regex
        self.partial_credit = partial_credit
        
        # Pre-compile regex patterns
        self._compiled_patterns: list[re.Pattern[str]] = []
        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._compiled_patterns = [
                re.compile(p, flags) for p in self.required_patterns
            ]

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Check coverage of required patterns.
        
        Args:
            predicted: The LLM's output.
            expected: Unused (patterns defined in constructor).
            metadata: Optional. Can override required_patterns.
        
        Returns:
            1.0 if all patterns found, partial credit if enabled, else 0.0.
        """
        pred = self._preprocess(predicted)
        
        patterns = metadata.get('required_patterns', self.required_patterns) if metadata else self.required_patterns
        if not patterns:
            return 1.0
        
        found = 0
        for i, pattern in enumerate(patterns):
            if self.use_regex and i < len(self._compiled_patterns):
                if self._compiled_patterns[i].search(pred):
                    found += 1
            elif self.use_regex:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                if re.search(pattern, pred, flags):
                    found += 1
            else:
                check_pattern = pattern if self.case_sensitive else pattern.lower()
                check_pred = pred if self.case_sensitive else pred.lower()
                if check_pattern in check_pred:
                    found += 1
        
        if self.partial_credit:
            return found / len(patterns)
        else:
            return 1.0 if found == len(patterns) else 0.0

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        info = super().get_evaluator_info()
        info.update({
            "required_patterns": self.required_patterns,
            "use_regex": self.use_regex,
            "partial_credit": self.partial_credit,
        })
        return info


class KeywordPresenceEvaluator(BaseEvaluator):
    """Evaluator that checks for presence of required keywords.
    
    Simpler than FieldCoverageEvaluator - just checks word presence.
    Supports weighted keywords for importance-based scoring.
    
    Example:
        >>> evaluator = KeywordPresenceEvaluator(
        ...     required_keywords=["positive", "sentiment"],
        ...     weights={"positive": 2.0}  # More important
        ... )
    """

    def __init__(
        self,
        required_keywords: list[str] | None = None,
        forbidden_keywords: list[str] | None = None,
        weights: dict[str, float] | None = None,
        word_boundary: bool = True,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize keyword presence evaluator.
        
        Args:
            required_keywords: Keywords that should appear.
            forbidden_keywords: Keywords that should NOT appear (deduct points).
            weights: Weight multipliers for keywords (default 1.0).
            word_boundary: If True, match whole words only.
            case_sensitive: Whether matching is case-sensitive.
            strip_whitespace: Whether to strip whitespace from output.
        """
        super().__init__(case_sensitive=case_sensitive, strip_whitespace=strip_whitespace)
        self.required_keywords = required_keywords or []
        self.forbidden_keywords = forbidden_keywords or []
        self.weights = weights or {}
        self.word_boundary = word_boundary

    def _check_keyword(self, text: str, keyword: str) -> bool:
        """Check if keyword is present in text."""
        if self.word_boundary:
            pattern = rf'\b{re.escape(keyword)}\b'
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return bool(re.search(pattern, text, flags))
        else:
            if self.case_sensitive:
                return keyword in text
            return keyword.lower() in text.lower()

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Check keyword presence with weighted scoring.
        
        Args:
            predicted: The LLM's output.
            expected: Unused.
            metadata: Optional. Can contain additional_keywords.
        
        Returns:
            Weighted score from 0.0 to 1.0.
        """
        pred = self._preprocess(predicted)
        
        total_weight = 0.0
        achieved_weight = 0.0
        
        # Check required keywords
        for keyword in self.required_keywords:
            weight = self.weights.get(keyword, 1.0)
            total_weight += weight
            if self._check_keyword(pred, keyword):
                achieved_weight += weight
        
        # Check forbidden keywords (deduct points)
        for keyword in self.forbidden_keywords:
            weight = self.weights.get(keyword, 1.0)
            if self._check_keyword(pred, keyword):
                achieved_weight -= weight
        
        if total_weight == 0:
            return 1.0
        
        return max(0.0, min(1.0, achieved_weight / total_weight))

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        info = super().get_evaluator_info()
        info.update({
            "required_keywords": self.required_keywords,
            "forbidden_keywords": self.forbidden_keywords,
            "weights": self.weights,
            "word_boundary": self.word_boundary,
        })
        return info


class LengthConstraintEvaluator(BaseEvaluator):
    """Evaluator that scores based on output length constraints.
    
    Useful for:
    - Penalizing too-short responses (probably incomplete)
    - Penalizing too-long responses (probably verbose/rambling)
    - Rewarding outputs in expected length range
    
    Example:
        >>> evaluator = LengthConstraintEvaluator(
        ...     min_chars=50, max_chars=500,
        ...     penalty_mode="linear"
        ... )
    """

    def __init__(
        self,
        min_chars: int | None = None,
        max_chars: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        min_lines: int | None = None,
        max_lines: int | None = None,
        penalty_mode: str = "linear",
        strip_whitespace: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        """Initialize length constraint evaluator.
        
        Args:
            min_chars: Minimum character count.
            max_chars: Maximum character count.
            min_words: Minimum word count.
            max_words: Maximum word count.
            min_lines: Minimum line count.
            max_lines: Maximum line count.
            penalty_mode: "linear" (gradual), "binary" (all or nothing), "soft" (gentle curve).
            strip_whitespace: Whether to strip whitespace.
            case_sensitive: Unused, kept for interface consistency.
        """
        super().__init__(case_sensitive=case_sensitive, strip_whitespace=strip_whitespace)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.max_words = max_words
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.penalty_mode = penalty_mode

    def _score_range(self, value: int, min_val: int | None, max_val: int | None) -> float:
        """Calculate score for a value within range constraints."""
        if min_val is None and max_val is None:
            return 1.0
        
        if min_val is not None and value < min_val:
            if self.penalty_mode == "binary":
                return 0.0
            elif self.penalty_mode == "linear":
                return float(max(0.0, value / min_val))
            else:  # soft
                return float(max(0.0, (value / min_val) ** 0.5))
        
        if max_val is not None and value > max_val:
            if self.penalty_mode == "binary":
                return 0.0
            elif self.penalty_mode == "linear":
                # Score decreases proportionally as we exceed max
                # At 2x max, score = 0.5; at 3x max, score = 0.33
                return float(max(0.0, max_val / value))
            else:  # soft
                overflow_ratio = value / max_val
                return float(max(0.0, 1.0 / overflow_ratio))
        
        return 1.0

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Score output based on length constraints.
        
        Args:
            predicted: The LLM's output.
            expected: Unused.
            metadata: Optional. Can override length constraints.
        
        Returns:
            Score from 0.0 to 1.0 based on length compliance.
        """
        if self.strip_whitespace:
            predicted = predicted.strip()
        
        scores = []
        
        # Character constraints
        if self.min_chars is not None or self.max_chars is not None:
            scores.append(self._score_range(len(predicted), self.min_chars, self.max_chars))
        
        # Word constraints
        if self.min_words is not None or self.max_words is not None:
            word_count = len(predicted.split())
            scores.append(self._score_range(word_count, self.min_words, self.max_words))
        
        # Line constraints
        if self.min_lines is not None or self.max_lines is not None:
            line_count = len(predicted.splitlines()) if predicted else 0
            scores.append(self._score_range(line_count, self.min_lines, self.max_lines))
        
        if not scores:
            return 1.0
        
        # Return minimum score (most restrictive constraint)
        return min(scores)

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        info = super().get_evaluator_info()
        info.update({
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "min_words": self.min_words,
            "max_words": self.max_words,
            "min_lines": self.min_lines,
            "max_lines": self.max_lines,
            "penalty_mode": self.penalty_mode,
        })
        return info


class OutputShapeEvaluator(BaseEvaluator):
    """Evaluator that checks structural shape of output.
    
    Validates output structure without checking exact content.
    Useful for checking format compliance like:
    - Starts with specific text
    - Ends with specific text
    - Contains markers/delimiters
    - Has expected section structure
    
    Example:
        >>> evaluator = OutputShapeEvaluator(
        ...     starts_with="Answer:",
        ...     ends_with=".",
        ...     contains_all=["Step 1:", "Step 2:"]
        ... )
    """

    def __init__(
        self,
        starts_with: str | None = None,
        ends_with: str | None = None,
        contains_all: list[str] | None = None,
        contains_any: list[str] | None = None,
        not_contains: list[str] | None = None,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize output shape evaluator.
        
        Args:
            starts_with: Required prefix.
            ends_with: Required suffix.
            contains_all: All of these must appear.
            contains_any: At least one of these must appear.
            not_contains: None of these should appear.
            case_sensitive: Whether checks are case-sensitive.
            strip_whitespace: Whether to strip whitespace.
        """
        super().__init__(case_sensitive=case_sensitive, strip_whitespace=strip_whitespace)
        self.starts_with = starts_with
        self.ends_with = ends_with
        self.contains_all = contains_all or []
        self.contains_any = contains_any or []
        self.not_contains = not_contains or []

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Check output shape compliance.
        
        Args:
            predicted: The LLM's output.
            expected: Unused.
            metadata: Optional. Can override shape constraints.
        
        Returns:
            Score from 0.0 to 1.0 based on shape compliance.
        """
        pred = self._preprocess(predicted)
        
        checks = []
        
        # Check prefix
        if self.starts_with:
            prefix = self.starts_with if self.case_sensitive else self.starts_with.lower()
            check_pred = pred if self.case_sensitive else pred.lower()
            checks.append(check_pred.startswith(prefix))
        
        # Check suffix
        if self.ends_with:
            suffix = self.ends_with if self.case_sensitive else self.ends_with.lower()
            check_pred = pred if self.case_sensitive else pred.lower()
            checks.append(check_pred.endswith(suffix))
        
        # Check contains_all
        for pattern in self.contains_all:
            check_pattern = pattern if self.case_sensitive else pattern.lower()
            check_pred = pred if self.case_sensitive else pred.lower()
            checks.append(check_pattern in check_pred)
        
        # Check contains_any
        if self.contains_any:
            found_any = False
            for pattern in self.contains_any:
                check_pattern = pattern if self.case_sensitive else pattern.lower()
                check_pred = pred if self.case_sensitive else pred.lower()
                if check_pattern in check_pred:
                    found_any = True
                    break
            checks.append(found_any)
        
        # Check not_contains
        for pattern in self.not_contains:
            check_pattern = pattern if self.case_sensitive else pattern.lower()
            check_pred = pred if self.case_sensitive else pred.lower()
            checks.append(check_pattern not in check_pred)
        
        if not checks:
            return 1.0
        
        # Return fraction of checks passed
        return sum(checks) / len(checks)

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return evaluator information."""
        info = super().get_evaluator_info()
        info.update({
            "starts_with": self.starts_with,
            "ends_with": self.ends_with,
            "contains_all": self.contains_all,
            "contains_any": self.contains_any,
            "not_contains": self.not_contains,
        })
        return info
