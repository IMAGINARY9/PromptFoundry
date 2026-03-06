"""Evaluators (objective functions) for PromptFoundry."""

from promptfoundry.evaluators.accuracy import ExactMatchEvaluator, FuzzyMatchEvaluator
from promptfoundry.evaluators.base import BaseEvaluator
from promptfoundry.evaluators.custom import CompositeEvaluator, CustomFunctionEvaluator
from promptfoundry.evaluators.format import ContainsEvaluator, RegexEvaluator
from promptfoundry.evaluators.proxy_metrics import (
    FieldCoverageEvaluator,
    JsonParseEvaluator,
    JsonSchemaEvaluator,
    KeywordPresenceEvaluator,
    LengthConstraintEvaluator,
    OutputShapeEvaluator,
)

__all__ = [
    # Base
    "BaseEvaluator",
    # Accuracy evaluators
    "ExactMatchEvaluator",
    "FuzzyMatchEvaluator",
    # Format evaluators
    "ContainsEvaluator",
    "RegexEvaluator",
    # Custom evaluators
    "CompositeEvaluator",
    "CustomFunctionEvaluator",
    # Cheap proxy metrics (MVP 2)
    "FieldCoverageEvaluator",
    "JsonParseEvaluator",
    "JsonSchemaEvaluator",
    "KeywordPresenceEvaluator",
    "LengthConstraintEvaluator",
    "OutputShapeEvaluator",
]
