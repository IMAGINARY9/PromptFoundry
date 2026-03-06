"""Evaluators (objective functions) for PromptFoundry."""

from promptfoundry.evaluators.accuracy import ExactMatchEvaluator, FuzzyMatchEvaluator
from promptfoundry.evaluators.base import BaseEvaluator
from promptfoundry.evaluators.custom import CompositeEvaluator, CustomFunctionEvaluator
from promptfoundry.evaluators.format import ContainsEvaluator, RegexEvaluator
from promptfoundry.evaluators.pipeline import (
    EvaluationStage,
    PipelineBuilder,
    PipelineResult,
    StagedPipelineEvaluator,
    StageResult,
    create_cheap_to_expensive_pipeline,
)
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
    # Staged pipeline (MVP 2)
    "EvaluationStage",
    "PipelineBuilder",
    "PipelineResult",
    "StagedPipelineEvaluator",
    "StageResult",
    "create_cheap_to_expensive_pipeline",
]
