"""Staged evaluation pipeline with early exit.

This module provides multi-stage evaluation where cheap evaluators filter
candidates before expensive scorers run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .base import BaseEvaluator


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluator duck-typing.

    Any object with an evaluate method matching this signature
    can be used in the pipeline.
    """

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate a prediction."""
        ...


@dataclass(frozen=True)
class EvaluationStage:
    """A single stage in the evaluation pipeline.

    Attributes:
        name: Human-readable stage name.
        evaluator: The evaluator to run.
        weight: Weight for score aggregation (default 1.0).
        threshold: Minimum score to pass to next stage (default 0.0).
        is_filter: If True, failing threshold skips remaining stages.
    """

    name: str
    evaluator: Evaluator | BaseEvaluator
    weight: float = 1.0
    threshold: float = 0.0
    is_filter: bool = True

    def __post_init__(self) -> None:
        """Validate stage configuration."""
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {self.threshold}")


@dataclass
class StageResult:
    """Result from running a single stage.

    Attributes:
        stage_name: Name of the executed stage.
        score: Score from this stage (0.0-1.0).
        passed: Whether the score met the threshold.
        skipped: Whether this stage was skipped due to earlier failure.
    """

    stage_name: str
    score: float
    passed: bool
    skipped: bool = False


@dataclass
class PipelineResult:
    """Complete result from running the evaluation pipeline.

    Attributes:
        final_score: Aggregated score across all stages.
        stage_results: Results from each stage.
        stages_completed: Number of stages that actually ran.
        early_exit: Whether pipeline exited early due to filter failure.
        early_exit_stage: Name of stage that caused early exit, if any.
    """

    final_score: float
    stage_results: list[StageResult] = field(default_factory=list)
    stages_completed: int = 0
    early_exit: bool = False
    early_exit_stage: str | None = None

    def get_stage_score(self, stage_name: str) -> float | None:
        """Get score for a specific stage.

        Args:
            stage_name: Name of the stage.

        Returns:
            Score if stage was run, None if skipped or not found.
        """
        for result in self.stage_results:
            if result.stage_name == stage_name:
                return result.score if not result.skipped else None
        return None

    @property
    def passed_all_filters(self) -> bool:
        """Check if all filter stages were passed."""
        return not self.early_exit


class StagedPipelineEvaluator:
    """Multi-stage evaluation pipeline with early exit.

    Runs evaluators in sequence. If a filter stage fails (score below
    threshold), remaining stages are skipped and a reduced score is returned.

    This enables cheap pre-filtering: run fast checks first, only invoke
    expensive evaluators (like LLM judges) on candidates that pass.

    Example:
        >>> from promptfoundry.evaluators.proxy_metrics import JsonParseEvaluator
        >>> from promptfoundry.evaluators.accuracy import FuzzyMatchEvaluator
        >>>
        >>> pipeline = StagedPipelineEvaluator([
        ...     EvaluationStage("json_check", JsonParseEvaluator(), threshold=1.0),
        ...     EvaluationStage("quality", FuzzyMatchEvaluator(), weight=2.0),
        ... ])
        >>> result = pipeline.evaluate_detailed('{"valid": true}', '{"valid": true}')
        >>> result.passed_all_filters
        True

    Attributes:
        stages: Ordered list of evaluation stages.
        fail_score: Score returned when first filter fails (default 0.0).
        aggregation: How to combine stage scores ("weighted_mean" or "product").
    """

    def __init__(
        self,
        stages: list[EvaluationStage],
        fail_score: float = 0.0,
        aggregation: str = "weighted_mean",
    ) -> None:
        """Initialize the staged pipeline.

        Args:
            stages: Ordered list of evaluation stages (cheap first).
            fail_score: Score to return when a filter stage fails.
            aggregation: Score aggregation method.

        Raises:
            ValueError: If stages list is empty or aggregation is invalid.
        """
        if not stages:
            raise ValueError("At least one stage is required")
        if aggregation not in ("weighted_mean", "product"):
            raise ValueError(f"Unknown aggregation: {aggregation}")

        self._stages = list(stages)
        self._fail_score = fail_score
        self._aggregation = aggregation

    @property
    def stages(self) -> list[EvaluationStage]:
        """Get the list of stages."""
        return list(self._stages)

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate and return only the final score.

        Args:
            predicted: The LLM's output.
            expected: The expected output.
            metadata: Optional additional context.

        Returns:
            Aggregated score from all stages, or fail_score on early exit.
        """
        result = self.evaluate_detailed(predicted, expected, metadata)
        return result.final_score

    def evaluate_detailed(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Evaluate with full stage-by-stage results.

        Args:
            predicted: The LLM's output.
            expected: The expected output.
            metadata: Optional additional context.

        Returns:
            PipelineResult with scores from each stage.
        """
        stage_results: list[StageResult] = []
        stages_completed = 0
        early_exit = False
        early_exit_stage: str | None = None

        for stage in self._stages:
            # Run evaluator
            score = stage.evaluator.evaluate(predicted, expected, metadata)
            stages_completed += 1

            # Check threshold
            passed = score >= stage.threshold
            stage_results.append(StageResult(
                stage_name=stage.name,
                score=score,
                passed=passed,
                skipped=False,
            ))

            # Early exit on filter failure
            if not passed and stage.is_filter:
                early_exit = True
                early_exit_stage = stage.name
                break

        # Mark remaining stages as skipped
        for stage in self._stages[stages_completed:]:
            stage_results.append(StageResult(
                stage_name=stage.name,
                score=0.0,
                passed=False,
                skipped=True,
            ))

        # Calculate final score
        if early_exit:
            final_score = self._calculate_partial_score(stage_results)
        else:
            final_score = self._aggregate_scores(stage_results)

        return PipelineResult(
            final_score=final_score,
            stage_results=stage_results,
            stages_completed=stages_completed,
            early_exit=early_exit,
            early_exit_stage=early_exit_stage,
        )

    def _aggregate_scores(self, results: list[StageResult]) -> float:
        """Aggregate scores from completed stages.

        Args:
            results: List of stage results.

        Returns:
            Aggregated score.
        """
        # Get non-skipped results with their weights
        scored_stages = [
            (r, stage)
            for r, stage in zip(results, self._stages, strict=False)
            if not r.skipped
        ]

        if not scored_stages:
            return self._fail_score

        if self._aggregation == "weighted_mean":
            total_weight = sum(stage.weight for _, stage in scored_stages)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(r.score * stage.weight for r, stage in scored_stages)
            return weighted_sum / total_weight

        elif self._aggregation == "product":
            # Product of scores, weighted by taking power
            product = 1.0
            for r, stage in scored_stages:
                # Avoid zero scores making everything zero - use small epsilon
                score = max(r.score, 1e-10)
                product *= score ** stage.weight
            # Normalize by total weight
            total_weight = sum(stage.weight for _, stage in scored_stages)
            if total_weight > 0:
                product = product ** (1.0 / total_weight)
            return float(product)

        return 0.0

    def _calculate_partial_score(self, results: list[StageResult]) -> float:
        """Calculate score when pipeline exited early.

        Applies a penalty based on how far through the pipeline we got.

        Args:
            results: All stage results (some may be skipped).

        Returns:
            Partial score with penalty.
        """
        completed = [r for r in results if not r.skipped]
        if not completed:
            return self._fail_score

        # Progress factor: what fraction of stages completed
        progress = len(completed) / len(results)

        # Score from completed stages
        completed_score = self._aggregate_scores(results)

        # Apply progress penalty
        return completed_score * progress

    def evaluate_batch(
        self,
        predictions: list[str],
        expected: list[str],
    ) -> list[float]:
        """Score multiple predictions.

        Args:
            predictions: List of LLM outputs.
            expected: List of expected outputs.

        Returns:
            List of scores.
        """
        if len(predictions) != len(expected):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(expected)} expected"
            )
        return [
            self.evaluate(pred, exp)
            for pred, exp in zip(predictions, expected, strict=True)
        ]

    def evaluate_batch_detailed(
        self,
        predictions: list[str],
        expected: list[str],
    ) -> list[PipelineResult]:
        """Score multiple predictions with full details.

        Args:
            predictions: List of LLM outputs.
            expected: List of expected outputs.

        Returns:
            List of PipelineResult objects.
        """
        if len(predictions) != len(expected):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(expected)} expected"
            )
        return [
            self.evaluate_detailed(pred, exp)
            for pred, exp in zip(predictions, expected, strict=True)
        ]

    def aggregate(self, scores: list[float]) -> float:
        """Aggregate scores into a single value.

        Args:
            scores: List of individual scores.

        Returns:
            Mean score.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_evaluator_info(self) -> dict[str, Any]:
        """Return information about this evaluator.

        Returns:
            Dictionary with pipeline metadata.
        """
        return {
            "name": "StagedPipelineEvaluator",
            "stages": [
                {
                    "name": stage.name,
                    "evaluator": type(stage.evaluator).__name__,
                    "weight": stage.weight,
                    "threshold": stage.threshold,
                    "is_filter": stage.is_filter,
                }
                for stage in self._stages
            ],
            "aggregation": self._aggregation,
            "fail_score": self._fail_score,
        }


class PipelineBuilder:
    """Fluent builder for creating evaluation pipelines.

    Example:
        >>> pipeline = (
        ...     PipelineBuilder()
        ...     .add_filter("json_valid", JsonParseEvaluator())
        ...     .add_scorer("quality", FuzzyMatchEvaluator(), weight=2.0)
        ...     .aggregate_with("weighted_mean")
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize empty builder."""
        self._stages: list[EvaluationStage] = []
        self._fail_score: float = 0.0
        self._aggregation: str = "weighted_mean"

    def add_filter(
        self,
        name: str,
        evaluator: Evaluator | BaseEvaluator,
        threshold: float = 0.5,
        weight: float = 1.0,
    ) -> PipelineBuilder:
        """Add a filter stage that can cause early exit.

        Args:
            name: Stage name.
            evaluator: Evaluator to use.
            threshold: Minimum score to pass (default 0.5).
            weight: Weight for aggregation (default 1.0).

        Returns:
            Self for chaining.
        """
        self._stages.append(EvaluationStage(
            name=name,
            evaluator=evaluator,
            weight=weight,
            threshold=threshold,
            is_filter=True,
        ))
        return self

    def add_scorer(
        self,
        name: str,
        evaluator: Evaluator | BaseEvaluator,
        weight: float = 1.0,
    ) -> PipelineBuilder:
        """Add a scorer stage (no early exit).

        Args:
            name: Stage name.
            evaluator: Evaluator to use.
            weight: Weight for aggregation (default 1.0).

        Returns:
            Self for chaining.
        """
        self._stages.append(EvaluationStage(
            name=name,
            evaluator=evaluator,
            weight=weight,
            threshold=0.0,
            is_filter=False,
        ))
        return self

    def with_fail_score(self, score: float) -> PipelineBuilder:
        """Set the score returned on filter failure.

        Args:
            score: Fail score (default 0.0).

        Returns:
            Self for chaining.
        """
        self._fail_score = score
        return self

    def aggregate_with(self, method: str) -> PipelineBuilder:
        """Set aggregation method.

        Args:
            method: Either "weighted_mean" or "product".

        Returns:
            Self for chaining.
        """
        if method not in ("weighted_mean", "product"):
            raise ValueError(f"Unknown aggregation: {method}")
        self._aggregation = method
        return self

    def build(self) -> StagedPipelineEvaluator:
        """Build the pipeline.

        Returns:
            Configured StagedPipelineEvaluator.

        Raises:
            ValueError: If no stages have been added.
        """
        if not self._stages:
            raise ValueError("At least one stage is required")
        return StagedPipelineEvaluator(
            stages=self._stages,
            fail_score=self._fail_score,
            aggregation=self._aggregation,
        )


def create_cheap_to_expensive_pipeline(
    cheap_evaluators: list[tuple[str, Evaluator | BaseEvaluator, float]],
    expensive_evaluator: tuple[str, Evaluator | BaseEvaluator, float],
    cheap_threshold: float = 0.5,
) -> StagedPipelineEvaluator:
    """Create a pipeline that filters with cheap evaluators first.

    Common pattern: run fast checks (JSON parsing, length, keywords) first,
    then only invoke expensive evaluators (LLM judges) on passing candidates.

    Args:
        cheap_evaluators: List of (name, evaluator, weight) for cheap stages.
        expensive_evaluator: Tuple of (name, evaluator, weight) for final stage.
        cheap_threshold: Minimum score for each cheap stage.

    Returns:
        Configured pipeline with cheap filters → expensive scorer.

    Example:
        >>> pipeline = create_cheap_to_expensive_pipeline(
        ...     cheap_evaluators=[
        ...         ("json_check", JsonParseEvaluator(), 0.5),
        ...         ("length", LengthConstraintEvaluator(min_words=10), 0.3),
        ...     ],
        ...     expensive_evaluator=("llm_judge", llm_based_evaluator, 2.0),
        ...     cheap_threshold=0.7,
        ... )
    """
    stages: list[EvaluationStage] = []

    # Add cheap filter stages
    for name, evaluator, weight in cheap_evaluators:
        stages.append(EvaluationStage(
            name=name,
            evaluator=evaluator,
            weight=weight,
            threshold=cheap_threshold,
            is_filter=True,
        ))

    # Add expensive scorer (no threshold - always contributes)
    exp_name, exp_eval, exp_weight = expensive_evaluator
    stages.append(EvaluationStage(
        name=exp_name,
        evaluator=exp_eval,
        weight=exp_weight,
        threshold=0.0,
        is_filter=False,
    ))

    return StagedPipelineEvaluator(stages=stages)
