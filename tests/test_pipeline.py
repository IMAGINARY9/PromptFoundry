"""Tests for staged evaluation pipeline.

Tests the StagedPipelineEvaluator, PipelineBuilder, and related utilities.
"""

from __future__ import annotations

from typing import Any

import pytest

from promptfoundry.evaluators.pipeline import (
    EvaluationStage,
    PipelineBuilder,
    PipelineResult,
    StagedPipelineEvaluator,
    StageResult,
    create_cheap_to_expensive_pipeline,
)


# =============================================================================
# Test fixtures - Simple evaluators for testing
# =============================================================================


class AlwaysPassEvaluator:
    """Evaluator that always returns 1.0."""

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        return 1.0


class AlwaysFailEvaluator:
    """Evaluator that always returns 0.0."""

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        return 0.0


class FixedScoreEvaluator:
    """Evaluator that returns a fixed score."""

    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        return self.score


class LengthBasedEvaluator:
    """Evaluator that scores based on prediction length match."""

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        if len(expected) == 0:
            return 1.0 if len(predicted) == 0 else 0.0
        ratio = len(predicted) / len(expected)
        return max(0.0, 1.0 - abs(1.0 - ratio))


class CallCountEvaluator:
    """Evaluator that counts how many times it's called."""

    def __init__(self, score: float = 1.0) -> None:
        self.call_count = 0
        self.score = score

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        self.call_count += 1
        return self.score


# =============================================================================
# EvaluationStage Tests
# =============================================================================


class TestEvaluationStage:
    """Tests for EvaluationStage dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a stage with defaults."""
        stage = EvaluationStage(
            name="test",
            evaluator=AlwaysPassEvaluator(),
        )
        assert stage.name == "test"
        assert stage.weight == 1.0
        assert stage.threshold == 0.0
        assert stage.is_filter is True

    def test_custom_values(self) -> None:
        """Test creating a stage with custom values."""
        stage = EvaluationStage(
            name="quality",
            evaluator=AlwaysPassEvaluator(),
            weight=2.5,
            threshold=0.7,
            is_filter=False,
        )
        assert stage.weight == 2.5
        assert stage.threshold == 0.7
        assert stage.is_filter is False

    def test_invalid_weight_raises(self) -> None:
        """Test that negative weight raises error."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            EvaluationStage(
                name="test",
                evaluator=AlwaysPassEvaluator(),
                weight=-1.0,
            )

    def test_invalid_threshold_raises(self) -> None:
        """Test that threshold out of bounds raises error."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            EvaluationStage(
                name="test",
                evaluator=AlwaysPassEvaluator(),
                threshold=1.5,
            )

        with pytest.raises(ValueError, match="Threshold must be in"):
            EvaluationStage(
                name="test",
                evaluator=AlwaysPassEvaluator(),
                threshold=-0.1,
            )

    def test_frozen(self) -> None:
        """Test that stage is immutable."""
        stage = EvaluationStage(name="test", evaluator=AlwaysPassEvaluator())
        with pytest.raises(Exception):  # FrozenInstanceError
            stage.name = "changed"  # type: ignore


# =============================================================================
# StageResult and PipelineResult Tests
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a stage result."""
        result = StageResult(
            stage_name="test",
            score=0.8,
            passed=True,
        )
        assert result.stage_name == "test"
        assert result.score == 0.8
        assert result.passed is True
        assert result.skipped is False

    def test_skipped_result(self) -> None:
        """Test creating a skipped result."""
        result = StageResult(
            stage_name="expensive",
            score=0.0,
            passed=False,
            skipped=True,
        )
        assert result.skipped is True


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a pipeline result."""
        result = PipelineResult(
            final_score=0.85,
            stages_completed=3,
        )
        assert result.final_score == 0.85
        assert result.stages_completed == 3
        assert result.early_exit is False
        assert result.early_exit_stage is None

    def test_with_stage_results(self) -> None:
        """Test pipeline result with stage details."""
        stages = [
            StageResult("s1", 1.0, True),
            StageResult("s2", 0.8, True),
        ]
        result = PipelineResult(
            final_score=0.9,
            stage_results=stages,
            stages_completed=2,
        )
        assert len(result.stage_results) == 2
        assert result.get_stage_score("s1") == 1.0
        assert result.get_stage_score("s2") == 0.8

    def test_get_stage_score_not_found(self) -> None:
        """Test getting score for non-existent stage."""
        result = PipelineResult(
            final_score=0.5,
            stage_results=[StageResult("s1", 0.5, True)],
            stages_completed=1,
        )
        assert result.get_stage_score("nonexistent") is None

    def test_get_stage_score_skipped(self) -> None:
        """Test getting score for skipped stage returns None."""
        result = PipelineResult(
            final_score=0.5,
            stage_results=[StageResult("s1", 0.0, False, skipped=True)],
            stages_completed=0,
        )
        assert result.get_stage_score("s1") is None

    def test_passed_all_filters(self) -> None:
        """Test passed_all_filters property."""
        passed = PipelineResult(final_score=1.0, early_exit=False)
        assert passed.passed_all_filters is True

        failed = PipelineResult(final_score=0.0, early_exit=True, early_exit_stage="s1")
        assert failed.passed_all_filters is False


# =============================================================================
# StagedPipelineEvaluator Tests
# =============================================================================


class TestStagedPipelineEvaluator:
    """Tests for the main pipeline evaluator."""

    def test_empty_stages_raises(self) -> None:
        """Test that empty stages list raises error."""
        with pytest.raises(ValueError, match="At least one stage"):
            StagedPipelineEvaluator([])

    def test_invalid_aggregation_raises(self) -> None:
        """Test that invalid aggregation raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            StagedPipelineEvaluator(
                [EvaluationStage("test", AlwaysPassEvaluator())],
                aggregation="invalid",
            )

    def test_single_stage_pass(self) -> None:
        """Test pipeline with single passing stage."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", AlwaysPassEvaluator()),
        ])
        score = pipeline.evaluate("pred", "exp")
        assert score == 1.0

    def test_single_stage_fail(self) -> None:
        """Test pipeline with single failing stage."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", AlwaysFailEvaluator(), threshold=0.5),
        ])
        score = pipeline.evaluate("pred", "exp")
        # Failed the filter, should get partial score
        assert score == 0.0

    def test_multiple_stages_all_pass(self) -> None:
        """Test pipeline where all stages pass."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.8), threshold=0.5),
            EvaluationStage("s2", FixedScoreEvaluator(1.0), threshold=0.5),
        ])
        score = pipeline.evaluate("pred", "exp")
        # Weighted mean: (0.8 * 1 + 1.0 * 1) / 2 = 0.9
        assert score == pytest.approx(0.9)

    def test_early_exit_on_filter_failure(self) -> None:
        """Test that pipeline exits early when filter fails."""
        expensive = CallCountEvaluator()
        
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("cheap", AlwaysFailEvaluator(), threshold=0.5),
            EvaluationStage("expensive", expensive),
        ])
        
        pipeline.evaluate("pred", "exp")
        
        # Expensive evaluator should NOT be called
        assert expensive.call_count == 0

    def test_non_filter_stage_continues(self) -> None:
        """Test that non-filter stages don't cause early exit."""
        final = CallCountEvaluator(score=1.0)
        
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.3), threshold=0.5, is_filter=False),
            EvaluationStage("s2", final),
        ])
        
        pipeline.evaluate("pred", "exp")
        
        # Final stage should still run despite s1 being below threshold
        assert final.call_count == 1

    def test_evaluate_detailed_results(self) -> None:
        """Test getting detailed results from evaluation."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.9), threshold=0.5),
            EvaluationStage("s2", FixedScoreEvaluator(0.7), threshold=0.5),
        ])
        
        result = pipeline.evaluate_detailed("pred", "exp")
        
        assert result.stages_completed == 2
        assert result.early_exit is False
        assert len(result.stage_results) == 2
        assert result.stage_results[0].stage_name == "s1"
        assert result.stage_results[0].score == 0.9
        assert result.stage_results[1].score == 0.7

    def test_early_exit_marks_stages_skipped(self) -> None:
        """Test that early exit marks remaining stages as skipped."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.2), threshold=0.5),
            EvaluationStage("s2", AlwaysPassEvaluator()),
            EvaluationStage("s3", AlwaysPassEvaluator()),
        ])
        
        result = pipeline.evaluate_detailed("pred", "exp")
        
        assert result.early_exit is True
        assert result.early_exit_stage == "s1"
        assert result.stages_completed == 1
        assert result.stage_results[1].skipped is True
        assert result.stage_results[2].skipped is True

    def test_weighted_aggregation(self) -> None:
        """Test weighted mean aggregation."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.6), weight=1.0),
            EvaluationStage("s2", FixedScoreEvaluator(0.8), weight=3.0),
        ])
        
        score = pipeline.evaluate("pred", "exp")
        # Weighted mean: (0.6 * 1 + 0.8 * 3) / 4 = 3.0 / 4 = 0.75
        assert score == pytest.approx(0.75)

    def test_product_aggregation(self) -> None:
        """Test product aggregation."""
        pipeline = StagedPipelineEvaluator(
            [
                EvaluationStage("s1", FixedScoreEvaluator(0.8), weight=1.0),
                EvaluationStage("s2", FixedScoreEvaluator(0.8), weight=1.0),
            ],
            aggregation="product",
        )
        
        score = pipeline.evaluate("pred", "exp")
        # Product: (0.8 * 0.8) ^ (1/2) = 0.64 ^ 0.5 = 0.8
        assert score == pytest.approx(0.8)

    def test_evaluate_batch(self) -> None:
        """Test batch evaluation."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", LengthBasedEvaluator()),
        ])
        
        scores = pipeline.evaluate_batch(
            predictions=["hello", "hi", "hello world"],
            expected=["hello", "hello", "hello"],
        )
        
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)  # exact match
        assert scores[1] < 1.0  # shorter
        assert scores[2] < 1.0  # longer

    def test_evaluate_batch_length_mismatch(self) -> None:
        """Test batch evaluation with mismatched lengths."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", AlwaysPassEvaluator()),
        ])
        
        with pytest.raises(ValueError, match="Length mismatch"):
            pipeline.evaluate_batch(["a", "b"], ["a"])

    def test_evaluate_batch_detailed(self) -> None:
        """Test batch evaluation with detailed results."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", FixedScoreEvaluator(0.7)),
        ])
        
        results = pipeline.evaluate_batch_detailed(["a", "b"], ["a", "b"])
        
        assert len(results) == 2
        assert all(isinstance(r, PipelineResult) for r in results)
        assert all(r.final_score == 0.7 for r in results)

    def test_aggregate_scores(self) -> None:
        """Test aggregating multiple pipeline run scores."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", AlwaysPassEvaluator()),
        ])
        
        agg = pipeline.aggregate([0.8, 0.6, 1.0])
        assert agg == pytest.approx(0.8)

    def test_aggregate_empty(self) -> None:
        """Test aggregating empty list."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", AlwaysPassEvaluator()),
        ])
        
        assert pipeline.aggregate([]) == 0.0

    def test_get_evaluator_info(self) -> None:
        """Test getting evaluator info."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(1.0), weight=2.0, threshold=0.5),
            EvaluationStage("s2", AlwaysPassEvaluator()),
        ])
        
        info = pipeline.get_evaluator_info()
        
        assert info["name"] == "StagedPipelineEvaluator"
        assert info["aggregation"] == "weighted_mean"
        assert len(info["stages"]) == 2
        assert info["stages"][0]["name"] == "s1"
        assert info["stages"][0]["weight"] == 2.0
        assert info["stages"][0]["threshold"] == 0.5

    def test_stages_property(self) -> None:
        """Test the stages property returns a copy."""
        original_stages = [EvaluationStage("test", AlwaysPassEvaluator())]
        pipeline = StagedPipelineEvaluator(original_stages)
        
        stages = pipeline.stages
        stages.append(EvaluationStage("new", AlwaysPassEvaluator()))
        
        # Original should be unchanged
        assert len(pipeline.stages) == 1

    def test_partial_score_on_early_exit(self) -> None:
        """Test that partial score reflects progress."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.4), threshold=0.5),  # fails
            EvaluationStage("s2", AlwaysPassEvaluator()),
            EvaluationStage("s3", AlwaysPassEvaluator()),
        ])
        
        result = pipeline.evaluate_detailed("pred", "exp")
        
        # Only 1/3 stages completed, so progress = 0.333...
        # Score of completed = 0.4
        # Partial = 0.4 * (1/3) ≈ 0.133
        assert result.final_score == pytest.approx(0.4 * (1/3), rel=0.01)


# =============================================================================
# PipelineBuilder Tests
# =============================================================================


class TestPipelineBuilder:
    """Tests for the fluent pipeline builder."""

    def test_basic_build(self) -> None:
        """Test building a simple pipeline."""
        pipeline = (
            PipelineBuilder()
            .add_filter("f1", AlwaysPassEvaluator())
            .build()
        )
        
        assert len(pipeline.stages) == 1

    def test_add_filter(self) -> None:
        """Test adding a filter stage."""
        pipeline = (
            PipelineBuilder()
            .add_filter("f1", FixedScoreEvaluator(0.8), threshold=0.7, weight=2.0)
            .build()
        )
        
        stage = pipeline.stages[0]
        assert stage.name == "f1"
        assert stage.threshold == 0.7
        assert stage.weight == 2.0
        assert stage.is_filter is True

    def test_add_scorer(self) -> None:
        """Test adding a scorer stage."""
        pipeline = (
            PipelineBuilder()
            .add_scorer("s1", FixedScoreEvaluator(0.5), weight=3.0)
            .build()
        )
        
        stage = pipeline.stages[0]
        assert stage.name == "s1"
        assert stage.threshold == 0.0
        assert stage.is_filter is False
        assert stage.weight == 3.0

    def test_with_fail_score(self) -> None:
        """Test setting fail score."""
        pipeline = (
            PipelineBuilder()
            .add_filter("f1", AlwaysFailEvaluator(), threshold=0.5)
            .with_fail_score(0.1)
            .build()
        )
        
        # Check via evaluator info
        info = pipeline.get_evaluator_info()
        assert info["fail_score"] == 0.1

    def test_aggregate_with(self) -> None:
        """Test setting aggregation method."""
        pipeline = (
            PipelineBuilder()
            .add_scorer("s1", AlwaysPassEvaluator())
            .aggregate_with("product")
            .build()
        )
        
        info = pipeline.get_evaluator_info()
        assert info["aggregation"] == "product"

    def test_invalid_aggregation(self) -> None:
        """Test invalid aggregation raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            (
                PipelineBuilder()
                .add_scorer("s1", AlwaysPassEvaluator())
                .aggregate_with("invalid")
            )

    def test_build_without_stages_raises(self) -> None:
        """Test building without stages raises error."""
        with pytest.raises(ValueError, match="At least one stage"):
            PipelineBuilder().build()

    def test_chaining(self) -> None:
        """Test fluent chaining builds correctly."""
        pipeline = (
            PipelineBuilder()
            .add_filter("json_check", FixedScoreEvaluator(1.0), threshold=1.0)
            .add_filter("length_check", FixedScoreEvaluator(0.8), threshold=0.5)
            .add_scorer("quality", FixedScoreEvaluator(0.9), weight=2.0)
            .with_fail_score(0.0)
            .aggregate_with("weighted_mean")
            .build()
        )
        
        assert len(pipeline.stages) == 3
        assert pipeline.stages[0].name == "json_check"
        assert pipeline.stages[1].name == "length_check"
        assert pipeline.stages[2].name == "quality"


# =============================================================================
# create_cheap_to_expensive_pipeline Tests
# =============================================================================


class TestCreateCheapToExpensivePipeline:
    """Tests for the convenience factory function."""

    def test_basic_creation(self) -> None:
        """Test creating a cheap-to-expensive pipeline."""
        cheap1 = FixedScoreEvaluator(1.0)
        cheap2 = FixedScoreEvaluator(0.8)
        expensive = FixedScoreEvaluator(0.9)
        
        pipeline = create_cheap_to_expensive_pipeline(
            cheap_evaluators=[
                ("json", cheap1, 0.5),
                ("length", cheap2, 0.3),
            ],
            expensive_evaluator=("llm", expensive, 2.0),
            cheap_threshold=0.5,
        )
        
        assert len(pipeline.stages) == 3
        
        # Cheap stages are filters
        assert pipeline.stages[0].is_filter is True
        assert pipeline.stages[0].threshold == 0.5
        assert pipeline.stages[1].is_filter is True
        
        # Expensive stage is not a filter
        assert pipeline.stages[2].is_filter is False
        assert pipeline.stages[2].weight == 2.0

    def test_cheap_filter_prevents_expensive(self) -> None:
        """Test that cheap filter failure skips expensive evaluator."""
        expensive = CallCountEvaluator()
        
        pipeline = create_cheap_to_expensive_pipeline(
            cheap_evaluators=[
                ("cheap", AlwaysFailEvaluator(), 1.0),
            ],
            expensive_evaluator=("expensive", expensive, 1.0),
            cheap_threshold=0.5,
        )
        
        pipeline.evaluate("pred", "exp")
        
        assert expensive.call_count == 0

    def test_passes_all_cheap_runs_expensive(self) -> None:
        """Test that passing all cheap stages runs expensive evaluator."""
        expensive = CallCountEvaluator(score=0.9)
        
        pipeline = create_cheap_to_expensive_pipeline(
            cheap_evaluators=[
                ("c1", AlwaysPassEvaluator(), 0.5),
                ("c2", AlwaysPassEvaluator(), 0.5),
            ],
            expensive_evaluator=("expensive", expensive, 2.0),
            cheap_threshold=0.5,
        )
        
        result = pipeline.evaluate_detailed("pred", "exp")
        
        assert expensive.call_count == 1
        assert result.passed_all_filters is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for complete pipeline scenarios."""

    def test_json_then_quality_pipeline(self) -> None:
        """Test realistic JSON validation → quality pipeline."""
        # Simulating: JSON check → content quality
        json_like = FixedScoreEvaluator(1.0)  # Pretend it's valid JSON
        quality = FixedScoreEvaluator(0.85)
        
        pipeline = (
            PipelineBuilder()
            .add_filter("json_valid", json_like, threshold=1.0)
            .add_scorer("quality", quality, weight=2.0)
            .build()
        )
        
        result = pipeline.evaluate_detailed('{"key": "value"}', '{"key": "value"}')
        
        assert result.passed_all_filters is True
        assert result.stages_completed == 2

    def test_multiple_cheap_filters(self) -> None:
        """Test pipeline with multiple cheap filters."""
        pipeline = (
            PipelineBuilder()
            .add_filter("not_empty", FixedScoreEvaluator(1.0), threshold=0.1)
            .add_filter("has_length", FixedScoreEvaluator(0.8), threshold=0.5)
            .add_filter("format_ok", FixedScoreEvaluator(0.9), threshold=0.5)
            .add_scorer("final_quality", FixedScoreEvaluator(0.95), weight=3.0)
            .build()
        )
        
        result = pipeline.evaluate_detailed("test output", "expected")
        
        assert result.stages_completed == 4
        assert result.passed_all_filters is True
        # All scores contribute
        assert result.final_score > 0

    def test_early_fail_cost_savings(self) -> None:
        """Test that early failure saves expensive evaluator calls."""
        call_counts: dict[str, int] = {"cheap": 0, "medium": 0, "expensive": 0}
        
        class TrackingEvaluator:
            def __init__(self, name: str, score: float):
                self.name = name
                self.score = score
            
            def evaluate(
                self,
                predicted: str,
                expected: str,
                metadata: dict[str, Any] | None = None,
            ) -> float:
                call_counts[self.name] += 1
                return self.score
        
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("cheap", TrackingEvaluator("cheap", 0.2), threshold=0.5),
            EvaluationStage("medium", TrackingEvaluator("medium", 1.0)),
            EvaluationStage("expensive", TrackingEvaluator("expensive", 1.0)),
        ])
        
        # Run on 10 candidates
        for _ in range(10):
            pipeline.evaluate("bad output", "expected")
        
        # Only cheap evaluator should run
        assert call_counts["cheap"] == 10
        assert call_counts["medium"] == 0
        assert call_counts["expensive"] == 0

    def test_metadata_passed_through(self) -> None:
        """Test that metadata is passed to all evaluators."""
        received_metadata: list[dict[str, Any]] = []
        
        class MetadataCapture:
            def evaluate(
                self,
                predicted: str,
                expected: str,
                metadata: dict[str, Any] | None = None,
            ) -> float:
                if metadata:
                    received_metadata.append(metadata)
                return 1.0
        
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", MetadataCapture()),
            EvaluationStage("s2", MetadataCapture()),
        ])
        
        pipeline.evaluate("pred", "exp", metadata={"task_id": "123"})
        
        assert len(received_metadata) == 2
        assert all(m["task_id"] == "123" for m in received_metadata)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_weight_stages(self) -> None:
        """Test stages with zero weight."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.5), weight=0.0),
            EvaluationStage("s2", FixedScoreEvaluator(1.0), weight=1.0),
        ])
        
        score = pipeline.evaluate("pred", "exp")
        # Only s2 contributes: 1.0
        assert score == pytest.approx(1.0)

    def test_all_zero_weights(self) -> None:
        """Test all stages with zero weight returns 0."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.5), weight=0.0),
            EvaluationStage("s2", FixedScoreEvaluator(1.0), weight=0.0),
        ])
        
        score = pipeline.evaluate("pred", "exp")
        assert score == 0.0

    def test_threshold_exactly_met(self) -> None:
        """Test threshold exactly at boundary passes."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.5), threshold=0.5),
            EvaluationStage("s2", AlwaysPassEvaluator()),
        ])
        
        result = pipeline.evaluate_detailed("pred", "exp")
        
        assert result.early_exit is False
        assert result.stages_completed == 2

    def test_threshold_just_below_fails(self) -> None:
        """Test threshold just below boundary fails."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(0.499), threshold=0.5),
            EvaluationStage("s2", AlwaysPassEvaluator()),
        ])
        
        result = pipeline.evaluate_detailed("pred", "exp")
        
        assert result.early_exit is True
        assert result.early_exit_stage == "s1"

    def test_single_failing_filter_mid_pipeline(self) -> None:
        """Test single failing filter in middle of pipeline."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("s1", FixedScoreEvaluator(1.0), threshold=0.5),
            EvaluationStage("s2", FixedScoreEvaluator(0.1), threshold=0.5),  # fails
            EvaluationStage("s3", AlwaysPassEvaluator()),
        ])
        
        result = pipeline.evaluate_detailed("pred", "exp")
        
        assert result.stages_completed == 2
        assert result.early_exit is True
        assert result.early_exit_stage == "s2"

    def test_empty_strings(self) -> None:
        """Test evaluation with empty strings."""
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", AlwaysPassEvaluator()),
        ])
        
        score = pipeline.evaluate("", "")
        assert score == 1.0

    def test_very_long_strings(self) -> None:
        """Test evaluation with very long strings."""
        long_str = "x" * 100000
        
        pipeline = StagedPipelineEvaluator([
            EvaluationStage("test", LengthBasedEvaluator()),
        ])
        
        score = pipeline.evaluate(long_str, long_str)
        assert score == 1.0
