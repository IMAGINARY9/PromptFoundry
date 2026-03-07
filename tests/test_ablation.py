"""Tests for MVP 3 ablation utilities module."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptfoundry.strategies.ablation import (
    AblationResult,
    AblationStudy,
    AblationTracker,
    OperatorMetrics,
)


class TestOperatorMetrics:
    """Test operator metrics dataclass."""

    def test_defaults(self) -> None:
        """Test default metric values."""
        metrics = OperatorMetrics(name="test_op")
        assert metrics.attempts == 0
        assert metrics.successes == 0
        assert metrics.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        metrics = OperatorMetrics(
            name="test_op",
            attempts=10,
            successes=4,
        )
        assert metrics.success_rate == 0.4

    def test_failure_rate_calculation(self) -> None:
        """Test failure rate calculation."""
        metrics = OperatorMetrics(
            name="test_op",
            attempts=10,
            failures=3,
        )
        assert metrics.failure_rate == 0.3

    def test_net_improvement(self) -> None:
        """Test net improvement calculation."""
        metrics = OperatorMetrics(
            name="test_op",
            total_improvement=0.5,
            total_regression=-0.2,
        )
        assert metrics.net_improvement == pytest.approx(0.3, rel=0.01)

    def test_avg_improvement_when_successful(self) -> None:
        """Test average improvement calculation."""
        metrics = OperatorMetrics(
            name="test_op",
            successes=5,
            total_improvement=0.5,
        )
        assert metrics.avg_improvement_when_successful == 0.1

    def test_effectiveness_score(self) -> None:
        """Test effectiveness score."""
        # Good operator
        good_metrics = OperatorMetrics(
            name="good_op",
            attempts=10,
            successes=8,
            failures=1,
            neutral=1,
            total_improvement=0.8,
        )
        # Should be positive
        assert good_metrics.effectiveness_score > 0

        # Bad operator
        bad_metrics = OperatorMetrics(
            name="bad_op",
            attempts=10,
            successes=1,
            failures=8,
            neutral=1,
            total_improvement=0.1,
        )
        # Should be negative or much lower
        assert bad_metrics.effectiveness_score < good_metrics.effectiveness_score

    def test_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = OperatorMetrics(
            name="test_op",
            attempts=5,
            successes=3,
            total_improvement=0.3,
        )
        result = metrics.to_dict()
        assert result["name"] == "test_op"
        assert result["attempts"] == 5
        assert result["success_rate"] == 0.6


class TestAblationTracker:
    """Test ablation tracker."""

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        tracker = AblationTracker(task_name="test_task")
        assert tracker.task_name == "test_task"

    def test_set_baseline(self) -> None:
        """Test baseline setting."""
        tracker = AblationTracker()
        tracker.set_baseline(0.5)
        assert tracker._baseline_fitness == 0.5

    def test_record_mutation_success(self) -> None:
        """Test recording successful mutation."""
        tracker = AblationTracker()

        tracker.record_mutation(
            operator_name="add_constraint",
            parent_fitness=0.5,
            child_fitness=0.7,
        )

        metrics = tracker.get_metrics()
        assert "add_constraint" in metrics
        assert metrics["add_constraint"].successes == 1
        assert metrics["add_constraint"].total_improvement == pytest.approx(0.2, rel=0.01)

    def test_record_mutation_failure(self) -> None:
        """Test recording failed mutation."""
        tracker = AblationTracker()

        tracker.record_mutation(
            operator_name="rephrase",
            parent_fitness=0.5,
            child_fitness=0.4,
        )

        metrics = tracker.get_metrics()
        assert metrics["rephrase"].failures == 1
        assert metrics["rephrase"].total_regression == pytest.approx(-0.1, rel=0.01)

    def test_record_mutation_neutral(self) -> None:
        """Test recording neutral mutation."""
        tracker = AblationTracker()

        tracker.record_mutation(
            operator_name="neutral_op",
            parent_fitness=0.5,
            child_fitness=0.5,
        )

        metrics = tracker.get_metrics()
        assert metrics["neutral_op"].neutral == 1
        assert metrics["neutral_op"].successes == 0
        assert metrics["neutral_op"].failures == 0

    def test_record_generation(self) -> None:
        """Test generation recording."""
        tracker = AblationTracker()

        tracker.record_generation(
            generation=1,
            best_fitness=0.8,
            avg_fitness=0.6,
            operator_counts={"op_a": 3, "op_b": 2},
        )

        assert len(tracker._generations) == 1
        assert tracker._generations[0]["best_fitness"] == 0.8

    def test_generate_result(self) -> None:
        """Test result generation."""
        tracker = AblationTracker(task_name="test_task")
        tracker.set_baseline(0.5)

        # Record some mutations
        for i in range(5):
            tracker.record_mutation("good_op", 0.5, 0.55 + i * 0.01)
        for i in range(5):
            tracker.record_mutation("bad_op", 0.5, 0.45 - i * 0.01)

        result = tracker.generate_result()

        assert result.task_name == "test_task"
        assert result.baseline_fitness == 0.5
        assert "good_op" in result.best_combination
        assert "bad_op" in result.worst_operators
        assert "good_op" in result.recommended_weights
        assert result.recommended_weights["good_op"] > result.recommended_weights["bad_op"]

    def test_get_summary(self) -> None:
        """Test summary generation."""
        tracker = AblationTracker(task_name="summary_test")
        tracker.set_baseline(0.5)
        tracker.record_mutation("op_a", 0.5, 0.6)
        tracker.record_mutation("op_b", 0.5, 0.4)

        summary = tracker.get_summary()

        assert "summary_test" in summary
        assert "op_a" in summary
        assert "op_b" in summary


class TestAblationResult:
    """Test ablation result dataclass."""

    def test_to_dict(self) -> None:
        """Test result serialization."""
        metrics = {"op": OperatorMetrics(name="op", attempts=5, successes=3)}
        result = AblationResult(
            task_name="test",
            baseline_fitness=0.5,
            operator_metrics=metrics,
        )

        data = result.to_dict()
        assert data["task_name"] == "test"
        assert "op" in data["operator_metrics"]

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading result."""
        metrics = {"op": OperatorMetrics(name="op", attempts=5, successes=3)}
        result = AblationResult(
            task_name="test",
            baseline_fitness=0.5,
            operator_metrics=metrics,
            best_combination=["op"],
            recommended_weights={"op": 1.5},
        )

        path = tmp_path / "ablation.json"
        result.save(path)
        assert path.exists()

        loaded = AblationResult.load(path)
        assert loaded.task_name == "test"
        assert loaded.baseline_fitness == 0.5
        assert "op" in loaded.operator_metrics
        assert loaded.best_combination == ["op"]


class TestAblationStudy:
    """Test ablation study."""

    def test_initialization(self) -> None:
        """Test study initialization."""
        study = AblationStudy(
            operators=["op_a", "op_b", "op_c"],
            baseline_fitness=0.5,
        )
        assert len(study.operators) == 3
        assert study.baseline_fitness == 0.5

    def test_record_run(self) -> None:
        """Test recording a run."""
        study = AblationStudy(operators=["op_a", "op_b"])

        study.record_run(
            config_name="all_ops",
            enabled_operators=["op_a", "op_b"],
            final_fitness=0.8,
            improvement=0.3,
            runtime_seconds=10.0,
        )

        assert "all_ops" in study._results
        assert study._results["all_ops"]["final_fitness"] == 0.8

    def test_leave_one_out_analysis(self) -> None:
        """Test leave-one-out analysis."""
        study = AblationStudy(operators=["op_a", "op_b", "op_c"])

        # All operators
        study.record_run("all", ["op_a", "op_b", "op_c"], 0.8, 0.3, 10)
        # Without op_a
        study.record_run("no_a", ["op_b", "op_c"], 0.7, 0.2, 10)
        # Without op_b
        study.record_run("no_b", ["op_a", "op_c"], 0.75, 0.25, 10)
        # Without op_c
        study.record_run("no_c", ["op_a", "op_b"], 0.65, 0.15, 10)

        impacts = study.get_leave_one_out_analysis()

        # op_a has biggest impact (0.8 - 0.7 = 0.1)
        assert impacts["op_a"] == pytest.approx(0.1, rel=0.01)
        # op_c has smallest impact (0.8 - 0.65 = 0.15)
        assert impacts["op_c"] == pytest.approx(0.15, rel=0.01)

    def test_isolated_analysis(self) -> None:
        """Test isolated operator analysis."""
        study = AblationStudy(
            operators=["op_a", "op_b"],
            baseline_fitness=0.5,
        )

        # Only op_a
        study.record_run("only_a", ["op_a"], 0.65, 0.15, 10)
        # Only op_b
        study.record_run("only_b", ["op_b"], 0.55, 0.05, 10)

        isolated = study.get_isolated_analysis()

        assert isolated["op_a"] == 0.15
        assert isolated["op_b"] == 0.05

    def test_generate_report(self) -> None:
        """Test report generation."""
        study = AblationStudy(
            operators=["op_a", "op_b"],
            baseline_fitness=0.5,
        )

        study.record_run("all", ["op_a", "op_b"], 0.8, 0.3, 10)
        study.record_run("no_a", ["op_b"], 0.7, 0.2, 10)
        study.record_run("only_a", ["op_a"], 0.65, 0.15, 10)

        report = study.generate_report()

        assert "Ablation Study Report" in report
        assert "Baseline Fitness: 0.5" in report
