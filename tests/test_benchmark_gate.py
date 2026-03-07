"""Tests for benchmark gate module.

Tests the BenchmarkGate, BenchmarkThreshold, and related functionality.
"""

from __future__ import annotations

import pytest

from promptfoundry.core.benchmark_gate import (
    DEFAULT_BENCHMARK_SUITE,
    BenchmarkGate,
    BenchmarkTask,
    BenchmarkTaskType,
    BenchmarkThreshold,
    GateResult,
    create_custom_gate,
    get_default_suite,
)
from promptfoundry.core.diagnostics import RunDiagnostics, RunStatus

# =============================================================================
# BenchmarkThreshold Tests
# =============================================================================


class TestBenchmarkThreshold:
    """Tests for BenchmarkThreshold dataclass."""

    def test_default_values(self) -> None:
        """Test default threshold values."""
        threshold = BenchmarkThreshold()
        assert threshold.min_improvement == 0.05
        assert threshold.min_improvement_percent == 5.0
        assert threshold.max_no_signal_rate == 0.3
        assert threshold.min_success_rate == 0.6

    def test_custom_values(self) -> None:
        """Test custom threshold values."""
        threshold = BenchmarkThreshold(
            min_improvement=0.1,
            min_improvement_percent=10.0,
            max_no_signal_rate=0.2,
            min_success_rate=0.8,
        )
        assert threshold.min_improvement == 0.1
        assert threshold.min_success_rate == 0.8

    def test_check_improvement_passes(self) -> None:
        """Test improvement check that passes."""
        threshold = BenchmarkThreshold(
            min_improvement=0.05,
            min_improvement_percent=5.0,
        )
        # 0.1 improvement from 0.5 baseline = 20% improvement
        assert threshold.check_improvement(0.1, 0.5) is True

    def test_check_improvement_fails_absolute(self) -> None:
        """Test improvement check fails on absolute threshold."""
        threshold = BenchmarkThreshold(min_improvement=0.1)
        # 0.05 < 0.1 threshold
        assert threshold.check_improvement(0.05, 0.5) is False

    def test_check_improvement_fails_percent(self) -> None:
        """Test improvement check fails on percent threshold."""
        threshold = BenchmarkThreshold(
            min_improvement=0.01,
            min_improvement_percent=10.0,
        )
        # 0.04 from 0.5 = 8% < 10%
        assert threshold.check_improvement(0.04, 0.5) is False

    def test_check_improvement_zero_baseline(self) -> None:
        """Test improvement check with zero baseline."""
        threshold = BenchmarkThreshold(min_improvement=0.05)
        # Zero baseline skips percent check
        assert threshold.check_improvement(0.1, 0.0) is True

    def test_frozen(self) -> None:
        """Test that threshold is immutable."""
        threshold = BenchmarkThreshold()
        with pytest.raises(Exception):
            threshold.min_improvement = 0.5  # type: ignore


# =============================================================================
# BenchmarkTask Tests
# =============================================================================


class TestBenchmarkTask:
    """Tests for BenchmarkTask dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a benchmark task."""
        task = BenchmarkTask(
            name="test_task",
            task_type=BenchmarkTaskType.CLASSIFICATION,
            task_file="examples/test.yaml",
        )
        assert task.name == "test_task"
        assert task.task_type == BenchmarkTaskType.CLASSIFICATION
        assert task.seed_prompt == "{input}"  # default

    def test_with_custom_threshold(self) -> None:
        """Test task with custom threshold."""
        threshold = BenchmarkThreshold(min_improvement=0.2)
        task = BenchmarkTask(
            name="hard_task",
            task_type=BenchmarkTaskType.REASONING,
            task_file="examples/hard.yaml",
            threshold=threshold,
        )
        assert task.threshold.min_improvement == 0.2


# =============================================================================
# GateResult Tests
# =============================================================================


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty gate result."""
        result = GateResult()
        assert result.passed is False
        assert len(result.task_results) == 0

    def test_add_task_result(self) -> None:
        """Test adding task results."""
        result = GateResult()
        result.add_task_result(
            task_name="task1",
            passed=True,
            improvement=0.1,
            status=RunStatus.SUCCESS,
            details={"runs": 5},
        )

        assert "task1" in result.task_results
        assert result.task_results["task1"]["passed"] is True
        assert result.task_results["task1"]["improvement"] == 0.1
        assert result.task_results["task1"]["runs"] == 5

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = GateResult(
            passed=True,
            failures=["failure1"],
            warnings=["warning1"],
        )
        d = result.to_dict()

        assert d["passed"] is True
        assert "failure1" in d["failures"]
        assert "warning1" in d["warnings"]


# =============================================================================
# BenchmarkGate Tests
# =============================================================================


class TestBenchmarkGate:
    """Tests for BenchmarkGate class."""

    def test_default_suite(self) -> None:
        """Test gate with default suite."""
        gate = BenchmarkGate()
        assert len(gate.suite) == len(DEFAULT_BENCHMARK_SUITE)

    def test_custom_suite(self) -> None:
        """Test gate with custom suite."""
        tasks = [
            BenchmarkTask(
                name="custom",
                task_type=BenchmarkTaskType.EXTRACTION,
                task_file="test.yaml",
            ),
        ]
        gate = BenchmarkGate(suite=tasks)
        assert len(gate.suite) == 1

    def test_get_task(self) -> None:
        """Test getting task by name."""
        gate = BenchmarkGate()
        task = gate.get_task("sentiment_classification")
        assert task is not None
        assert task.task_type == BenchmarkTaskType.CLASSIFICATION

    def test_get_task_not_found(self) -> None:
        """Test getting non-existent task."""
        gate = BenchmarkGate()
        task = gate.get_task("nonexistent")
        assert task is None

    def test_check_single_run_passes(self) -> None:
        """Test checking a passing run."""
        gate = BenchmarkGate()
        diag = RunDiagnostics(
            task_name="test",
            seed_fitness=0.5,
            best_fitness=0.7,
            improvement=0.2,
            improvement_percent=40.0,
            status=RunStatus.SUCCESS,
            elapsed_time_seconds=10.0,
        )

        passed, failures = gate.check_single_run(diag)

        assert passed is True
        assert len(failures) == 0

    def test_check_single_run_fails_improvement(self) -> None:
        """Test checking a run that fails improvement threshold."""
        threshold = BenchmarkThreshold(min_improvement=0.1)
        gate = BenchmarkGate(global_threshold=threshold)

        diag = RunDiagnostics(
            task_name="test",
            improvement=0.02,
            status=RunStatus.SUCCESS,
        )

        passed, failures = gate.check_single_run(diag)

        assert passed is False
        assert any("below threshold" in f for f in failures)

    def test_check_single_run_fails_no_signal(self) -> None:
        """Test checking a no-signal run."""
        gate = BenchmarkGate()
        diag = RunDiagnostics(
            task_name="test",
            improvement=0.0,
            status=RunStatus.NO_SIGNAL,
        )

        passed, failures = gate.check_single_run(diag)

        assert passed is False
        assert any("no signal" in f.lower() for f in failures)

    def test_check_single_run_fails_runtime(self) -> None:
        """Test checking a run that exceeds runtime."""
        threshold = BenchmarkThreshold(max_runtime_seconds=10.0)
        gate = BenchmarkGate(global_threshold=threshold)

        diag = RunDiagnostics(
            task_name="test",
            improvement=0.2,
            status=RunStatus.SUCCESS,
            elapsed_time_seconds=100.0,
        )

        passed, failures = gate.check_single_run(diag)

        assert passed is False
        assert any("exceeds" in f for f in failures)

    def test_check_results_empty(self) -> None:
        """Test checking empty results."""
        gate = BenchmarkGate()
        result = gate.check_results([])

        assert result.passed is False
        assert "No benchmark results" in result.failures[0]

    def test_check_results_all_pass(self) -> None:
        """Test checking results that all pass."""
        threshold = BenchmarkThreshold(
            min_improvement=0.05,
            min_success_rate=0.5,
            max_no_signal_rate=0.5,
        )
        gate = BenchmarkGate(global_threshold=threshold)

        diagnostics = [
            RunDiagnostics(
                task_name="task1",
                seed_fitness=0.5,
                improvement=0.2,
                status=RunStatus.SUCCESS,
            ),
            RunDiagnostics(
                task_name="task1",
                seed_fitness=0.5,
                improvement=0.15,
                status=RunStatus.SUCCESS,
            ),
        ]

        result = gate.check_results(diagnostics)

        assert result.passed is True
        assert result.summary["total_runs"] == 2
        assert result.summary["successful_runs"] == 2

    def test_check_results_task_fails_success_rate(self) -> None:
        """Test checking results where success rate is too low."""
        threshold = BenchmarkThreshold(
            min_improvement=0.01,
            min_success_rate=0.8,  # High requirement
            max_no_signal_rate=0.1,
        )
        gate = BenchmarkGate(global_threshold=threshold)

        diagnostics = [
            RunDiagnostics(task_name="task1", improvement=0.1, status=RunStatus.SUCCESS),
            RunDiagnostics(task_name="task1", improvement=0.0, status=RunStatus.NO_SIGNAL),
            RunDiagnostics(task_name="task1", improvement=0.0, status=RunStatus.NO_SIGNAL),
        ]

        result = gate.check_results(diagnostics)

        # 1/3 = 33% success rate < 80% required
        assert result.passed is False
        assert any("Success rate" in f for f in result.failures)

    def test_check_results_task_fails_no_signal_rate(self) -> None:
        """Test checking results where no-signal rate is too high."""
        threshold = BenchmarkThreshold(
            min_improvement=0.01,
            min_success_rate=0.3,
            max_no_signal_rate=0.2,  # Low tolerance
        )
        gate = BenchmarkGate(global_threshold=threshold)

        diagnostics = [
            RunDiagnostics(task_name="task1", improvement=0.1, status=RunStatus.SUCCESS),
            RunDiagnostics(task_name="task1", improvement=0.0, status=RunStatus.NO_SIGNAL),
        ]

        result = gate.check_results(diagnostics)

        # 1/2 = 50% no-signal rate > 20% max
        assert result.passed is False
        assert any("No-signal rate" in f for f in result.failures)

    def test_check_results_multiple_tasks(self) -> None:
        """Test checking results from multiple tasks."""
        threshold = BenchmarkThreshold(min_improvement=0.05)
        gate = BenchmarkGate(global_threshold=threshold)

        diagnostics = [
            RunDiagnostics(task_name="task1", improvement=0.1, status=RunStatus.SUCCESS),
            RunDiagnostics(task_name="task2", improvement=0.08, status=RunStatus.SUCCESS),
        ]

        result = gate.check_results(diagnostics)

        assert result.summary["tasks_checked"] == 2
        assert "task1" in result.task_results
        assert "task2" in result.task_results

    def test_check_results_warns_incomplete_suite(self) -> None:
        """Test warning when not all suite tasks are run."""
        gate = BenchmarkGate()  # Has 3 default tasks

        diagnostics = [
            RunDiagnostics(task_name="sentiment_classification", improvement=0.2, status=RunStatus.SUCCESS),
        ]

        result = gate.check_results(diagnostics)

        # Should warn about missing tasks
        assert any("benchmark tasks were run" in w for w in result.warnings)

    def test_format_report_passed(self) -> None:
        """Test formatting a passing report."""
        result = GateResult(
            passed=True,
            summary={"total_runs": 5, "successful_runs": 5, "tasks_checked": 2, "tasks_passed": 2},
            task_results={"task1": {"passed": True, "improvement": 0.1}},
        )

        gate = BenchmarkGate()
        report = gate.format_report(result)

        assert "PASSED" in report
        assert "task1" in report

    def test_format_report_failed(self) -> None:
        """Test formatting a failing report."""
        result = GateResult(
            passed=False,
            failures=["[task1] Success rate 30% below minimum 60%"],
            warnings=["Only 1 of 3 tasks run"],
        )

        gate = BenchmarkGate()
        report = gate.format_report(result)

        assert "FAILED" in report
        assert "Failures" in report
        assert "Success rate" in report


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_default_suite(self) -> None:
        """Test getting default suite."""
        suite = get_default_suite()
        assert len(suite) == 3
        assert any(t.name == "sentiment_classification" for t in suite)
        assert any(t.name == "json_formatting" for t in suite)
        assert any(t.name == "arithmetic_reasoning" for t in suite)

    def test_create_custom_gate(self) -> None:
        """Test creating custom gate."""
        gate = create_custom_gate(
            min_improvement=0.1,
            min_success_rate=0.8,
            max_no_signal_rate=0.1,
        )

        # Verify the gate works with custom thresholds
        diag = RunDiagnostics(
            task_name="test",
            improvement=0.05,  # Below 0.1 threshold
            status=RunStatus.SUCCESS,
        )

        passed, failures = gate.check_single_run(diag)
        assert passed is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for benchmark gate."""

    def test_full_benchmark_workflow(self) -> None:
        """Test complete benchmark gate workflow."""
        # Create gate with custom thresholds
        gate = create_custom_gate(
            min_improvement=0.05,
            min_success_rate=0.6,
            max_no_signal_rate=0.4,
        )

        # Simulate benchmark results
        diagnostics = [
            # Task 1: 3 runs, 2 success, 1 no-signal
            RunDiagnostics(
                task_name="extraction_task",
                seed_fitness=0.4,
                improvement=0.15,
                status=RunStatus.SUCCESS,
                elapsed_time_seconds=30.0,
            ),
            RunDiagnostics(
                task_name="extraction_task",
                seed_fitness=0.4,
                improvement=0.12,
                status=RunStatus.SUCCESS,
                elapsed_time_seconds=25.0,
            ),
            RunDiagnostics(
                task_name="extraction_task",
                seed_fitness=0.4,
                improvement=0.0,
                status=RunStatus.NO_SIGNAL,
                elapsed_time_seconds=45.0,
            ),
            # Task 2: 2 runs, both success
            RunDiagnostics(
                task_name="formatting_task",
                seed_fitness=0.5,
                improvement=0.2,
                status=RunStatus.SUCCESS,
                elapsed_time_seconds=20.0,
            ),
            RunDiagnostics(
                task_name="formatting_task",
                seed_fitness=0.5,
                improvement=0.18,
                status=RunStatus.SUCCESS,
                elapsed_time_seconds=22.0,
            ),
        ]

        # Check results
        result = gate.check_results(diagnostics)

        # Should pass overall
        assert result.passed is True
        assert result.summary["total_runs"] == 5
        assert result.summary["successful_runs"] == 4
        assert result.summary["tasks_checked"] == 2

        # Generate report
        report = gate.format_report(result)
        assert "PASSED" in report
        assert "extraction_task" in report
        assert "formatting_task" in report

    def test_benchmark_with_task_specific_thresholds(self) -> None:
        """Test benchmark with task-specific thresholds."""
        # Create tasks with different thresholds
        easy_task = BenchmarkTask(
            name="easy",
            task_type=BenchmarkTaskType.CLASSIFICATION,
            task_file="easy.yaml",
            threshold=BenchmarkThreshold(min_improvement=0.02),  # Low bar
        )
        hard_task = BenchmarkTask(
            name="hard",
            task_type=BenchmarkTaskType.REASONING,
            task_file="hard.yaml",
            threshold=BenchmarkThreshold(min_improvement=0.15),  # High bar
        )

        gate = BenchmarkGate(suite=[easy_task, hard_task])

        # Diagnostic that passes easy but fails hard
        diag_easy = RunDiagnostics(
            task_name="easy",
            improvement=0.05,  # Passes 0.02 threshold
            status=RunStatus.SUCCESS,
        )
        diag_hard = RunDiagnostics(
            task_name="hard",
            improvement=0.05,  # Fails 0.15 threshold
            status=RunStatus.SUCCESS,
        )

        # Check individually
        easy_passed, _ = gate.check_single_run(diag_easy, easy_task)
        hard_passed, hard_failures = gate.check_single_run(diag_hard, hard_task)

        assert easy_passed is True
        assert hard_passed is False
        assert any("threshold" in f for f in hard_failures)
