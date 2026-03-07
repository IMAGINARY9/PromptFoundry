"""Tests for diagnostics and reporting module.

Tests the RunDiagnostics, BenchmarkSummary, and formatting functions.
"""

from __future__ import annotations

import pytest

from promptfoundry.core.diagnostics import (
    BenchmarkSummary,
    GenerationMetrics,
    RunDiagnostics,
    RunStatus,
    TerminationReason,
    format_benchmark_summary,
    format_diagnostics_report,
)

# =============================================================================
# GenerationMetrics Tests
# =============================================================================


class TestGenerationMetrics:
    """Tests for GenerationMetrics dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating generation metrics."""
        metrics = GenerationMetrics(
            generation=1,
            best_fitness=0.9,
            average_fitness=0.7,
        )
        assert metrics.generation == 1
        assert metrics.best_fitness == 0.9
        assert metrics.average_fitness == 0.7

    def test_with_timing(self) -> None:
        """Test metrics with timing data."""
        metrics = GenerationMetrics(
            generation=5,
            best_fitness=0.95,
            average_fitness=0.8,
            evaluation_time_ms=1500.0,
            llm_calls=10,
            cache_hits=5,
        )
        assert metrics.evaluation_time_ms == 1500.0
        assert metrics.llm_calls == 10
        assert metrics.cache_hits == 5

    def test_avg_latency_per_call(self) -> None:
        """Test average latency calculation."""
        metrics = GenerationMetrics(
            generation=1,
            best_fitness=0.9,
            average_fitness=0.8,
            evaluation_time_ms=1000.0,
            llm_calls=8,
            cache_hits=2,
        )
        # 1000ms / 10 total calls = 100ms per call
        assert metrics.avg_latency_per_call_ms == pytest.approx(100.0)

    def test_avg_latency_zero_calls(self) -> None:
        """Test latency with zero calls."""
        metrics = GenerationMetrics(
            generation=1,
            best_fitness=0.9,
            average_fitness=0.8,
            evaluation_time_ms=100.0,
            llm_calls=0,
            cache_hits=0,
        )
        assert metrics.avg_latency_per_call_ms == 0.0


# =============================================================================
# RunDiagnostics Tests
# =============================================================================


class TestRunDiagnostics:
    """Tests for RunDiagnostics class."""

    def test_basic_creation(self) -> None:
        """Test creating run diagnostics."""
        diag = RunDiagnostics(
            task_name="test_task",
            seed_fitness=0.5,
            best_fitness=0.8,
        )
        assert diag.task_name == "test_task"
        assert diag.seed_fitness == 0.5
        assert diag.best_fitness == 0.8

    def test_analyze_success(self) -> None:
        """Test analyzing a successful run."""
        history_data = {
            "task_name": "sentiment",
            "generations": [
                {"generation": 0, "best_fitness": 0.5, "average_fitness": 0.4, "population_size": 10},
                {"generation": 1, "best_fitness": 0.7, "average_fitness": 0.6, "population_size": 10},
                {"generation": 2, "best_fitness": 0.9, "average_fitness": 0.75, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(
            history_data,
            termination_reason="max_generations",
            elapsed_time=30.0,
        )

        assert diag.task_name == "sentiment"
        assert diag.seed_fitness == 0.5
        assert diag.best_fitness == 0.9
        assert diag.improvement == pytest.approx(0.4)
        assert diag.improvement_percent == pytest.approx(80.0)
        assert diag.total_generations == 3
        assert diag.status == RunStatus.SUCCESS
        assert diag.termination_reason == TerminationReason.MAX_GENERATIONS

    def test_analyze_no_signal(self) -> None:
        """Test analyzing a no-signal run."""
        history_data = {
            "task_name": "hard_task",
            "generations": [
                {"generation": 0, "best_fitness": 0.5, "average_fitness": 0.4, "population_size": 10},
                {"generation": 1, "best_fitness": 0.5, "average_fitness": 0.45, "population_size": 10},
                {"generation": 2, "best_fitness": 0.5, "average_fitness": 0.48, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(
            history_data,
            termination_reason="patience_exhausted",
        )

        assert diag.improvement == pytest.approx(0.0)
        assert diag.status == RunStatus.NO_SIGNAL
        assert "NO SIGNAL" in diag.warnings[0]

    def test_analyze_interrupted(self) -> None:
        """Test analyzing an interrupted run."""
        history_data = {
            "task_name": "long_task",
            "generations": [
                {"generation": 0, "best_fitness": 0.3, "average_fitness": 0.25, "population_size": 10},
                {"generation": 1, "best_fitness": 0.5, "average_fitness": 0.4, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(
            history_data,
            termination_reason="interrupted",
        )

        assert diag.termination_reason == TerminationReason.INTERRUPTED
        assert diag.status == RunStatus.PARTIAL  # Some improvement
        assert diag.improvement > 0

    def test_analyze_empty_generations(self) -> None:
        """Test analyzing history with no generations."""
        history_data = {"task_name": "failed", "generations": []}

        diag = RunDiagnostics.analyze(history_data)

        assert diag.status == RunStatus.FAILED
        assert "No generation data" in diag.warnings[0]

    def test_analyze_with_timing_metadata(self) -> None:
        """Test analyzing with per-generation timing."""
        history_data = {
            "task_name": "timed_task",
            "generations": [
                {
                    "generation": 0,
                    "best_fitness": 0.5,
                    "average_fitness": 0.4,
                    "population_size": 10,
                    "metadata": {"evaluation_time_ms": 1000, "llm_calls": 10, "cache_hits": 0},
                },
                {
                    "generation": 1,
                    "best_fitness": 0.8,
                    "average_fitness": 0.6,
                    "population_size": 10,
                    "metadata": {"evaluation_time_ms": 800, "llm_calls": 5, "cache_hits": 5},
                },
            ],
        }

        diag = RunDiagnostics.analyze(
            history_data,
            total_llm_calls=15,
            total_cache_hits=5,
        )

        assert diag.total_llm_calls == 15
        assert diag.cache_hit_rate == pytest.approx(0.25)
        assert len(diag.generations) == 2
        assert diag.generations[0].evaluation_time_ms == 1000
        assert diag.generations[1].llm_calls == 5

    def test_detect_plateau_warning(self) -> None:
        """Test detection of fitness plateau."""
        history_data = {
            "task_name": "plateau_task",
            "generations": [
                {"generation": 0, "best_fitness": 0.5, "average_fitness": 0.4, "population_size": 10},
                {"generation": 1, "best_fitness": 0.8, "average_fitness": 0.6, "population_size": 10},
                {"generation": 2, "best_fitness": 0.8, "average_fitness": 0.7, "population_size": 10},
                {"generation": 3, "best_fitness": 0.8, "average_fitness": 0.75, "population_size": 10},
                {"generation": 4, "best_fitness": 0.8, "average_fitness": 0.78, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(history_data)

        plateau_warnings = [w for w in diag.warnings if "PLATEAU" in w]
        assert len(plateau_warnings) == 1

    def test_detect_high_cache_warning(self) -> None:
        """Test detection of high cache hit rate."""
        history_data = {
            "task_name": "cache_task",
            "generations": [
                {"generation": 0, "best_fitness": 0.6, "average_fitness": 0.5, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(
            history_data,
            total_llm_calls=10,
            total_cache_hits=90,  # 90% cache hits
        )

        cache_warnings = [w for w in diag.warnings if "CACHE" in w]
        assert len(cache_warnings) == 1

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        diag = RunDiagnostics(
            task_name="test",
            seed_fitness=0.5,
            best_fitness=0.8,
            improvement=0.3,
            improvement_percent=60.0,
            termination_reason=TerminationReason.MAX_GENERATIONS,
            status=RunStatus.SUCCESS,
            total_generations=10,
            warnings=["test warning"],
        )

        d = diag.to_dict()

        assert d["task_name"] == "test"
        assert d["termination_reason"] == "max_generations"
        assert d["status"] == "success"
        assert d["warnings"] == ["test warning"]


# =============================================================================
# BenchmarkSummary Tests
# =============================================================================


class TestBenchmarkSummary:
    """Tests for BenchmarkSummary class."""

    def test_empty_summary(self) -> None:
        """Test empty benchmark summary."""
        summary = BenchmarkSummary()

        assert summary.total_runs == 0
        assert summary.successful_runs == 0
        assert summary.average_improvement == 0.0

    def test_add_run(self) -> None:
        """Test adding runs."""
        summary = BenchmarkSummary()

        diag = RunDiagnostics(
            task_name="task1",
            status=RunStatus.SUCCESS,
            improvement=0.3,
            elapsed_time_seconds=10.0,
        )
        summary.add_run(diag)

        assert summary.total_runs == 1
        assert summary.successful_runs == 1

    def test_multiple_runs(self) -> None:
        """Test statistics with multiple runs."""
        summary = BenchmarkSummary()

        summary.add_run(RunDiagnostics(
            task_name="task1",
            status=RunStatus.SUCCESS,
            improvement=0.3,
            elapsed_time_seconds=10.0,
        ))
        summary.add_run(RunDiagnostics(
            task_name="task1",
            status=RunStatus.SUCCESS,
            improvement=0.5,
            elapsed_time_seconds=20.0,
        ))
        summary.add_run(RunDiagnostics(
            task_name="task2",
            status=RunStatus.NO_SIGNAL,
            improvement=0.0,
            elapsed_time_seconds=15.0,
        ))

        assert summary.total_runs == 3
        assert summary.successful_runs == 2
        assert summary.no_signal_runs == 1
        assert summary.average_improvement == pytest.approx(0.8 / 3)
        assert summary.average_runtime == pytest.approx(15.0)

    def test_by_task(self) -> None:
        """Test grouping by task."""
        summary = BenchmarkSummary()

        summary.add_run(RunDiagnostics(task_name="task1", improvement=0.3))
        summary.add_run(RunDiagnostics(task_name="task1", improvement=0.4))
        summary.add_run(RunDiagnostics(task_name="task2", improvement=0.2))

        by_task = summary.by_task()

        assert len(by_task["task1"]) == 2
        assert len(by_task["task2"]) == 1

    def test_by_status(self) -> None:
        """Test grouping by status."""
        summary = BenchmarkSummary()

        summary.add_run(RunDiagnostics(status=RunStatus.SUCCESS))
        summary.add_run(RunDiagnostics(status=RunStatus.SUCCESS))
        summary.add_run(RunDiagnostics(status=RunStatus.NO_SIGNAL))

        by_status = summary.by_status()

        assert len(by_status[RunStatus.SUCCESS]) == 2
        assert len(by_status[RunStatus.NO_SIGNAL]) == 1

    def test_task_stats(self) -> None:
        """Test per-task statistics."""
        summary = BenchmarkSummary()

        summary.add_run(RunDiagnostics(
            task_name="task1",
            status=RunStatus.SUCCESS,
            improvement=0.3,
            elapsed_time_seconds=10.0,
        ))
        summary.add_run(RunDiagnostics(
            task_name="task1",
            status=RunStatus.SUCCESS,
            improvement=0.5,
            elapsed_time_seconds=20.0,
        ))

        stats = summary.task_stats()

        assert stats["task1"]["num_runs"] == 2
        assert stats["task1"]["success_rate"] == 1.0
        assert stats["task1"]["avg_improvement"] == pytest.approx(0.4)
        assert stats["task1"]["max_improvement"] == pytest.approx(0.5)
        assert stats["task1"]["min_improvement"] == pytest.approx(0.3)
        assert stats["task1"]["avg_runtime_s"] == pytest.approx(15.0)

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        summary = BenchmarkSummary()
        summary.add_run(RunDiagnostics(task_name="test", improvement=0.5))

        d = summary.to_dict()

        assert d["total_runs"] == 1
        assert "task_stats" in d
        assert "runs" in d
        assert len(d["runs"]) == 1


# =============================================================================
# Formatting Tests
# =============================================================================


class TestFormatDiagnosticsReport:
    """Tests for format_diagnostics_report function."""

    def test_basic_format(self) -> None:
        """Test basic report formatting."""
        diag = RunDiagnostics(
            task_name="test_task",
            seed_fitness=0.5,
            best_fitness=0.8,
            improvement=0.3,
            improvement_percent=60.0,
            termination_reason=TerminationReason.MAX_GENERATIONS,
            status=RunStatus.SUCCESS,
            total_generations=10,
            total_evaluations=100,
            elapsed_time_seconds=30.0,
        )

        report = format_diagnostics_report(diag)

        assert "test_task" in report
        assert "SUCCESS" in report
        assert "0.5" in report  # seed fitness
        assert "0.8" in report  # best fitness
        assert "60.0%" in report or "+60" in report  # improvement

    def test_format_with_warnings(self) -> None:
        """Test report with warnings."""
        diag = RunDiagnostics(
            task_name="warned_task",
            status=RunStatus.NO_SIGNAL,
            warnings=["NO SIGNAL: Test warning"],
        )

        report = format_diagnostics_report(diag)

        assert "Warnings" in report
        assert "NO SIGNAL" in report

    def test_format_with_generations(self) -> None:
        """Test report with per-generation data."""
        diag = RunDiagnostics(
            task_name="gen_task",
            generations=[
                GenerationMetrics(0, 0.5, 0.4, evaluation_time_ms=1000, llm_calls=10),
                GenerationMetrics(1, 0.7, 0.6, evaluation_time_ms=800, llm_calls=8),
            ],
        )

        report = format_diagnostics_report(diag)

        assert "Per-Generation" in report
        assert "Time(ms)" in report


class TestFormatBenchmarkSummary:
    """Tests for format_benchmark_summary function."""

    def test_basic_format(self) -> None:
        """Test basic summary formatting."""
        summary = BenchmarkSummary()
        summary.add_run(RunDiagnostics(
            task_name="task1",
            status=RunStatus.SUCCESS,
            improvement=0.3,
        ))

        report = format_benchmark_summary(summary)

        assert "BENCHMARK SUMMARY" in report
        assert "Total Runs" in report
        assert "1" in report

    def test_format_with_task_breakdown(self) -> None:
        """Test summary with task breakdown."""
        summary = BenchmarkSummary()
        summary.add_run(RunDiagnostics(
            task_name="sentiment",
            status=RunStatus.SUCCESS,
            improvement=0.3,
            elapsed_time_seconds=10.0,
        ))
        summary.add_run(RunDiagnostics(
            task_name="json",
            status=RunStatus.NO_SIGNAL,
            improvement=0.0,
            elapsed_time_seconds=15.0,
        ))

        report = format_benchmark_summary(summary)

        assert "Per-Task Breakdown" in report
        assert "sentiment" in report
        assert "json" in report


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unknown_termination_reason(self) -> None:
        """Test handling unknown termination reason."""
        history_data = {
            "generations": [
                {"generation": 0, "best_fitness": 0.5, "average_fitness": 0.4, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(
            history_data,
            termination_reason="some_weird_reason",
        )

        assert diag.termination_reason == TerminationReason.UNKNOWN

    def test_zero_seed_fitness_improvement(self) -> None:
        """Test improvement calculation from zero seed."""
        history_data = {
            "generations": [
                {"generation": 0, "best_fitness": 0.0, "average_fitness": 0.0, "population_size": 10},
                {"generation": 1, "best_fitness": 0.5, "average_fitness": 0.3, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(history_data)

        assert diag.improvement == 0.5
        assert diag.improvement_percent == 100.0  # From zero to something

    def test_negative_improvement_status(self) -> None:
        """Test that no improvement leads to NO_SIGNAL."""
        history_data = {
            "generations": [
                {"generation": 0, "best_fitness": 0.8, "average_fitness": 0.7, "population_size": 10},
                {"generation": 1, "best_fitness": 0.75, "average_fitness": 0.65, "population_size": 10},
            ],
        }

        diag = RunDiagnostics.analyze(history_data)

        # Best is still 0.8 from gen 0, so improvement = 0
        assert diag.best_fitness == 0.8
        assert diag.seed_fitness == 0.8
        assert diag.improvement == 0.0
        assert diag.status == RunStatus.NO_SIGNAL
