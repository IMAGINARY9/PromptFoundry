"""Benchmark gate for validating optimization quality.

This module defines benchmark suites and improvement thresholds
that optimization runs must meet before release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from promptfoundry.core.diagnostics import RunDiagnostics, RunStatus


class BenchmarkTaskType(Enum):
    """Categories of benchmark tasks."""

    EXTRACTION = "extraction"
    FORMATTING = "formatting"
    REASONING = "reasoning"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class BenchmarkThreshold:
    """Minimum improvement thresholds for a benchmark.

    Attributes:
        min_improvement: Minimum absolute improvement required.
        min_improvement_percent: Minimum percentage improvement required.
        max_no_signal_rate: Maximum fraction of no-signal runs allowed.
        min_success_rate: Minimum fraction of successful runs required.
        max_runtime_seconds: Maximum allowed runtime per run.
    """

    min_improvement: float = 0.05
    min_improvement_percent: float = 5.0
    max_no_signal_rate: float = 0.3
    min_success_rate: float = 0.6
    max_runtime_seconds: float = 300.0

    def check_improvement(self, improvement: float, baseline: float) -> bool:
        """Check if improvement meets thresholds.

        Args:
            improvement: Absolute improvement value.
            baseline: Baseline (seed) fitness.

        Returns:
            True if improvement meets thresholds.
        """
        if improvement < self.min_improvement:
            return False

        if baseline > 0:
            percent = (improvement / baseline) * 100
            if percent < self.min_improvement_percent:
                return False

        return True


@dataclass
class BenchmarkTask:
    """A single benchmark task definition.

    Attributes:
        name: Task name (e.g., "sentiment_classification").
        task_type: Category of the task.
        task_file: Path to the task YAML file.
        seed_prompt: Default seed prompt for this task.
        threshold: Improvement thresholds for this task.
        description: Human-readable description.
    """

    name: str
    task_type: BenchmarkTaskType
    task_file: str
    seed_prompt: str = "{input}"
    threshold: BenchmarkThreshold = field(default_factory=BenchmarkThreshold)
    description: str = ""


@dataclass
class GateResult:
    """Result of running a benchmark gate check.

    Attributes:
        passed: Whether all gates were passed.
        task_results: Per-task gate results.
        summary: Overall summary statistics.
        failures: List of specific failure messages.
        warnings: List of warning messages.
    """

    passed: bool = False
    task_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_task_result(
        self,
        task_name: str,
        passed: bool,
        improvement: float,
        status: RunStatus,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a task result."""
        self.task_results[task_name] = {
            "passed": passed,
            "improvement": improvement,
            "status": status.value,
            **(details or {}),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "task_results": self.task_results,
            "summary": self.summary,
            "failures": self.failures,
            "warnings": self.warnings,
        }


# =============================================================================
# Default Benchmark Suite
# =============================================================================

# Thresholds calibrated for different task types
EXTRACTION_THRESHOLD = BenchmarkThreshold(
    min_improvement=0.05,
    min_improvement_percent=5.0,
    max_no_signal_rate=0.3,
    min_success_rate=0.6,
)

FORMATTING_THRESHOLD = BenchmarkThreshold(
    min_improvement=0.03,
    min_improvement_percent=3.0,
    max_no_signal_rate=0.4,  # Formatting can be harder
    min_success_rate=0.5,
)

REASONING_THRESHOLD = BenchmarkThreshold(
    min_improvement=0.05,
    min_improvement_percent=5.0,
    max_no_signal_rate=0.5,  # Reasoning is hardest
    min_success_rate=0.4,
)

CLASSIFICATION_THRESHOLD = BenchmarkThreshold(
    min_improvement=0.10,  # Classification should show clear improvement
    min_improvement_percent=10.0,
    max_no_signal_rate=0.2,
    min_success_rate=0.7,
)


# Default benchmark suite
DEFAULT_BENCHMARK_SUITE: list[BenchmarkTask] = [
    BenchmarkTask(
        name="sentiment_classification",
        task_type=BenchmarkTaskType.CLASSIFICATION,
        task_file="examples/sentiment_task.yaml",
        seed_prompt="Classify the sentiment of the following text as positive, negative, or neutral.\n\nText: {input}\n\nSentiment:",
        threshold=CLASSIFICATION_THRESHOLD,
        description="Classify text sentiment",
    ),
    BenchmarkTask(
        name="json_formatting",
        task_type=BenchmarkTaskType.FORMATTING,
        task_file="examples/json_formatting_task.yaml",
        seed_prompt="Extract the information from the following text and format it as JSON.\n\nText: {input}\n\nJSON:",
        threshold=FORMATTING_THRESHOLD,
        description="Extract and format data as JSON",
    ),
    BenchmarkTask(
        name="arithmetic_reasoning",
        task_type=BenchmarkTaskType.REASONING,
        task_file="examples/arithmetic_task.yaml",
        seed_prompt="Solve the following math problem step by step.\n\nProblem: {input}\n\nAnswer:",
        threshold=REASONING_THRESHOLD,
        description="Solve word math problems",
    ),
]


class BenchmarkGate:
    """Gate for validating benchmark results meet quality thresholds.

    Example:
        >>> gate = BenchmarkGate(suite=DEFAULT_BENCHMARK_SUITE)
        >>> result = gate.check_results(diagnostics_list)
        >>> if result.passed:
        ...     print("All benchmarks passed!")
        ... else:
        ...     for failure in result.failures:
        ...         print(f"FAIL: {failure}")
    """

    def __init__(
        self,
        suite: list[BenchmarkTask] | None = None,
        global_threshold: BenchmarkThreshold | None = None,
    ) -> None:
        """Initialize the benchmark gate.

        Args:
            suite: List of benchmark tasks. Defaults to DEFAULT_BENCHMARK_SUITE.
            global_threshold: Global threshold to apply if task-specific not set.
        """
        self._suite = suite or DEFAULT_BENCHMARK_SUITE
        self._global_threshold = global_threshold or BenchmarkThreshold()
        self._task_map = {task.name: task for task in self._suite}

    @property
    def suite(self) -> list[BenchmarkTask]:
        """Get the benchmark suite."""
        return list(self._suite)

    def get_task(self, name: str) -> BenchmarkTask | None:
        """Get a task by name."""
        return self._task_map.get(name)

    def check_single_run(
        self,
        diag: RunDiagnostics,
        task: BenchmarkTask | None = None,
    ) -> tuple[bool, list[str]]:
        """Check if a single run meets thresholds.

        Args:
            diag: Diagnostics from the run.
            task: Optional task definition with specific thresholds.

        Returns:
            Tuple of (passed, list of failure reasons).
        """
        threshold = task.threshold if task else self._global_threshold
        failures: list[str] = []

        # Check improvement
        if not threshold.check_improvement(diag.improvement, diag.seed_fitness):
            failures.append(
                f"Improvement {diag.improvement:.4f} "
                f"({diag.improvement_percent:.1f}%) below threshold "
                f"({threshold.min_improvement:.4f}, {threshold.min_improvement_percent:.1f}%)"
            )

        # Check status
        if diag.status == RunStatus.NO_SIGNAL:
            failures.append("Run produced no signal (no improvement over baseline)")

        if diag.status == RunStatus.FAILED:
            failures.append("Run failed")

        # Check runtime
        if diag.elapsed_time_seconds > threshold.max_runtime_seconds:
            failures.append(
                f"Runtime {diag.elapsed_time_seconds:.1f}s exceeds "
                f"max {threshold.max_runtime_seconds:.1f}s"
            )

        return len(failures) == 0, failures

    def check_results(
        self,
        diagnostics: list[RunDiagnostics],
    ) -> GateResult:
        """Check multiple run results against benchmark gates.

        Args:
            diagnostics: List of diagnostics from benchmark runs.

        Returns:
            GateResult with pass/fail status and details.
        """
        result = GateResult()

        if not diagnostics:
            result.failures.append("No benchmark results provided")
            return result

        # Group by task
        by_task: dict[str, list[RunDiagnostics]] = {}
        for diag in diagnostics:
            task_name = diag.task_name
            if task_name not in by_task:
                by_task[task_name] = []
            by_task[task_name].append(diag)

        # Check each task
        all_passed = True
        total_runs = 0
        successful_runs = 0
        no_signal_runs = 0

        for task_name, task_diags in by_task.items():
            task = self.get_task(task_name)
            threshold = task.threshold if task else self._global_threshold

            # Task stats
            task_success = sum(1 for d in task_diags if d.status == RunStatus.SUCCESS)
            task_no_signal = sum(1 for d in task_diags if d.status == RunStatus.NO_SIGNAL)
            task_total = len(task_diags)
            task_improvements = [d.improvement for d in task_diags]
            avg_improvement = sum(task_improvements) / len(task_improvements) if task_improvements else 0

            success_rate = task_success / task_total if task_total > 0 else 0
            no_signal_rate = task_no_signal / task_total if task_total > 0 else 0

            task_passed = True
            task_failures: list[str] = []

            # Check success rate
            if success_rate < threshold.min_success_rate:
                task_passed = False
                task_failures.append(
                    f"Success rate {success_rate*100:.0f}% below minimum {threshold.min_success_rate*100:.0f}%"
                )

            # Check no-signal rate
            if no_signal_rate > threshold.max_no_signal_rate:
                task_passed = False
                task_failures.append(
                    f"No-signal rate {no_signal_rate*100:.0f}% above maximum {threshold.max_no_signal_rate*100:.0f}%"
                )

            # Check average improvement
            if not threshold.check_improvement(avg_improvement, task_diags[0].seed_fitness):
                task_passed = False
                task_failures.append(
                    f"Average improvement {avg_improvement:.4f} below threshold"
                )

            result.add_task_result(
                task_name=task_name,
                passed=task_passed,
                improvement=avg_improvement,
                status=RunStatus.SUCCESS if task_passed else RunStatus.NO_SIGNAL,
                details={
                    "runs": task_total,
                    "success_rate": success_rate,
                    "no_signal_rate": no_signal_rate,
                    "failures": task_failures,
                },
            )

            if not task_passed:
                all_passed = False
                for failure in task_failures:
                    result.failures.append(f"[{task_name}] {failure}")

            total_runs += task_total
            successful_runs += task_success
            no_signal_runs += task_no_signal

        # Overall summary
        result.summary = {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "no_signal_runs": no_signal_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
            "tasks_checked": len(by_task),
            "tasks_passed": sum(1 for t in result.task_results.values() if t["passed"]),
        }

        result.passed = all_passed

        # Add warnings for edge cases
        if total_runs < len(self._suite):
            result.warnings.append(
                f"Only {len(by_task)} of {len(self._suite)} benchmark tasks were run"
            )

        return result

    def format_report(self, gate_result: GateResult) -> str:
        """Format a gate result as a report string.

        Args:
            gate_result: Result to format.

        Returns:
            Formatted report string.
        """
        lines = []

        lines.append("=" * 60)
        lines.append("BENCHMARK GATE REPORT")
        lines.append("=" * 60)

        # Overall status
        status = "PASSED" if gate_result.passed else "FAILED"
        lines.append(f"\nStatus: {status}")

        # Summary
        summary = gate_result.summary
        lines.append("\n--- Summary ---")
        lines.append(f"Total Runs:      {summary.get('total_runs', 0)}")
        lines.append(f"Successful:      {summary.get('successful_runs', 0)}")
        lines.append(f"No Signal:       {summary.get('no_signal_runs', 0)}")
        lines.append(f"Success Rate:    {summary.get('success_rate', 0)*100:.0f}%")
        lines.append(f"Tasks Checked:   {summary.get('tasks_checked', 0)}")
        lines.append(f"Tasks Passed:    {summary.get('tasks_passed', 0)}")

        # Per-task results
        lines.append("\n--- Task Results ---")
        for task_name, task_result in gate_result.task_results.items():
            status_str = "PASS" if task_result["passed"] else "FAIL"
            lines.append(f"\n{task_name}: {status_str}")
            lines.append(f"  Improvement: {task_result['improvement']:.4f}")
            lines.append(f"  Runs: {task_result.get('runs', 'N/A')}")
            lines.append(f"  Success Rate: {task_result.get('success_rate', 0)*100:.0f}%")
            if task_result.get("failures"):
                for failure in task_result["failures"]:
                    lines.append(f"  - {failure}")

        # Failures
        if gate_result.failures:
            lines.append("\n--- Failures ---")
            for failure in gate_result.failures:
                lines.append(f"  x {failure}")

        # Warnings
        if gate_result.warnings:
            lines.append("\n--- Warnings ---")
            for warning in gate_result.warnings:
                lines.append(f"  ! {warning}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def get_default_suite() -> list[BenchmarkTask]:
    """Get the default benchmark suite."""
    return list(DEFAULT_BENCHMARK_SUITE)


def create_custom_gate(
    min_improvement: float = 0.05,
    min_success_rate: float = 0.6,
    max_no_signal_rate: float = 0.3,
) -> BenchmarkGate:
    """Create a gate with custom thresholds.

    Args:
        min_improvement: Minimum improvement required.
        min_success_rate: Minimum success rate.
        max_no_signal_rate: Maximum no-signal rate.

    Returns:
        Configured BenchmarkGate.
    """
    threshold = BenchmarkThreshold(
        min_improvement=min_improvement,
        min_success_rate=min_success_rate,
        max_no_signal_rate=max_no_signal_rate,
    )
    return BenchmarkGate(global_threshold=threshold)
