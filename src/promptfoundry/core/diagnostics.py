"""Diagnostics and reporting for optimization runs.

This module provides detailed reporting on optimization progress,
including no-signal detection, latency tracking, and run status analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TerminationReason(Enum):
    """Reason why an optimization run terminated."""

    MAX_GENERATIONS = "max_generations"
    PATIENCE_EXHAUSTED = "patience_exhausted"
    RUNTIME_BUDGET = "runtime_budget"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    UNKNOWN = "unknown"


class RunStatus(Enum):
    """Status of an optimization run."""

    SUCCESS = "success"  # Found improvement
    NO_SIGNAL = "no_signal"  # No improvement over seed
    PARTIAL = "partial"  # Some improvement but interrupted
    FAILED = "failed"  # Error during run


@dataclass
class GenerationMetrics:
    """Detailed metrics for a single generation.
    
    Attributes:
        generation: Generation number.
        best_fitness: Best fitness in this generation.
        average_fitness: Average fitness.
        fitness_std: Standard deviation of fitness values.
        population_size: Number of individuals evaluated.
        evaluation_time_ms: Total evaluation time in milliseconds.
        llm_calls: Number of LLM API calls made.
        cache_hits: Number of evaluations served from cache.
        timestamp: When this generation completed.
    """

    generation: int
    best_fitness: float
    average_fitness: float
    fitness_std: float = 0.0
    population_size: int = 0
    evaluation_time_ms: float = 0.0
    llm_calls: int = 0
    cache_hits: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def avg_latency_per_call_ms(self) -> float:
        """Average latency per LLM call in milliseconds."""
        total_calls = self.llm_calls + self.cache_hits
        if total_calls == 0:
            return 0.0
        return self.evaluation_time_ms / total_calls


@dataclass
class RunDiagnostics:
    """Complete diagnostics for an optimization run.
    
    Attributes:
        task_name: Name of the optimization task.
        seed_fitness: Fitness of the seed prompt (baseline).
        best_fitness: Best fitness achieved.
        improvement: Absolute improvement over seed.
        improvement_percent: Percentage improvement over seed.
        termination_reason: Why the run ended.
        status: Overall run status.
        total_generations: Number of generations completed.
        total_evaluations: Total prompt evaluations.
        total_llm_calls: Total LLM API calls (excluding cache).
        cache_hit_rate: Fraction of evaluations served from cache.
        elapsed_time_seconds: Total run time.
        avg_generation_time_ms: Average time per generation.
        generations: Per-generation metrics.
        warnings: List of warnings/issues detected.
    """

    task_name: str = ""
    seed_fitness: float = 0.0
    best_fitness: float = 0.0
    improvement: float = 0.0
    improvement_percent: float = 0.0
    termination_reason: TerminationReason = TerminationReason.UNKNOWN
    status: RunStatus = RunStatus.SUCCESS
    total_generations: int = 0
    total_evaluations: int = 0
    total_llm_calls: int = 0
    cache_hit_rate: float = 0.0
    elapsed_time_seconds: float = 0.0
    avg_generation_time_ms: float = 0.0
    generations: list[GenerationMetrics] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def analyze(
        cls,
        history_data: dict[str, Any],
        termination_reason: str = "unknown",
        elapsed_time: float = 0.0,
        total_llm_calls: int = 0,
        total_cache_hits: int = 0,
    ) -> "RunDiagnostics":
        """Analyze optimization history and produce diagnostics.
        
        Args:
            history_data: Loaded history JSON data.
            termination_reason: String reason for termination.
            elapsed_time: Total elapsed time in seconds.
            total_llm_calls: Total LLM API calls made.
            total_cache_hits: Total cache hits.
            
        Returns:
            RunDiagnostics with analysis results.
        """
        diag = cls()
        diag.task_name = history_data.get("task", history_data.get("task_name", ""))
        diag.elapsed_time_seconds = elapsed_time
        diag.total_llm_calls = total_llm_calls
        
        # Parse termination reason
        try:
            diag.termination_reason = TerminationReason(termination_reason)
        except ValueError:
            diag.termination_reason = TerminationReason.UNKNOWN
        
        # Process generation data
        generations = history_data.get("generations", [])
        if not generations:
            diag.status = RunStatus.FAILED
            diag.warnings.append("No generation data found")
            return diag
        
        diag.total_generations = len(generations)
        
        # Find seed fitness (first generation best)
        diag.seed_fitness = generations[0].get("best_fitness", 0.0)
        
        # Find best fitness
        diag.best_fitness = max(g.get("best_fitness", 0.0) for g in generations)
        
        # Calculate improvement
        diag.improvement = diag.best_fitness - diag.seed_fitness
        if diag.seed_fitness > 0:
            diag.improvement_percent = (diag.improvement / diag.seed_fitness) * 100
        elif diag.best_fitness > 0:
            diag.improvement_percent = 100.0  # From zero to something
        
        # Calculate total evaluations
        diag.total_evaluations = sum(
            g.get("population_size", 0) for g in generations
        )
        
        # Cache statistics
        total_calls = total_llm_calls + total_cache_hits
        if total_calls > 0:
            diag.cache_hit_rate = total_cache_hits / total_calls
        
        # Parse per-generation metrics
        for g in generations:
            metrics = GenerationMetrics(
                generation=g.get("generation", 0),
                best_fitness=g.get("best_fitness", 0.0),
                average_fitness=g.get("average_fitness", 0.0),
                population_size=g.get("population_size", 0),
                evaluation_time_ms=g.get("metadata", {}).get("evaluation_time_ms", 0.0),
                llm_calls=g.get("metadata", {}).get("llm_calls", 0),
                cache_hits=g.get("metadata", {}).get("cache_hits", 0),
                timestamp=g.get("timestamp", ""),
            )
            diag.generations.append(metrics)
        
        # Calculate average generation time
        if diag.generations:
            total_gen_time = sum(g.evaluation_time_ms for g in diag.generations)
            diag.avg_generation_time_ms = total_gen_time / len(diag.generations)
        
        # Determine run status
        diag.status = cls._determine_status(diag)
        
        # Add warnings
        diag.warnings = cls._detect_warnings(diag)
        
        return diag

    @staticmethod
    def _determine_status(diag: "RunDiagnostics") -> RunStatus:
        """Determine the overall status of a run."""
        if diag.termination_reason == TerminationReason.ERROR:
            return RunStatus.FAILED
        
        if diag.termination_reason == TerminationReason.INTERRUPTED:
            if diag.improvement > 0.001:
                return RunStatus.PARTIAL
            return RunStatus.NO_SIGNAL
        
        if diag.improvement <= 0.001:  # No meaningful improvement
            return RunStatus.NO_SIGNAL
        
        return RunStatus.SUCCESS

    @staticmethod
    def _detect_warnings(diag: "RunDiagnostics") -> list[str]:
        """Detect potential issues in the run."""
        warnings = []
        
        # No signal warning
        if diag.improvement <= 0.001:
            warnings.append(
                "NO SIGNAL: No improvement over seed prompt detected. "
                "Consider different mutation operators or longer runs."
            )
        
        # Low improvement warning
        if 0 < diag.improvement < 0.05 and diag.seed_fitness > 0:
            warnings.append(
                f"LOW SIGNAL: Only {diag.improvement_percent:.1f}% improvement achieved. "
                "Results may not be statistically significant."
            )
        
        # Early termination warning
        if diag.termination_reason == TerminationReason.PATIENCE_EXHAUSTED:
            if diag.total_generations < 5:
                warnings.append(
                    "EARLY STOP: Run ended due to patience after few generations. "
                    "Consider increasing patience or changing strategy."
                )
        
        # Budget exhausted warning
        if diag.termination_reason == TerminationReason.RUNTIME_BUDGET:
            warnings.append(
                "BUDGET LIMIT: Run stopped due to runtime budget. "
                "Results may be suboptimal."
            )
        
        # High cache hit rate (might indicate duplicate prompts)
        if diag.cache_hit_rate > 0.8:
            warnings.append(
                f"HIGH CACHE: {diag.cache_hit_rate*100:.0f}% cache hit rate suggests "
                "many duplicate prompts. Consider more diverse mutations."
            )
        
        # Fitness plateau detection
        if len(diag.generations) >= 3:
            last_3 = diag.generations[-3:]
            if all(abs(g.best_fitness - diag.best_fitness) < 0.001 for g in last_3):
                warnings.append(
                    "PLATEAU: Best fitness hasn't improved in last 3 generations."
                )
        
        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostics to a dictionary for serialization."""
        return {
            "task_name": self.task_name,
            "seed_fitness": self.seed_fitness,
            "best_fitness": self.best_fitness,
            "improvement": self.improvement,
            "improvement_percent": self.improvement_percent,
            "termination_reason": self.termination_reason.value,
            "status": self.status.value,
            "total_generations": self.total_generations,
            "total_evaluations": self.total_evaluations,
            "total_llm_calls": self.total_llm_calls,
            "cache_hit_rate": self.cache_hit_rate,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "avg_generation_time_ms": self.avg_generation_time_ms,
            "warnings": self.warnings,
            "generations": [
                {
                    "generation": g.generation,
                    "best_fitness": g.best_fitness,
                    "average_fitness": g.average_fitness,
                    "evaluation_time_ms": g.evaluation_time_ms,
                    "llm_calls": g.llm_calls,
                    "cache_hits": g.cache_hits,
                }
                for g in self.generations
            ],
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSummary:
    """Summary of multiple benchmark runs for comparison.
    
    Attributes:
        runs: List of run diagnostics to summarize.
        task_comparison: Comparison across different tasks.
        operator_comparison: Comparison across different mutation operators.
    """

    runs: list[RunDiagnostics] = field(default_factory=list)

    def add_run(self, diag: RunDiagnostics) -> None:
        """Add a run to the summary."""
        self.runs.append(diag)

    @property
    def total_runs(self) -> int:
        """Total number of runs."""
        return len(self.runs)

    @property
    def successful_runs(self) -> int:
        """Number of successful runs."""
        return sum(1 for r in self.runs if r.status == RunStatus.SUCCESS)

    @property
    def no_signal_runs(self) -> int:
        """Number of runs with no signal."""
        return sum(1 for r in self.runs if r.status == RunStatus.NO_SIGNAL)

    @property
    def average_improvement(self) -> float:
        """Average improvement across all runs."""
        if not self.runs:
            return 0.0
        return sum(r.improvement for r in self.runs) / len(self.runs)

    @property
    def average_runtime(self) -> float:
        """Average runtime in seconds."""
        if not self.runs:
            return 0.0
        return sum(r.elapsed_time_seconds for r in self.runs) / len(self.runs)

    def by_task(self) -> dict[str, list[RunDiagnostics]]:
        """Group runs by task name."""
        grouped: dict[str, list[RunDiagnostics]] = {}
        for run in self.runs:
            task = run.task_name or "unknown"
            if task not in grouped:
                grouped[task] = []
            grouped[task].append(run)
        return grouped

    def by_status(self) -> dict[RunStatus, list[RunDiagnostics]]:
        """Group runs by status."""
        grouped: dict[RunStatus, list[RunDiagnostics]] = {}
        for run in self.runs:
            if run.status not in grouped:
                grouped[run.status] = []
            grouped[run.status].append(run)
        return grouped

    def task_stats(self) -> dict[str, dict[str, float]]:
        """Calculate statistics per task.
        
        Returns:
            Dict mapping task name to stats dict.
        """
        stats: dict[str, dict[str, float]] = {}
        for task, runs in self.by_task().items():
            improvements = [r.improvement for r in runs]
            runtimes = [r.elapsed_time_seconds for r in runs]
            success_rate = sum(1 for r in runs if r.status == RunStatus.SUCCESS) / len(runs)
            
            stats[task] = {
                "num_runs": len(runs),
                "success_rate": success_rate,
                "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
                "max_improvement": max(improvements) if improvements else 0,
                "min_improvement": min(improvements) if improvements else 0,
                "avg_runtime_s": sum(runtimes) / len(runtimes) if runtimes else 0,
            }
        return stats

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "no_signal_runs": self.no_signal_runs,
            "average_improvement": self.average_improvement,
            "average_runtime_s": self.average_runtime,
            "task_stats": self.task_stats(),
            "runs": [r.to_dict() for r in self.runs],
        }


def format_diagnostics_report(diag: RunDiagnostics) -> str:
    """Format diagnostics as a human-readable report.
    
    Args:
        diag: RunDiagnostics to format.
        
    Returns:
        Formatted report string.
    """
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append(f"OPTIMIZATION DIAGNOSTICS REPORT")
    lines.append("=" * 60)
    
    # Summary
    lines.append(f"\nTask: {diag.task_name}")
    lines.append(f"Status: {diag.status.value.upper()}")
    lines.append(f"Termination: {diag.termination_reason.value}")
    
    # Scores
    lines.append(f"\n--- Fitness ---")
    lines.append(f"Seed Fitness:    {diag.seed_fitness:.4f}")
    lines.append(f"Best Fitness:    {diag.best_fitness:.4f}")
    lines.append(f"Improvement:     {diag.improvement:+.4f} ({diag.improvement_percent:+.1f}%)")
    
    # Statistics
    lines.append(f"\n--- Statistics ---")
    lines.append(f"Generations:     {diag.total_generations}")
    lines.append(f"Evaluations:     {diag.total_evaluations}")
    lines.append(f"LLM Calls:       {diag.total_llm_calls}")
    lines.append(f"Cache Hit Rate:  {diag.cache_hit_rate*100:.1f}%")
    lines.append(f"Elapsed Time:    {diag.elapsed_time_seconds:.2f}s")
    lines.append(f"Avg Gen Time:    {diag.avg_generation_time_ms:.1f}ms")
    
    # Warnings
    if diag.warnings:
        lines.append(f"\n--- Warnings ---")
        for warning in diag.warnings:
            lines.append(f"⚠ {warning}")
    
    # Per-generation latency
    if diag.generations:
        lines.append(f"\n--- Per-Generation ---")
        lines.append(f"{'Gen':>4} {'Best':>8} {'Avg':>8} {'Time(ms)':>10} {'Calls':>6}")
        lines.append("-" * 40)
        for g in diag.generations[:10]:  # Show first 10
            lines.append(
                f"{g.generation:>4} {g.best_fitness:>8.4f} {g.average_fitness:>8.4f} "
                f"{g.evaluation_time_ms:>10.1f} {g.llm_calls:>6}"
            )
        if len(diag.generations) > 10:
            lines.append(f"... ({len(diag.generations) - 10} more generations)")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def format_benchmark_summary(summary: BenchmarkSummary) -> str:
    """Format benchmark summary as a human-readable report.
    
    Args:
        summary: BenchmarkSummary to format.
        
    Returns:
        Formatted summary string.
    """
    lines = []
    
    lines.append("=" * 60)
    lines.append("BENCHMARK SUMMARY REPORT")
    lines.append("=" * 60)
    
    lines.append(f"\n--- Overview ---")
    lines.append(f"Total Runs:        {summary.total_runs}")
    lines.append(f"Successful:        {summary.successful_runs}")
    lines.append(f"No Signal:         {summary.no_signal_runs}")
    lines.append(f"Avg Improvement:   {summary.average_improvement:.4f}")
    lines.append(f"Avg Runtime:       {summary.average_runtime:.2f}s")
    
    # Per-task breakdown
    task_stats = summary.task_stats()
    if task_stats:
        lines.append(f"\n--- Per-Task Breakdown ---")
        for task, stats in task_stats.items():
            lines.append(f"\n  {task}:")
            lines.append(f"    Runs:          {stats['num_runs']:.0f}")
            lines.append(f"    Success Rate:  {stats['success_rate']*100:.0f}%")
            lines.append(f"    Avg Improve:   {stats['avg_improvement']:.4f}")
            lines.append(f"    Max Improve:   {stats['max_improvement']:.4f}")
            lines.append(f"    Avg Runtime:   {stats['avg_runtime_s']:.2f}s")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
