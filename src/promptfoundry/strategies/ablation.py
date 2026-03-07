"""Ablation utilities for evaluating mutation operator quality.

This module provides tools to measure the effectiveness of individual
mutation operators in isolation and in combination, enabling data-driven
selection of operator configurations.

MVP 3 Feature: Ablation utilities for operator quality assessment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class OperatorMetrics:
    """Metrics for a single mutation operator.

    Attributes:
        name: Operator name.
        attempts: Total number of times operator was applied.
        successes: Number of times operator improved fitness.
        failures: Number of times operator decreased fitness.
        neutral: Number of times operator had no effect.
        total_improvement: Sum of positive fitness deltas.
        total_regression: Sum of negative fitness deltas.
        best_improvement: Best single improvement.
        worst_regression: Worst single regression.
    """

    name: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    neutral: int = 0
    total_improvement: float = 0.0
    total_regression: float = 0.0
    best_improvement: float = 0.0
    worst_regression: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return self.failures / self.attempts if self.attempts > 0 else 0.0

    @property
    def net_improvement(self) -> float:
        """Calculate net improvement (improvement - regression)."""
        return self.total_improvement + self.total_regression

    @property
    def avg_improvement_when_successful(self) -> float:
        """Calculate average improvement when the operator succeeds."""
        return self.total_improvement / self.successes if self.successes > 0 else 0.0

    @property
    def effectiveness_score(self) -> float:
        """Calculate overall effectiveness score.

        Combines success rate, average improvement, and consistency.
        Range: -1 to +2 (negative = harmful, positive = beneficial).
        """
        if self.attempts == 0:
            return 0.0

        success_component = self.success_rate  # 0-1
        improvement_component = min(1.0, self.avg_improvement_when_successful)  # 0-1
        consistency = 1 - self.failure_rate  # 0-1

        return success_component + improvement_component * consistency - self.failure_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "neutral": self.neutral,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "total_improvement": round(self.total_improvement, 4),
            "total_regression": round(self.total_regression, 4),
            "net_improvement": round(self.net_improvement, 4),
            "best_improvement": round(self.best_improvement, 4),
            "worst_regression": round(self.worst_regression, 4),
            "avg_improvement_when_successful": round(self.avg_improvement_when_successful, 4),
            "effectiveness_score": round(self.effectiveness_score, 4),
        }


@dataclass
class AblationResult:
    """Result of an ablation study.

    Attributes:
        task_name: Name of the task used for ablation.
        baseline_fitness: Fitness without any mutations.
        operator_metrics: Metrics for each operator tested.
        best_combination: Best operator combination found.
        worst_operators: Operators that consistently hurt performance.
        recommended_weights: Recommended operator weights.
        timestamp: When the ablation was run.
        config: Configuration used for the ablation.
    """

    task_name: str
    baseline_fitness: float = 0.0
    operator_metrics: dict[str, OperatorMetrics] = field(default_factory=dict)
    best_combination: list[str] = field(default_factory=list)
    worst_operators: list[str] = field(default_factory=list)
    recommended_weights: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "baseline_fitness": self.baseline_fitness,
            "operator_metrics": {
                name: metrics.to_dict() for name, metrics in self.operator_metrics.items()
            },
            "best_combination": self.best_combination,
            "worst_operators": self.worst_operators,
            "recommended_weights": self.recommended_weights,
            "timestamp": self.timestamp,
            "config": self.config,
        }

    def save(self, path: Path) -> None:
        """Save ablation result to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> AblationResult:
        """Load ablation result from JSON file."""
        data = json.loads(path.read_text())

        operator_metrics = {}
        for name, metrics_data in data.get("operator_metrics", {}).items():
            operator_metrics[name] = OperatorMetrics(
                name=metrics_data["name"],
                attempts=metrics_data["attempts"],
                successes=metrics_data["successes"],
                failures=metrics_data["failures"],
                neutral=metrics_data["neutral"],
                total_improvement=metrics_data["total_improvement"],
                total_regression=metrics_data["total_regression"],
                best_improvement=metrics_data["best_improvement"],
                worst_regression=metrics_data["worst_regression"],
            )

        return cls(
            task_name=data["task_name"],
            baseline_fitness=data["baseline_fitness"],
            operator_metrics=operator_metrics,
            best_combination=data.get("best_combination", []),
            worst_operators=data.get("worst_operators", []),
            recommended_weights=data.get("recommended_weights", {}),
            timestamp=data.get("timestamp", ""),
            config=data.get("config", {}),
        )


class AblationTracker:
    """Tracks operator performance during optimization for ablation analysis.

    Collects per-operator statistics during an optimization run to enable
    post-hoc analysis of operator effectiveness.
    """

    def __init__(self, task_name: str = "unknown") -> None:
        """Initialize ablation tracker.

        Args:
            task_name: Name of the task being optimized.
        """
        self.task_name = task_name
        self._metrics: dict[str, OperatorMetrics] = {}
        self._generations: list[dict[str, Any]] = []
        self._baseline_fitness: float = 0.0

    def set_baseline(self, fitness: float) -> None:
        """Set baseline fitness (seed prompt fitness).

        Args:
            fitness: Baseline fitness score.
        """
        self._baseline_fitness = fitness

    def record_mutation(
        self,
        operator_name: str,
        parent_fitness: float,
        child_fitness: float,
    ) -> None:
        """Record the result of a mutation operation.

        Args:
            operator_name: Name of the mutation operator.
            parent_fitness: Fitness of parent prompt.
            child_fitness: Fitness of mutated prompt.
        """
        if operator_name not in self._metrics:
            self._metrics[operator_name] = OperatorMetrics(name=operator_name)

        metrics = self._metrics[operator_name]
        metrics.attempts += 1

        delta = child_fitness - parent_fitness

        if delta > 0.001:  # Improvement threshold
            metrics.successes += 1
            metrics.total_improvement += delta
            metrics.best_improvement = max(metrics.best_improvement, delta)
        elif delta < -0.001:  # Regression threshold
            metrics.failures += 1
            metrics.total_regression += delta
            metrics.worst_regression = min(metrics.worst_regression, delta)
        else:
            metrics.neutral += 1

    def record_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        operator_counts: dict[str, int],
    ) -> None:
        """Record generation-level statistics.

        Args:
            generation: Generation number.
            best_fitness: Best fitness in generation.
            avg_fitness: Average fitness in generation.
            operator_counts: Count of each operator used in generation.
        """
        self._generations.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "operator_counts": operator_counts,
            }
        )

    def get_metrics(self) -> dict[str, OperatorMetrics]:
        """Get all operator metrics."""
        return self._metrics.copy()

    def generate_result(self) -> AblationResult:
        """Generate ablation result from collected data.

        Returns:
            Complete ablation result with recommendations.
        """
        # Identify worst operators (negative effectiveness)
        worst_operators = [
            name
            for name, metrics in self._metrics.items()
            if metrics.effectiveness_score < 0 and metrics.attempts >= 5
        ]

        # Identify best operators
        sorted_operators = sorted(
            self._metrics.items(),
            key=lambda x: x[1].effectiveness_score,
            reverse=True,
        )
        best_combination = [
            name
            for name, metrics in sorted_operators[:5]
            if metrics.effectiveness_score > 0
        ]

        # Calculate recommended weights
        recommended_weights = self._calculate_recommended_weights()

        return AblationResult(
            task_name=self.task_name,
            baseline_fitness=self._baseline_fitness,
            operator_metrics=self._metrics,
            best_combination=best_combination,
            worst_operators=worst_operators,
            recommended_weights=recommended_weights,
            config={"num_generations": len(self._generations)},
        )

    def _calculate_recommended_weights(self) -> dict[str, float]:
        """Calculate recommended operator weights based on effectiveness."""
        weights = {}

        # Find max effectiveness for normalization
        max_effectiveness = max(
            (m.effectiveness_score for m in self._metrics.values()),
            default=1.0,
        )
        if max_effectiveness <= 0:
            max_effectiveness = 1.0

        for name, metrics in self._metrics.items():
            if metrics.attempts < 3:
                # Not enough data, use neutral weight
                weights[name] = 1.0
                continue

            # Scale effectiveness to weight (0.5 to 2.5)
            normalized = metrics.effectiveness_score / max_effectiveness
            weight = 1.0 + normalized * 1.5

            # Penalize high-failure operators
            if metrics.failure_rate > 0.5:
                weight *= 0.7

            weights[name] = max(0.3, min(2.5, weight))

        return weights

    def get_summary(self) -> str:
        """Generate human-readable summary of ablation results.

        Returns:
            Formatted summary string.
        """
        lines = [
            f"Ablation Summary for: {self.task_name}",
            f"Baseline Fitness: {self._baseline_fitness:.4f}",
            f"Operators Tested: {len(self._metrics)}",
            f"Generations Recorded: {len(self._generations)}",
            "",
            "Operator Performance (sorted by effectiveness):",
            "-" * 70,
        ]

        sorted_metrics = sorted(
            self._metrics.values(),
            key=lambda x: x.effectiveness_score,
            reverse=True,
        )

        for metrics in sorted_metrics:
            lines.append(
                f"  {metrics.name:<30} "
                f"success={metrics.success_rate:.2%} "
                f"effect={metrics.effectiveness_score:+.3f} "
                f"attempts={metrics.attempts}"
            )

        result = self.generate_result()
        if result.best_combination:
            lines.extend(
                ["", "Recommended Operators:", *[f"  - {op}" for op in result.best_combination]]
            )

        if result.worst_operators:
            lines.extend(
                ["", "Operators to Avoid:", *[f"  - {op}" for op in result.worst_operators]]
            )

        return "\n".join(lines)


class AblationStudy:
    """Conducts ablation studies to evaluate operator configurations.

    Provides methods to:
    - Run leave-one-out ablation (disable one operator at a time)
    - Run isolated operator tests (enable only one operator at a time)
    - Compare operator combinations
    """

    def __init__(
        self,
        operators: list[str],
        baseline_fitness: float = 0.0,
    ) -> None:
        """Initialize ablation study.

        Args:
            operators: List of operator names to study.
            baseline_fitness: Baseline fitness for comparison.
        """
        self.operators = operators
        self.baseline_fitness = baseline_fitness
        self._results: dict[str, dict[str, Any]] = {}

    def record_run(
        self,
        config_name: str,
        enabled_operators: list[str],
        final_fitness: float,
        improvement: float,
        runtime_seconds: float,
    ) -> None:
        """Record results of a configuration run.

        Args:
            config_name: Name of this configuration.
            enabled_operators: Which operators were enabled.
            final_fitness: Final fitness achieved.
            improvement: Improvement over baseline.
            runtime_seconds: How long the run took.
        """
        self._results[config_name] = {
            "enabled_operators": enabled_operators,
            "disabled_operators": [op for op in self.operators if op not in enabled_operators],
            "final_fitness": final_fitness,
            "improvement": improvement,
            "runtime_seconds": runtime_seconds,
        }

    def get_leave_one_out_analysis(self) -> dict[str, float]:
        """Analyze leave-one-out results.

        Returns:
            Dictionary of operator -> impact (positive = operator helps).
        """
        # Find the "all operators" baseline
        all_ops_fitness = None
        for name, result in self._results.items():
            if len(result["disabled_operators"]) == 0:
                all_ops_fitness = result["final_fitness"]
                break

        if all_ops_fitness is None:
            return {}

        # Calculate impact of each operator
        impacts = {}
        for name, result in self._results.items():
            disabled = result["disabled_operators"]
            if len(disabled) == 1:
                # This is a leave-one-out run
                operator = disabled[0]
                # Impact = how much worse it is without this operator
                impacts[operator] = all_ops_fitness - result["final_fitness"]

        return impacts

    def get_isolated_analysis(self) -> dict[str, float]:
        """Analyze isolated operator results.

        Returns:
            Dictionary of operator -> isolated improvement.
        """
        isolated = {}
        for name, result in self._results.items():
            enabled = result["enabled_operators"]
            if len(enabled) == 1:
                operator = enabled[0]
                isolated[operator] = result["improvement"]

        return isolated

    def generate_report(self) -> str:
        """Generate ablation study report.

        Returns:
            Formatted report string.
        """
        lines = [
            "Ablation Study Report",
            "=" * 60,
            f"Baseline Fitness: {self.baseline_fitness:.4f}",
            f"Operators Studied: {len(self.operators)}",
            f"Configurations Tested: {len(self._results)}",
            "",
        ]

        # Leave-one-out analysis
        loo_impacts = self.get_leave_one_out_analysis()
        if loo_impacts:
            lines.extend(["Leave-One-Out Analysis:", "-" * 40])
            sorted_impacts = sorted(loo_impacts.items(), key=lambda x: x[1], reverse=True)
            for operator, impact in sorted_impacts:
                symbol = "+" if impact > 0 else ""
                lines.append(f"  {operator:<30} {symbol}{impact:.4f}")
            lines.append("")

        # Isolated analysis
        isolated = self.get_isolated_analysis()
        if isolated:
            lines.extend(["Isolated Operator Analysis:", "-" * 40])
            sorted_isolated = sorted(isolated.items(), key=lambda x: x[1], reverse=True)
            for operator, improvement in sorted_isolated:
                symbol = "+" if improvement > 0 else ""
                lines.append(f"  {operator:<30} {symbol}{improvement:.4f}")
            lines.append("")

        # Best configurations
        if self._results:
            lines.extend(["Best Configurations:", "-" * 40])
            sorted_configs = sorted(
                self._results.items(),
                key=lambda x: x[1]["final_fitness"],
                reverse=True,
            )[:5]
            for name, result in sorted_configs:
                lines.append(f"  {name}: fitness={result['final_fitness']:.4f}")

        return "\n".join(lines)
