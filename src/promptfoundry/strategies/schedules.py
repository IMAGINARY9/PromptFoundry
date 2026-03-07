"""Adaptive mutation scheduling for evolutionary optimization.

This module provides dynamic adjustment of mutation rates and operator
selection based on optimization progress and population state.

MVP 3 Feature: Adaptive mutation schedules that respond to convergence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SchedulePhase(Enum):
    """Phase of the optimization run."""

    EXPLORATION = "exploration"  # Early: aggressive mutation
    BALANCED = "balanced"  # Middle: balanced exploration/exploitation
    EXPLOITATION = "exploitation"  # Late: fine-tuning
    CONVERGED = "converged"  # Stalled: increase diversity


@dataclass
class MutationScheduleState:
    """Current state of the mutation schedule.

    Attributes:
        generation: Current generation number.
        max_generations: Maximum planned generations.
        current_mutation_rate: Current mutation rate (0-1).
        current_crossover_rate: Current crossover rate (0-1).
        phase: Current optimization phase.
        stall_count: Number of generations without improvement.
        best_fitness: Best fitness seen so far.
        avg_fitness: Average fitness of current population.
        diversity_score: Current population diversity (0-1).
    """

    generation: int = 0
    max_generations: int = 20
    current_mutation_rate: float = 0.3
    current_crossover_rate: float = 0.7
    phase: SchedulePhase = SchedulePhase.EXPLORATION
    stall_count: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity_score: float = 1.0
    operator_weights: dict[str, float] = field(default_factory=dict)

    def progress_ratio(self) -> float:
        """Return progress through optimization (0-1)."""
        if self.max_generations <= 0:
            return 0.0
        return min(1.0, self.generation / self.max_generations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "max_generations": self.max_generations,
            "mutation_rate": self.current_mutation_rate,
            "crossover_rate": self.current_crossover_rate,
            "phase": self.phase.value,
            "stall_count": self.stall_count,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "diversity_score": self.diversity_score,
        }


class MutationSchedule(ABC):
    """Abstract base class for mutation schedules."""

    @abstractmethod
    def update(
        self,
        state: MutationScheduleState,
        fitness_scores: list[float],
    ) -> MutationScheduleState:
        """Update schedule state based on generation results.

        Args:
            state: Current schedule state.
            fitness_scores: Fitness scores from the generation.

        Returns:
            Updated schedule state.
        """
        ...

    @abstractmethod
    def get_mutation_rate(self, state: MutationScheduleState) -> float:
        """Get current mutation rate.

        Args:
            state: Current schedule state.

        Returns:
            Mutation rate (0-1).
        """
        ...

    @abstractmethod
    def get_crossover_rate(self, state: MutationScheduleState) -> float:
        """Get current crossover rate.

        Args:
            state: Current schedule state.

        Returns:
            Crossover rate (0-1).
        """
        ...


class ConstantSchedule(MutationSchedule):
    """Constant mutation rate schedule (baseline)."""

    def __init__(
        self,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
    ) -> None:
        """Initialize constant schedule.

        Args:
            mutation_rate: Fixed mutation rate.
            crossover_rate: Fixed crossover rate.
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def update(
        self,
        state: MutationScheduleState,
        fitness_scores: list[float],
    ) -> MutationScheduleState:
        """Update state (no changes for constant schedule)."""
        state.generation += 1
        state.current_mutation_rate = self.mutation_rate
        state.current_crossover_rate = self.crossover_rate

        if fitness_scores:
            state.avg_fitness = sum(fitness_scores) / len(fitness_scores)
            current_best = max(fitness_scores)
            if current_best > state.best_fitness:
                state.best_fitness = current_best
                state.stall_count = 0
            else:
                state.stall_count += 1

        return state

    def get_mutation_rate(self, state: MutationScheduleState) -> float:
        """Return constant mutation rate."""
        return self.mutation_rate

    def get_crossover_rate(self, state: MutationScheduleState) -> float:
        """Return constant crossover rate."""
        return self.crossover_rate


class LinearDecaySchedule(MutationSchedule):
    """Linear decay from high to low mutation rate over generations."""

    def __init__(
        self,
        initial_mutation_rate: float = 0.5,
        final_mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ) -> None:
        """Initialize linear decay schedule.

        Args:
            initial_mutation_rate: Starting mutation rate.
            final_mutation_rate: Ending mutation rate.
            crossover_rate: Fixed crossover rate.
        """
        self.initial_mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.crossover_rate = crossover_rate

    def update(
        self,
        state: MutationScheduleState,
        fitness_scores: list[float],
    ) -> MutationScheduleState:
        """Update state with linear decay."""
        state.generation += 1
        state.current_mutation_rate = self.get_mutation_rate(state)
        state.current_crossover_rate = self.crossover_rate

        progress = state.progress_ratio()
        if progress < 0.33:
            state.phase = SchedulePhase.EXPLORATION
        elif progress < 0.66:
            state.phase = SchedulePhase.BALANCED
        else:
            state.phase = SchedulePhase.EXPLOITATION

        if fitness_scores:
            state.avg_fitness = sum(fitness_scores) / len(fitness_scores)
            current_best = max(fitness_scores)
            if current_best > state.best_fitness:
                state.best_fitness = current_best
                state.stall_count = 0
            else:
                state.stall_count += 1

        return state

    def get_mutation_rate(self, state: MutationScheduleState) -> float:
        """Calculate linearly decayed mutation rate."""
        progress = state.progress_ratio()
        rate = self.initial_mutation_rate - progress * (
            self.initial_mutation_rate - self.final_mutation_rate
        )
        return max(self.final_mutation_rate, min(self.initial_mutation_rate, rate))

    def get_crossover_rate(self, state: MutationScheduleState) -> float:
        """Return constant crossover rate."""
        return self.crossover_rate


class AdaptiveSchedule(MutationSchedule):
    """Adaptive schedule that responds to optimization dynamics.

    Increases mutation when:
    - Fitness stalls (no improvement for N generations)
    - Diversity drops too low

    Decreases mutation when:
    - Fitness is improving consistently
    - Late in optimization (fine-tuning)
    """

    def __init__(
        self,
        base_mutation_rate: float = 0.3,
        min_mutation_rate: float = 0.1,
        max_mutation_rate: float = 0.7,
        crossover_rate: float = 0.7,
        stall_threshold: int = 3,
        diversity_threshold: float = 0.5,
        adaptation_rate: float = 0.1,
    ) -> None:
        """Initialize adaptive schedule.

        Args:
            base_mutation_rate: Starting/baseline mutation rate.
            min_mutation_rate: Minimum allowed mutation rate.
            max_mutation_rate: Maximum allowed mutation rate.
            crossover_rate: Base crossover rate.
            stall_threshold: Generations without improvement before increasing mutation.
            diversity_threshold: Diversity level below which to increase mutation.
            adaptation_rate: How quickly to adapt rates.
        """
        self.base_mutation_rate = base_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.crossover_rate = crossover_rate
        self.stall_threshold = stall_threshold
        self.diversity_threshold = diversity_threshold
        self.adaptation_rate = adaptation_rate

    def update(
        self,
        state: MutationScheduleState,
        fitness_scores: list[float],
    ) -> MutationScheduleState:
        """Update state with adaptive adjustments."""
        state.generation += 1

        if fitness_scores:
            state.avg_fitness = sum(fitness_scores) / len(fitness_scores)
            current_best = max(fitness_scores)
            if current_best > state.best_fitness:
                state.best_fitness = current_best
                state.stall_count = 0
            else:
                state.stall_count += 1

        # Determine phase
        state.phase = self._determine_phase(state)

        # Adapt mutation rate
        state.current_mutation_rate = self._adapt_mutation_rate(state)
        state.current_crossover_rate = self._adapt_crossover_rate(state)

        return state

    def _determine_phase(self, state: MutationScheduleState) -> SchedulePhase:
        """Determine current optimization phase."""
        progress = state.progress_ratio()

        # Check for convergence/stall
        if state.stall_count >= self.stall_threshold:
            return SchedulePhase.CONVERGED

        # Check diversity
        if state.diversity_score < self.diversity_threshold:
            return SchedulePhase.CONVERGED

        # Progress-based phases
        if progress < 0.25:
            return SchedulePhase.EXPLORATION
        elif progress < 0.75:
            return SchedulePhase.BALANCED
        else:
            return SchedulePhase.EXPLOITATION

    def _adapt_mutation_rate(self, state: MutationScheduleState) -> float:
        """Calculate adapted mutation rate."""
        rate = state.current_mutation_rate

        if state.phase == SchedulePhase.CONVERGED:
            # Increase mutation to escape local optima
            rate = min(
                self.max_mutation_rate,
                rate + self.adaptation_rate * 2,
            )
        elif state.phase == SchedulePhase.EXPLORATION:
            # Higher mutation for exploration
            target = self.base_mutation_rate + 0.15
            rate = rate + self.adaptation_rate * (target - rate)
        elif state.phase == SchedulePhase.EXPLOITATION:
            # Lower mutation for fine-tuning
            target = self.base_mutation_rate - 0.1
            rate = rate + self.adaptation_rate * (target - rate)
        else:
            # Move toward base rate
            rate = rate + self.adaptation_rate * (self.base_mutation_rate - rate)

        return max(self.min_mutation_rate, min(self.max_mutation_rate, rate))

    def _adapt_crossover_rate(self, state: MutationScheduleState) -> float:
        """Calculate adapted crossover rate."""
        rate = self.crossover_rate

        # Reduce crossover when converged (focus on mutation)
        if state.phase == SchedulePhase.CONVERGED:
            rate = max(0.3, rate - 0.2)
        elif state.phase == SchedulePhase.EXPLOITATION:
            rate = min(0.9, rate + 0.1)

        return rate

    def get_mutation_rate(self, state: MutationScheduleState) -> float:
        """Get current mutation rate."""
        return state.current_mutation_rate

    def get_crossover_rate(self, state: MutationScheduleState) -> float:
        """Get current crossover rate."""
        return state.current_crossover_rate


class OperatorAdaptiveSchedule(AdaptiveSchedule):
    """Adaptive schedule with per-operator weight adjustment.

    Extends AdaptiveSchedule to also adjust weights of individual
    mutation operators based on their historical performance.
    """

    def __init__(
        self,
        *args: Any,
        operator_learning_rate: float = 0.2,
        min_operator_weight: float = 0.3,
        max_operator_weight: float = 3.0,
        **kwargs: Any,
    ) -> None:
        """Initialize operator-adaptive schedule.

        Args:
            *args: Arguments for AdaptiveSchedule.
            operator_learning_rate: How quickly to adjust operator weights.
            min_operator_weight: Minimum operator weight.
            max_operator_weight: Maximum operator weight.
            **kwargs: Keyword arguments for AdaptiveSchedule.
        """
        super().__init__(*args, **kwargs)
        self.operator_learning_rate = operator_learning_rate
        self.min_operator_weight = min_operator_weight
        self.max_operator_weight = max_operator_weight
        self._operator_stats: dict[str, dict[str, float]] = {}

    def record_operator_result(
        self,
        operator_name: str,
        fitness_delta: float,
        was_improvement: bool,
    ) -> None:
        """Record result of an operator application.

        Args:
            operator_name: Name of the mutation operator.
            fitness_delta: Change in fitness (positive = improvement).
            was_improvement: Whether this resulted in improvement.
        """
        stats = self._operator_stats.setdefault(
            operator_name,
            {"attempts": 0, "wins": 0, "total_delta": 0, "weight": 1.0},
        )
        stats["attempts"] += 1
        stats["total_delta"] += fitness_delta
        if was_improvement:
            stats["wins"] += 1

    def get_operator_weights(self) -> dict[str, float]:
        """Get current operator weights.

        Returns:
            Dictionary of operator name -> weight.
        """
        weights = {}
        for name, stats in self._operator_stats.items():
            attempts = stats["attempts"]
            if attempts < 3:
                # Not enough data, use default
                weights[name] = 1.0
                continue

            win_rate = stats["wins"] / attempts
            avg_delta = stats["total_delta"] / attempts

            # Weight based on win rate and average improvement
            weight = 1.0 + (win_rate * self.operator_learning_rate)
            if avg_delta > 0:
                weight += avg_delta * self.operator_learning_rate
            elif avg_delta < 0:
                weight /= 1.0 + abs(avg_delta) * self.operator_learning_rate

            weights[name] = max(
                self.min_operator_weight,
                min(self.max_operator_weight, weight),
            )

        return weights

    def update(
        self,
        state: MutationScheduleState,
        fitness_scores: list[float],
    ) -> MutationScheduleState:
        """Update state including operator weights."""
        state = super().update(state, fitness_scores)
        state.operator_weights = self.get_operator_weights()
        return state


def create_schedule(
    schedule_type: str = "adaptive",
    **kwargs: Any,
) -> MutationSchedule:
    """Factory function to create mutation schedules.

    Args:
        schedule_type: Type of schedule ("constant", "linear", "adaptive", "operator").
        **kwargs: Schedule-specific parameters.

    Returns:
        Configured mutation schedule.
    """
    schedules: dict[str, type[MutationSchedule]] = {
        "constant": ConstantSchedule,
        "linear": LinearDecaySchedule,
        "adaptive": AdaptiveSchedule,
        "operator": OperatorAdaptiveSchedule,
    }

    schedule_class = schedules.get(schedule_type.lower())
    if not schedule_class:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    result: MutationSchedule = schedule_class(**kwargs)
    return result
