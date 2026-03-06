"""Base strategy class for optimization algorithms.

This module provides the abstract base class that all optimization
strategies must extend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from promptfoundry.core.history import OptimizationHistory
    from promptfoundry.core.population import Population
    from promptfoundry.core.prompt import Prompt


@dataclass
class StrategyConfig:
    """Base configuration for optimization strategies.

    Attributes:
        population_size: Number of individuals in population.
        max_generations: Maximum generations to run.
        patience: Generations without improvement before early stopping.
        seed: Random seed for reproducibility.
        extra: Strategy-specific configuration.
    """

    population_size: int = 20
    max_generations: int = 100
    patience: int = 10
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for optimization strategies.

    Subclasses must implement the core optimization methods:
    - initialize: Create initial population
    - evolve: Generate next generation
    - should_terminate: Check stopping criteria
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialize the strategy.

        Args:
            config: Strategy configuration. Uses defaults if None.
        """
        self.config = config or StrategyConfig()
        self._setup_random_state()

    def _setup_random_state(self) -> None:
        """Set up random state for reproducibility."""
        import random

        import numpy as np

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

    @abstractmethod
    def initialize(
        self,
        seed_prompt: Prompt,
        population_size: int,
    ) -> Population:
        """Create the initial population.

        Args:
            seed_prompt: The starting prompt to build from.
            population_size: Number of individuals to create.

        Returns:
            Initial population.
        """
        pass

    @abstractmethod
    def evolve(
        self,
        population: Population,
        fitness_scores: list[float],
    ) -> Population:
        """Generate the next generation.

        Args:
            population: Current population.
            fitness_scores: Fitness scores for each individual.

        Returns:
            New population for the next generation.
        """
        pass

    def should_terminate(
        self,
        history: OptimizationHistory,
        max_generations: int,
        patience: int,
    ) -> bool:
        """Check if optimization should stop.

        Default implementation checks:
        1. Maximum generations reached
        2. No improvement for 'patience' generations

        Args:
            history: Optimization history so far.
            max_generations: Maximum allowed generations.
            patience: Generations without improvement before stopping.

        Returns:
            True if optimization should terminate.
        """
        if len(history.generations) >= max_generations:
            return True

        if len(history.generations) < patience:
            return False

        # Check for improvement in last 'patience' generations
        trajectory = history.fitness_trajectory
        recent = trajectory[-patience:]
        best_recent = max(recent)
        best_ever = max(trajectory)

        return best_recent < best_ever

    def get_strategy_info(self) -> dict[str, Any]:
        """Return information about this strategy.

        Returns:
            Dictionary with strategy metadata.
        """
        return {
            "name": self.__class__.__name__,
            "config": {
                "population_size": self.config.population_size,
                "max_generations": self.config.max_generations,
                "patience": self.config.patience,
                "seed": self.config.seed,
                **self.config.extra,
            },
        }
