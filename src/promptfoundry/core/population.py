"""Population and individual models for evolutionary optimization.

This module defines the Individual and Population classes used
in evolutionary and population-based optimization strategies.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptfoundry.core.prompt import Prompt


@dataclass
class Individual:
    """Single prompt individual in a population.

    Represents one candidate prompt along with its fitness score
    and genealogical information.

    Attributes:
        prompt: The prompt this individual represents.
        fitness: Evaluated fitness score (None if not yet evaluated).
        generation: Generation number when this individual was created.
        parent_ids: IDs of parent individuals (for genealogy tracking).
        id: Unique identifier for this individual.
    """

    prompt: Prompt
    fitness: float | None = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self) -> str:
        """Return a brief representation."""
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        return f"Individual(id={self.id}, fitness={fitness_str}, gen={self.generation})"

    def __lt__(self, other: Individual) -> bool:
        """Compare individuals by fitness (for sorting).

        Individuals with None fitness are considered less fit than any scored individual.
        """
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness

    def with_fitness(self, fitness: float) -> Individual:
        """Create a new Individual with updated fitness.

        Args:
            fitness: The evaluated fitness score.

        Returns:
            New Individual with the fitness value set.
        """
        return Individual(
            prompt=self.prompt,
            fitness=fitness,
            generation=self.generation,
            parent_ids=self.parent_ids,
            id=self.id,
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the individual.
        """
        return {
            "id": self.id,
            "prompt_text": self.prompt.text,
            "prompt_id": self.prompt.id,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


@dataclass
class Population:
    """Collection of individuals in a single generation.

    Attributes:
        individuals: List of individuals in this population.
        generation: Generation number.
    """

    individuals: list[Individual]
    generation: int = 0

    def __len__(self) -> int:
        """Return the population size."""
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over individuals."""
        return iter(self.individuals)

    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]

    @property
    def best(self) -> Individual | None:
        """Return the individual with highest fitness.

        Returns:
            The best individual, or None if population is empty
            or no individuals have been evaluated.
        """
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda x: x.fitness or 0.0)

    @property
    def average_fitness(self) -> float | None:
        """Calculate average fitness of evaluated individuals.

        Returns:
            Average fitness, or None if no individuals are evaluated.
        """
        evaluated = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if not evaluated:
            return None
        return sum(evaluated) / len(evaluated)

    @property
    def fitness_scores(self) -> list[float | None]:
        """Return fitness scores for all individuals.

        Returns:
            List of fitness scores (None for unevaluated individuals).
        """
        return [ind.fitness for ind in self.individuals]

    def sorted_by_fitness(self, descending: bool = True) -> list[Individual]:
        """Return individuals sorted by fitness.

        Args:
            descending: If True, highest fitness first.

        Returns:
            Sorted list of individuals.
        """
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        return sorted(evaluated, key=lambda x: x.fitness or 0.0, reverse=descending)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the population.
        """
        return {
            "generation": self.generation,
            "size": len(self.individuals),
            "individuals": [ind.to_dict() for ind in self.individuals],
            "best_fitness": self.best.fitness if self.best else None,
            "average_fitness": self.average_fitness,
        }
