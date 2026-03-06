"""Optimization history and result tracking.

This module defines classes for tracking optimization progress,
storing results, and enabling checkpointing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from promptfoundry.core.population import Population
from promptfoundry.core.prompt import Prompt


@dataclass
class GenerationRecord:
    """Record of a single generation's state and statistics.

    Attributes:
        generation: Generation number.
        best_fitness: Best fitness in this generation.
        average_fitness: Average fitness in this generation.
        best_prompt: Text of the best prompt.
        population_size: Number of individuals.
        timestamp: When this generation was recorded.
        metadata: Additional generation-specific data.
    """

    generation: int
    best_fitness: float
    average_fitness: float
    best_prompt: str
    population_size: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Final result of an optimization run.

    Attributes:
        best_prompt: The best-performing prompt found.
        best_score: The fitness score of the best prompt.
        total_generations: Total number of generations run.
        total_evaluations: Total number of prompt evaluations.
        elapsed_time: Total optimization time in seconds.
        convergence_generation: Generation where best was found.
        history: Complete optimization history.
        termination_reason: Why the optimization stopped.
        total_llm_calls: Total LLM API calls made.
        total_cache_hits: Total evaluations served from cache.
    """

    best_prompt: Prompt
    best_score: float
    total_generations: int
    total_evaluations: int
    elapsed_time: float
    convergence_generation: int
    history: OptimizationHistory
    termination_reason: str = "unknown"
    total_llm_calls: int = 0
    total_cache_hits: int = 0

    def __str__(self) -> str:
        """Return a summary of the result."""
        return (
            f"OptimizationResult(\n"
            f"  best_score={self.best_score:.4f},\n"
            f"  generations={self.total_generations},\n"
            f"  evaluations={self.total_evaluations},\n"
            f"  converged_at=gen_{self.convergence_generation},\n"
            f"  elapsed={self.elapsed_time:.2f}s,\n"
            f"  termination={self.termination_reason}\n"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        seed_fitness = 0.0
        if self.history.generations:
            seed_fitness = self.history.generations[0].best_fitness

        return {
            "best_prompt": self.best_prompt.text,
            "best_score": self.best_score,
            "total_generations": self.total_generations,
            "total_evaluations": self.total_evaluations,
            "elapsed_time": self.elapsed_time,
            "convergence_generation": self.convergence_generation,
            "termination_reason": self.termination_reason,
            "total_llm_calls": self.total_llm_calls,
            "total_cache_hits": self.total_cache_hits,
            "seed_fitness": seed_fitness,
            "history": self.history.to_dict(),
        }


@dataclass
class OptimizationHistory:
    """Complete history of an optimization run.

    Tracks all generations, enabling analysis and checkpointing.

    Attributes:
        generations: List of generation records.
        config: Configuration used for this run.
        seed_prompt: The initial seed prompt.
        start_time: When optimization started.
        task_name: Name of the optimization task.
    """

    generations: list[GenerationRecord] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    seed_prompt: str = ""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    task_name: str = ""

    def add_generation(
        self,
        population: Population,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a generation's state.

        Args:
            population: The population to record.
            metadata: Optional additional metadata (timing, cache stats, etc.).
        """
        best = population.best
        avg = population.average_fitness

        if best is None or avg is None:
            raise ValueError("Cannot record unevaluated population")

        record = GenerationRecord(
            generation=population.generation,
            best_fitness=best.fitness or 0.0,
            average_fitness=avg,
            best_prompt=best.prompt.text,
            population_size=len(population),
            metadata=metadata or {},
        )
        self.generations.append(record)

    @property
    def best_ever(self) -> tuple[str, float] | None:
        """Return the best prompt and score across all generations.

        Returns:
            Tuple of (prompt_text, score) or None if no generations recorded.
        """
        if not self.generations:
            return None

        best_gen = max(self.generations, key=lambda g: g.best_fitness)
        return best_gen.best_prompt, best_gen.best_fitness

    @property
    def fitness_trajectory(self) -> list[float]:
        """Return best fitness values across generations.

        Returns:
            List of best fitness values per generation.
        """
        return [g.best_fitness for g in self.generations]

    @property
    def average_trajectory(self) -> list[float]:
        """Return average fitness values across generations.

        Returns:
            List of average fitness values per generation.
        """
        return [g.average_fitness for g in self.generations]

    def save(self, path: str | Path) -> None:
        """Save history to a JSON file.

        Args:
            path: Path to save the history file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert history to dictionary for serialization."""
        return {
            "task_name": self.task_name,
            "seed_prompt": self.seed_prompt,
            "start_time": self.start_time,
            "config": self.config,
            "generations": [
                {
                    "generation": g.generation,
                    "best_fitness": g.best_fitness,
                    "average_fitness": g.average_fitness,
                    "best_prompt": g.best_prompt,
                    "population_size": g.population_size,
                    "timestamp": g.timestamp,
                    "metadata": g.metadata,
                }
                for g in self.generations
            ],
        }

    @classmethod
    def load(cls, path: str | Path) -> OptimizationHistory:
        """Load history from a JSON file.

        Args:
            path: Path to the history file.

        Returns:
            Loaded OptimizationHistory instance.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        history = cls(
            task_name=data.get("task_name", ""),
            seed_prompt=data.get("seed_prompt", ""),
            start_time=data.get("start_time", ""),
            config=data.get("config", {}),
        )

        for g in data.get("generations", []):
            history.generations.append(
                GenerationRecord(
                    generation=g["generation"],
                    best_fitness=g["best_fitness"],
                    average_fitness=g["average_fitness"],
                    best_prompt=g["best_prompt"],
                    population_size=g["population_size"],
                    timestamp=g.get("timestamp", ""),
                    metadata=g.get("metadata", {}),
                )
            )

        return history
