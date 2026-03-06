"""Optimizer controller - main orchestration component.

This module provides the Optimizer class that coordinates the complete
prompt optimization workflow: population management, evaluation,
strategy evolution, and progress tracking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from promptfoundry.core.history import OptimizationHistory, OptimizationResult
from promptfoundry.core.population import Individual, Population
from promptfoundry.core.prompt import Prompt
from promptfoundry.core.protocols import Evaluator, LLMClient, OptimizationStrategy
from promptfoundry.core.task import Example, Task


class ProgressCallback(Protocol):
    """Protocol for progress callbacks during optimization."""

    def __call__(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        best_prompt: str,
    ) -> None:
        """Report optimization progress.

        Args:
            generation: Current generation number.
            best_fitness: Best fitness in current generation.
            avg_fitness: Average fitness in current generation.
            best_prompt: Text of the best prompt.
        """
        ...


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer.

    Attributes:
        max_generations: Maximum number of generations to run.
        population_size: Number of individuals in the population.
        patience: Generations without improvement before early stopping.
        checkpoint_dir: Directory for saving checkpoints (None to disable).
        checkpoint_frequency: Save checkpoint every N generations.
        batch_size: Number of prompts to evaluate per outer batch.
        max_concurrency: Maximum number of concurrent LLM requests.
    """

    max_generations: int = 50
    population_size: int = 10
    patience: int = 10
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 5
    batch_size: int = 5
    max_concurrency: int = 1


@dataclass
class OptimizationState:
    """Internal state during optimization.

    Tracks progress, best results, and convergence information.
    """

    current_generation: int = 0
    best_score: float = 0.0
    best_prompt: Prompt | None = None
    generations_without_improvement: int = 0
    total_evaluations: int = 0
    start_time: float = field(default_factory=time.time)

    def update_best(self, prompt: Prompt, score: float) -> bool:
        """Update best result if score improves.

        Args:
            prompt: Candidate prompt.
            score: Candidate score.

        Returns:
            True if this is a new best, False otherwise.
        """
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
            self.generations_without_improvement = 0
            return True
        else:
            self.generations_without_improvement += 1
            return False


class Optimizer:
    """Main optimizer controller for prompt optimization.

    Coordinates:
    - Population initialization and evolution (via Strategy)
    - Prompt evaluation (via LLM Client + Evaluator)
    - Progress tracking and history management
    - Checkpointing and resumption

    Example:
        >>> from promptfoundry.core import Prompt, Task
        >>> from promptfoundry.strategies import GeneticAlgorithmStrategy
        >>> from promptfoundry.evaluators import ExactMatchEvaluator
        >>> from promptfoundry.llm import OpenAICompatClient
        >>>
        >>> optimizer = Optimizer(
        ...     strategy=GeneticAlgorithmStrategy(),
        ...     evaluator=ExactMatchEvaluator(),
        ...     llm_client=OpenAICompatClient(),
        ... )
        >>> result = await optimizer.optimize(seed_prompt, task)
    """

    def __init__(
        self,
        strategy: OptimizationStrategy,
        evaluator: Evaluator,
        llm_client: LLMClient,
        config: OptimizerConfig | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            strategy: Optimization strategy (e.g., GeneticAlgorithmStrategy).
            evaluator: Evaluator for scoring outputs.
            llm_client: LLM client for generating completions.
            config: Optimizer configuration.
        """
        self.strategy = strategy
        self.evaluator = evaluator
        self.llm_client = llm_client
        self.config = config or OptimizerConfig()

        self._callbacks: list[ProgressCallback] = []
        self._history: OptimizationHistory | None = None
        self._state: OptimizationState | None = None

        # Cache exact evaluations within a single run.
        # avoids re‑evaluating identical combinations when evolution produces
        # duplicates or when checkpoint/resume happens.
        self._score_cache: dict[tuple[str, str, str, str | None], float] = {}
        max_concurrency = max(1, self.config.max_concurrency)
        self._sem = asyncio.Semaphore(max_concurrency)

    def add_callback(self, callback: ProgressCallback) -> None:
        """Add a progress callback.

        Args:
            callback: Callback function to invoke on each generation.
        """
        self._callbacks.append(callback)

    async def optimize(
        self,
        seed_prompt: Prompt,
        task: Task,
        resume_from: str | Path | None = None,
    ) -> OptimizationResult:
        """Run the complete optimization loop.

        Args:
            seed_prompt: Initial prompt to optimize.
            task: Task with examples for evaluation.
            resume_from: Path to checkpoint to resume from.

        Returns:
            OptimizationResult with best prompt and statistics.
        """
        # Initialize state
        self._state = OptimizationState(start_time=time.time())
        self._history = OptimizationHistory(
            seed_prompt=seed_prompt.text,
            task_name=task.name,
            config=self._get_config_dict(),
        )

        # Resume from checkpoint if provided
        if resume_from:
            await self._resume_from_checkpoint(Path(resume_from))

        # Initialize or continue population
        if self._state.current_generation == 0:
            population = self.strategy.initialize(
                seed_prompt, self.config.population_size
            )
            self._state.best_prompt = seed_prompt
        else:
            # Would need to reconstruct population from checkpoint
            population = self.strategy.initialize(
                seed_prompt, self.config.population_size
            )

        # Main optimization loop
        while not self._should_terminate():
            # Evaluate current population
            try:
                fitness_scores = await self._evaluate_population(population, task)
            except asyncio.CancelledError:
                # optimization was cancelled (e.g. Ctrl-C); stop early
                break
            self._state.total_evaluations += len(population)

            # Find generation best
            gen_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            gen_best = population[gen_best_idx]
            gen_best_score = fitness_scores[gen_best_idx]
            avg_score = sum(fitness_scores) / len(fitness_scores)

            # Update state
            self._state.update_best(gen_best.prompt, gen_best_score)

            # Record history
            evaluated_pop = Population(
                individuals=[
                    ind.with_fitness(score)
                    for ind, score in zip(population.individuals, fitness_scores, strict=True)
                ],
                generation=self._state.current_generation,
            )
            self._history.add_generation(evaluated_pop)

            # Invoke callbacks
            self._invoke_callbacks(
                self._state.current_generation,
                gen_best_score,
                avg_score,
                gen_best.prompt.text,
            )

            # Checkpoint if configured
            await self._maybe_checkpoint()

            # Evolve to next generation
            population = self.strategy.evolve(population, fitness_scores)
            self._state.current_generation += 1

        # Build final result
        elapsed = time.time() - self._state.start_time

        return OptimizationResult(
            best_prompt=self._state.best_prompt or seed_prompt,
            best_score=self._state.best_score,
            total_generations=self._state.current_generation,
            total_evaluations=self._state.total_evaluations,
            elapsed_time=elapsed,
            convergence_generation=self._find_convergence_generation(),
            history=self._history,
        )

    async def _evaluate_population(
        self,
        population: Population,
        task: Task,
    ) -> list[float]:
        """Evaluate all individuals in a population.

        Args:
            population: Population to evaluate.
            task: Task with examples.

        Returns:
            List of fitness scores for each individual.
        """
        fitness_scores: list[float] = []

        # Process in batches for efficiency
        for i in range(0, len(population), self.config.batch_size):
            batch = population.individuals[i : i + self.config.batch_size]
            batch_scores = await self._evaluate_batch(batch, task.examples, task.system_prompt)
            fitness_scores.extend(batch_scores)

        return fitness_scores

    async def _evaluate_batch(
        self,
        individuals: list[Individual],
        examples: list[Example],
        system_prompt: str | None = None,
    ) -> list[float]:
        """Evaluate a batch of individuals against examples.

        This version launches all example evaluations concurrently and uses an
        in‑memory cache to avoid duplicated LLM requests.  A semaphore throttles
        concurrent calls to protect slow or single‑threaded LLM backends.
        """
        async def _score_example(ind: Individual, example: Example) -> float:
            key = (
                ind.prompt.text,
                example.input,
                example.expected_output,
                system_prompt,
            )
            if key in self._score_cache:
                return self._score_cache[key]

            prompt_text = self._format_prompt(ind.prompt, example)
            try:
                async with self._sem:
                    completion = await self.llm_client.complete(
                        prompt_text, system_prompt=system_prompt
                    )
                score = self.evaluator.evaluate(completion, example.expected_output)
            except Exception:
                score = 0.0

            self._score_cache[key] = score
            return score

        scores: list[float] = []

        for ind in individuals:
            tasks = [_score_example(ind, ex) for ex in examples]
            ind_scores = await asyncio.gather(*tasks)
            fitness = self.evaluator.aggregate(ind_scores)
            scores.append(fitness)

        return scores

    def _format_prompt(self, prompt: Prompt, example: Example) -> str:
        """Format a prompt with an example input.

        Args:
            prompt: The prompt template.
            example: Example to apply.

        Returns:
            Formatted prompt string.
        """
        text = prompt.text

        # Replace {input} placeholder if present
        if "{input}" in text:
            return text.replace("{input}", example.input)

        # Otherwise append input
        return f"{text}\n\nInput: {example.input}"

    def _should_terminate(self) -> bool:
        """Check if optimization should stop.

        Returns:
            True if should terminate.
        """
        if self._state is None:
            return True

        # Max generations reached
        if self._state.current_generation >= self.config.max_generations:
            return True

        # Early stopping due to no improvement
        if self._state.generations_without_improvement >= self.config.patience:
            return True

        # Perfect score achieved
        if self._state.best_score >= 1.0:
            return True

        return False

    def _invoke_callbacks(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        best_prompt: str,
    ) -> None:
        """Invoke all registered callbacks.

        Args:
            generation: Current generation.
            best_fitness: Best fitness this generation.
            avg_fitness: Average fitness this generation.
            best_prompt: Best prompt text.
        """
        for callback in self._callbacks:
            try:
                callback(generation, best_fitness, avg_fitness, best_prompt)
            except Exception:
                pass  # Don't let callback errors stop optimization

    async def _maybe_checkpoint(self) -> None:
        """Save checkpoint if configured and due."""
        if not self.config.checkpoint_dir or self._state is None or self._history is None:
            return

        if self._state.current_generation % self.config.checkpoint_frequency == 0:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"checkpoint_gen_{self._state.current_generation}.json"
            self._history.save(checkpoint_path)

    async def _resume_from_checkpoint(self, path: Path) -> None:
        """Resume optimization from a checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        if path.exists():
            self._history = OptimizationHistory.load(path)
            if self._state and self._history.generations:
                self._state.current_generation = len(self._history.generations)

                # Restore best from history
                best = self._history.best_ever
                if best:
                    self._state.best_prompt = Prompt(text=best[0])
                    self._state.best_score = best[1]

    def _find_convergence_generation(self) -> int:
        """Find the generation where the best score was first achieved.

        Returns:
            Generation number where best was found.
        """
        if not self._history or not self._history.generations:
            return 0

        target = self._state.best_score if self._state else 0.0

        for gen in self._history.generations:
            if gen.best_fitness >= target:
                return gen.generation

        return len(self._history.generations) - 1

    def _get_config_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "max_generations": self.config.max_generations,
            "population_size": self.config.population_size,
            "patience": self.config.patience,
            "batch_size": self.config.batch_size,
        }

    @property
    def history(self) -> OptimizationHistory | None:
        """Return the optimization history."""
        return self._history
