"""Optimizer controller - main orchestration component.

This module provides the Optimizer class that coordinates the complete
prompt optimization workflow: population management, evaluation,
strategy evolution, and progress tracking.
"""

from __future__ import annotations

import asyncio
import base64
import json
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
        runtime_budget_seconds: Maximum runtime in seconds (0 for unlimited).
    """

    max_generations: int = 50
    population_size: int = 10
    patience: int = 10
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 5
    batch_size: int = 5
    max_concurrency: int = 1
    runtime_budget_seconds: float = 0.0
    adaptive_early_stopping: bool = True
    plateau_window: int = 3
    min_progress_delta: float = 0.01
    budget_buffer_ratio: float = 0.85

    @classmethod
    def from_runtime_config(cls, runtime_config: Any) -> OptimizerConfig:
        """Create OptimizerConfig from a RuntimeConfig.

        Args:
            runtime_config: RuntimeConfig instance.

        Returns:
            OptimizerConfig with values from RuntimeConfig.
        """
        return cls(
            max_generations=runtime_config.max_generations,
            population_size=runtime_config.population_size,
            patience=runtime_config.patience,
            batch_size=runtime_config.batch_size,
            max_concurrency=runtime_config.max_concurrency,
            checkpoint_frequency=runtime_config.checkpoint_frequency,
            runtime_budget_seconds=runtime_config.runtime_budget_seconds,
        )


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
    total_example_evaluations: int = 0
    total_llm_calls: int = 0
    total_cache_hits: int = 0
    termination_reason: str = "unknown"
    start_time: float = field(default_factory=time.time)
    # aggregated interactions across all evaluations
    interactions: list[dict[str, Any]] = field(default_factory=list)

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


@dataclass
class CachedEvaluation:
    """Cached evaluation details for a prompt/example pair."""

    score: float
    prompt_text: str
    completion: str


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
        self._current_population: Population | None = None

        # Cache exact evaluations within a single run.
        # avoids re‑evaluating identical combinations when evolution produces
        # duplicates or when checkpoint/resume happens.
        self._score_cache: dict[tuple[str, str, str, str | None], CachedEvaluation] = {}
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
        resumed_population: Population | None = None
        if resume_from:
            resumed_population = await self._resume_from_checkpoint(Path(resume_from))

        # Initialize or continue population
        if resumed_population is not None:
            population = resumed_population
        elif self._state.current_generation == 0:
            population = self.strategy.initialize(seed_prompt, self.config.population_size)
            self._state.best_prompt = seed_prompt
        else:
            # Fallback for legacy checkpoints that did not store population state.
            population = self.strategy.initialize(seed_prompt, self.config.population_size)
        self._current_population = population

        # Main optimization loop
        while not self._should_terminate():
            # Track generation timing
            gen_start_time = time.time()
            cache_size_before = len(self._score_cache)

            # Evaluate current population
            try:
                fitness_scores, interactions = await self._evaluate_population(population, task)
            except asyncio.CancelledError:
                # optimization was cancelled (e.g. Ctrl-C); stop early
                self._state.termination_reason = "interrupted"
                break

            # record interactions in global state for result
            if interactions:
                self._state.interactions.extend(interactions)

            # Calculate generation metrics
            gen_elapsed_ms = (time.time() - gen_start_time) * 1000
            cache_size_after = len(self._score_cache)
            num_examples = len(task.examples)
            total_evaluations_this_gen = len(population) * num_examples
            cache_hits_this_gen = total_evaluations_this_gen - (cache_size_after - cache_size_before)
            llm_calls_this_gen = cache_size_after - cache_size_before
            
            # Update state totals
            self._state.total_evaluations += len(population)
            self._state.total_example_evaluations += total_evaluations_this_gen
            self._state.total_llm_calls += llm_calls_this_gen
            self._state.total_cache_hits += cache_hits_this_gen

            evaluated_pop = Population(
                individuals=[
                    ind.with_fitness(score)
                    for ind, score in zip(population.individuals, fitness_scores, strict=True)
                ],
                generation=self._state.current_generation,
            )

            operator_stats: dict[str, Any] = {}
            generation_operator_summary: dict[str, Any] = {}
            if hasattr(self.strategy, "record_generation_feedback"):
                self.strategy.record_generation_feedback(evaluated_pop, fitness_scores)  # type: ignore[attr-defined]
            if hasattr(self.strategy, "get_operator_stats"):
                operator_stats = self.strategy.get_operator_stats()  # type: ignore[attr-defined]
            if hasattr(self.strategy, "get_last_generation_summary"):
                generation_operator_summary = self.strategy.get_last_generation_summary()  # type: ignore[attr-defined]

            # Find generation best
            gen_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            gen_best = population[gen_best_idx]
            gen_best_score = fitness_scores[gen_best_idx]
            avg_score = sum(fitness_scores) / len(fitness_scores)

            # Update state
            self._state.update_best(gen_best.prompt, gen_best_score)

            # Record history with timing metadata
            metadata = {
                "evaluation_time_ms": gen_elapsed_ms,
                "llm_calls": llm_calls_this_gen,
                "cache_hits": cache_hits_this_gen,
                "population_evaluations": len(population),
                "example_evaluations": total_evaluations_this_gen,
            }
            if interactions:
                # store raw prompts/completions for debugging
                metadata["interactions"] = interactions
            if operator_stats:
                metadata["operator_stats"] = operator_stats
            if generation_operator_summary:
                metadata["generation_operator_summary"] = generation_operator_summary

            self._history.add_generation(
                evaluated_pop,
                metadata=metadata,
            )

            # Invoke callbacks
            self._invoke_callbacks(
                self._state.current_generation,
                gen_best_score,
                avg_score,
                gen_best.prompt.text,
            )

            # Evolve to next generation
            population = self.strategy.evolve(population, fitness_scores)
            self._state.current_generation += 1
            self._current_population = population

            # Checkpoint if configured
            await self._maybe_checkpoint()

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
            termination_reason=self._state.termination_reason,
            total_llm_calls=self._state.total_llm_calls,
            total_cache_hits=self._state.total_cache_hits,
            total_example_evaluations=self._state.total_example_evaluations,
            operator_stats=operator_stats,
            interactions=self._state.interactions,
        )


    async def _evaluate_population(
        self,
        population: Population,
        task: Task,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Evaluate an entire population and record interactions.

        Returns:
            Tuple of (fitness scores list, interactions list).
        """
        fitness_scores: list[float] = []
        interactions: list[dict[str, Any]] = []

        # Process in batches for efficiency
        for i in range(0, len(population), self.config.batch_size):
            batch = population.individuals[i : i + self.config.batch_size]
            batch_scores, batch_interactions = await self._evaluate_batch(
                batch, task.examples, task.system_prompt
            )
            fitness_scores.extend(batch_scores)
            interactions.extend(batch_interactions)

        return fitness_scores, interactions

    async def _evaluate_batch(
        self,
        individuals: list[Individual],
        examples: list[Example],
        system_prompt: str | None = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Evaluate a batch of individuals against examples.

        Returns:
            Tuple of (fitness scores, interaction log).
        """

        async def _score_example(
            ind: Individual,
            example: Example,
        ) -> tuple[float, str, str, bool]:
            prompt_text = self._format_prompt(ind.prompt, example)
            key = (
                ind.prompt.text,
                example.input,
                example.expected_output,
                system_prompt,
            )
            if key in self._score_cache:
                cached = self._score_cache[key]
                return cached.score, cached.prompt_text, cached.completion, True

            try:
                async with self._sem:
                    completion = await self.llm_client.complete(
                        prompt_text, system_prompt=system_prompt
                    )
                score = self.evaluator.evaluate(
                    completion,
                    example.expected_output,
                    example.metadata,
                )
            except Exception:
                completion = ""
                score = 0.0

            self._score_cache[key] = CachedEvaluation(
                score=score,
                prompt_text=prompt_text,
                completion=completion,
            )
            return score, prompt_text, completion, False

        scores: list[float] = []
        interactions: list[dict[str, Any]] = []

        for ind in individuals:
            tasks = []
            for ex in examples:
                async def _task(ind=ind, ex=ex):
                    sc, prompt_text, completion, from_cache = await _score_example(ind, ex)
                    interactions.append(
                        {
                            "prompt": prompt_text,
                            "completion": completion,
                            "expected": ex.expected_output,
                            "score": sc,
                            "cached": from_cache,
                        }
                    )
                    return sc
                tasks.append(_task())
            ind_scores = await asyncio.gather(*tasks)
            fitness = self.evaluator.aggregate(ind_scores)
            scores.append(fitness)

        return scores, interactions

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
            self._state.termination_reason = "max_generations"
            return True

        if self._state.best_score >= 1.0:
            self._state.termination_reason = "perfect_score"
            return True

        if self._is_plateau_detected():
            elapsed = time.time() - self._state.start_time
            if (
                self.config.runtime_budget_seconds > 0
                and elapsed >= self.config.runtime_budget_seconds * self.config.budget_buffer_ratio
            ):
                self._state.termination_reason = "budget_plateau"
                return True

            adaptive_patience = min(self.config.patience, max(1, self.config.plateau_window))
            if self._state.generations_without_improvement >= adaptive_patience:
                self._state.termination_reason = "adaptive_plateau"
                return True

        # Early stopping due to no improvement
        if self._state.generations_without_improvement >= self.config.patience:
            self._state.termination_reason = "patience_exhausted"
            return True

        # Runtime budget exceeded
        if self.config.runtime_budget_seconds > 0:
            elapsed = time.time() - self._state.start_time
            if elapsed >= self.config.runtime_budget_seconds:
                self._state.termination_reason = "runtime_budget"
                return True

        return False

    def _is_plateau_detected(self) -> bool:
        """Detect low-progress runs that can stop early without losing signal."""
        if not self.config.adaptive_early_stopping or self._history is None:
            return False

        window = self.config.plateau_window
        if window < 1 or len(self._history.generations) < window + 1:
            return False

        recent = self._history.generations[-(window + 1) :]
        net_progress = recent[-1].best_fitness - recent[0].best_fitness
        return net_progress <= self.config.min_progress_delta

    def get_termination_reason(self) -> str:
        """Get the reason for termination.

        Returns:
            Human-readable termination reason.
        """
        if self._state is None:
            return "Not started"

        if self._state.current_generation >= self.config.max_generations:
            return "Max generations reached"

        if self._state.generations_without_improvement >= self.config.patience:
            return f"Early stopping (no improvement for {self.config.patience} generations)"

        if self._state.best_score >= 1.0:
            return "Perfect score achieved"

        if self._state.termination_reason == "adaptive_plateau":
            return "Adaptive plateau stop"

        if self._state.termination_reason == "budget_plateau":
            return "Budget-aware plateau stop"

        if self.config.runtime_budget_seconds > 0:
            elapsed = time.time() - self._state.start_time
            if elapsed >= self.config.runtime_budget_seconds:
                return f"Runtime budget exceeded ({self.config.runtime_budget_seconds}s)"

        return "Running"

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

            checkpoint_path = (
                checkpoint_dir / f"checkpoint_gen_{self._state.current_generation}.json"
            )
            self._save_checkpoint(checkpoint_path)

    async def _resume_from_checkpoint(self, path: Path) -> Population | None:
        """Resume optimization from a checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if "state" not in payload:
                self._history = OptimizationHistory.from_dict(payload)
                if self._state and self._history.generations:
                    self._state.current_generation = len(self._history.generations)
                    best = self._history.best_ever
                    if best:
                        self._state.best_prompt = Prompt(text=best[0])
                        self._state.best_score = best[1]
                return None

            self._history = OptimizationHistory.from_dict(payload["history"])
            if self._state is None:
                return None

            state_data = payload["state"]
            self._state.current_generation = int(state_data.get("current_generation", 0))
            self._state.best_score = float(state_data.get("best_score", 0.0))
            best_prompt_data = state_data.get("best_prompt")
            if best_prompt_data is not None:
                self._state.best_prompt = self._deserialize_prompt(best_prompt_data)
            self._state.generations_without_improvement = int(
                state_data.get("generations_without_improvement", 0)
            )
            self._state.total_evaluations = int(state_data.get("total_evaluations", 0))
            self._state.total_example_evaluations = int(
                state_data.get("total_example_evaluations", 0)
            )
            self._state.total_llm_calls = int(state_data.get("total_llm_calls", 0))
            self._state.total_cache_hits = int(state_data.get("total_cache_hits", 0))
            self._state.termination_reason = str(state_data.get("termination_reason", "unknown"))
            self._state.interactions = list(state_data.get("interactions", []))
            elapsed_before = float(state_data.get("elapsed_time", 0.0))
            self._state.start_time = time.time() - elapsed_before

            self._score_cache = self._deserialize_score_cache(payload.get("score_cache", []))
            self._restore_random_state(payload.get("random_state"))

            strategy_state = payload.get("strategy_state")
            if strategy_state and hasattr(self.strategy, "load_checkpoint_state"):
                self.strategy.load_checkpoint_state(strategy_state)  # type: ignore[attr-defined]

            population_data = payload.get("population")
            if population_data is not None:
                population = self._deserialize_population(population_data)
                self._current_population = population
                return population

        return None

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
            "max_concurrency": self.config.max_concurrency,
            "runtime_budget_seconds": self.config.runtime_budget_seconds,
            "adaptive_early_stopping": self.config.adaptive_early_stopping,
            "plateau_window": self.config.plateau_window,
            "min_progress_delta": self.config.min_progress_delta,
            "budget_buffer_ratio": self.config.budget_buffer_ratio,
        }

    def _save_checkpoint(self, path: Path) -> None:
        """Persist enough optimizer state to continue a run exactly."""
        if self._state is None or self._history is None:
            return

        strategy_state: dict[str, Any] | None = None
        if hasattr(self.strategy, "get_checkpoint_state"):
            strategy_state = self.strategy.get_checkpoint_state()  # type: ignore[attr-defined]

        payload = {
            "history": self._history.to_dict(),
            "state": {
                "current_generation": self._state.current_generation,
                "best_score": self._state.best_score,
                "best_prompt": self._serialize_prompt(self._state.best_prompt)
                if self._state.best_prompt is not None
                else None,
                "generations_without_improvement": self._state.generations_without_improvement,
                "total_evaluations": self._state.total_evaluations,
                "total_example_evaluations": self._state.total_example_evaluations,
                "total_llm_calls": self._state.total_llm_calls,
                "total_cache_hits": self._state.total_cache_hits,
                "termination_reason": self._state.termination_reason,
                "elapsed_time": time.time() - self._state.start_time,
                "interactions": self._state.interactions,
            },
            "population": self._serialize_population(self._current_population),
            "score_cache": self._serialize_score_cache(),
            "strategy_state": strategy_state,
            "random_state": self._capture_random_state(),
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _serialize_prompt(self, prompt: Prompt) -> dict[str, Any]:
        """Convert a prompt into checkpoint-safe data."""
        return {
            "text": prompt.text,
            "id": prompt.id,
            "metadata": prompt.metadata,
        }

    def _deserialize_prompt(self, data: dict[str, Any]) -> Prompt:
        """Restore a prompt from checkpoint data."""
        prompt_id = data.get("id")
        if prompt_id:
            return Prompt(
                text=data["text"],
                id=prompt_id,
                metadata=data.get("metadata", {}),
            )
        return Prompt(
            text=data["text"],
            metadata=data.get("metadata", {}),
        )

    def _serialize_population(self, population: Population | None) -> dict[str, Any] | None:
        """Serialize the current population for checkpointing."""
        if population is None:
            return None

        return {
            "generation": population.generation,
            "individuals": [
                {
                    "id": individual.id,
                    "fitness": individual.fitness,
                    "generation": individual.generation,
                    "parent_ids": individual.parent_ids,
                    "prompt": self._serialize_prompt(individual.prompt),
                }
                for individual in population.individuals
            ],
        }

    def _deserialize_population(self, data: dict[str, Any]) -> Population:
        """Restore a population from checkpoint data."""
        individuals = [
            Individual(
                prompt=self._deserialize_prompt(item["prompt"]),
                fitness=item.get("fitness"),
                generation=item.get("generation", data.get("generation", 0)),
                parent_ids=item.get("parent_ids", []),
                id=item.get("id"),
            )
            for item in data.get("individuals", [])
        ]
        return Population(individuals=individuals, generation=data.get("generation", 0))

    def _serialize_score_cache(self) -> list[dict[str, Any]]:
        """Serialize the evaluation cache for checkpointing."""
        entries: list[dict[str, Any]] = []
        for key, cached in self._score_cache.items():
            entries.append(
                {
                    "prompt_text_template": key[0],
                    "example_input": key[1],
                    "expected_output": key[2],
                    "system_prompt": key[3],
                    "score": cached.score,
                    "prompt_text": cached.prompt_text,
                    "completion": cached.completion,
                }
            )
        return entries

    def _deserialize_score_cache(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[tuple[str, str, str, str | None], CachedEvaluation]:
        """Restore the evaluation cache from checkpoint data."""
        cache: dict[tuple[str, str, str, str | None], CachedEvaluation] = {}
        for entry in entries:
            key = (
                entry["prompt_text_template"],
                entry["example_input"],
                entry["expected_output"],
                entry.get("system_prompt"),
            )
            cache[key] = CachedEvaluation(
                score=float(entry["score"]),
                prompt_text=entry["prompt_text"],
                completion=entry["completion"],
            )
        return cache

    def _capture_random_state(self) -> str:
        """Capture Python RNG state for resumable stochastic search."""
        return base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")

    def _restore_random_state(self, encoded_state: str | None) -> None:
        """Restore Python RNG state from checkpoint data."""
        if not encoded_state:
            return
        random.setstate(pickle.loads(base64.b64decode(encoded_state.encode("ascii"))))

    @property
    def history(self) -> OptimizationHistory | None:
        """Return the optimization history."""
        return self._history
