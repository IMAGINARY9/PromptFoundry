"""Protocol definitions for PromptFoundry components.

This module defines the abstract interfaces (protocols) that components
must implement. Using protocols enables loose coupling and easy testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from promptfoundry.core.history import OptimizationHistory
    from promptfoundry.core.population import Population
    from promptfoundry.core.prompt import Prompt


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for objective functions that score prompt outputs.

    Evaluators compare predicted outputs against expected outputs
    and return a score between 0.0 and 1.0.
    """

    def evaluate(
        self,
        predicted: str,
        expected: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Score a single prediction.

        Args:
            predicted: The LLM's output.
            expected: The expected/desired output.
            metadata: Optional additional context.

        Returns:
            Score between 0.0 (worst) and 1.0 (best).
        """
        ...

    def evaluate_batch(
        self,
        predictions: list[str],
        expected: list[str],
    ) -> list[float]:
        """Score multiple predictions.

        Args:
            predictions: List of LLM outputs.
            expected: List of expected outputs.

        Returns:
            List of scores, one per prediction.
        """
        ...

    def aggregate(self, scores: list[float]) -> float:
        """Aggregate multiple scores into a single fitness value.

        Args:
            scores: List of individual scores.

        Returns:
            Aggregated score (typically mean).
        """
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM backend adapters.

    LLM clients handle communication with language model backends,
    including local models and API-based services.
    """

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for a prompt.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt.
            **kwargs: Additional generation parameters.

        Returns:
            The generated completion text.
        """
        ...

    async def complete_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate completions for multiple prompts.

        Args:
            prompts: List of prompts to complete.
            system_prompt: Optional system prompt for all.
            **kwargs: Additional generation parameters.

        Returns:
            List of completions, one per prompt.
        """
        ...


@runtime_checkable
class OptimizationStrategy(Protocol):
    """Protocol for optimization algorithms.

    Strategies define how populations evolve over generations,
    including initialization, selection, and offspring generation.
    """

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
        ...

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
        ...

    def should_terminate(
        self,
        history: OptimizationHistory,
        max_generations: int,
        patience: int,
    ) -> bool:
        """Check if optimization should stop.

        Args:
            history: Optimization history so far.
            max_generations: Maximum allowed generations.
            patience: Generations without improvement before stopping.

        Returns:
            True if optimization should terminate.
        """
        ...


@runtime_checkable
class MutationOperator(Protocol):
    """Protocol for prompt mutation operators.

    Mutation operators modify prompts to create variants.
    """

    def mutate(self, prompt: Prompt) -> Prompt:
        """Create a mutated variant of a prompt.

        Args:
            prompt: The prompt to mutate.

        Returns:
            A new, mutated prompt.
        """
        ...


@runtime_checkable
class CrossoverOperator(Protocol):
    """Protocol for prompt crossover operators.

    Crossover operators combine two prompts to create offspring.
    """

    def crossover(self, parent1: Prompt, parent2: Prompt) -> tuple[Prompt, Prompt]:
        """Create offspring from two parent prompts.

        Args:
            parent1: First parent prompt.
            parent2: Second parent prompt.

        Returns:
            Tuple of two offspring prompts.
        """
        ...


@runtime_checkable
class SelectionOperator(Protocol):
    """Protocol for selection operators.

    Selection operators choose individuals for reproduction.
    """

    def select(
        self,
        population: Population,
        n: int,
    ) -> list[Prompt]:
        """Select individuals for reproduction.

        Args:
            population: Population to select from.
            n: Number of individuals to select.

        Returns:
            List of selected prompts.
        """
        ...
