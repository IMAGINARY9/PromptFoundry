"""Evolutionary (Genetic Algorithm) optimization strategy.

This module implements a genetic algorithm-based optimization strategy
for prompt engineering, including mutation, crossover, and selection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from promptfoundry.core.population import Individual, Population
from promptfoundry.core.prompt import Prompt
from promptfoundry.strategies.base import BaseStrategy, StrategyConfig


@dataclass
class EvolutionaryConfig(StrategyConfig):
    """Configuration for evolutionary strategy.

    Attributes:
        mutation_rate: Probability of mutation per individual (0.0-1.0).
        crossover_rate: Probability of crossover (0.0-1.0).
        tournament_size: Number of individuals in tournament selection.
        elitism: Number of top individuals to preserve unchanged.
    """

    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elitism: int = 2


class GeneticAlgorithmStrategy(BaseStrategy):
    """Genetic algorithm-based optimization strategy.

    Uses tournament selection, single-point crossover, and text mutation
    to evolve a population of prompts toward better performance.
    """

    def __init__(self, config: EvolutionaryConfig | None = None) -> None:
        """Initialize the genetic algorithm strategy.

        Args:
            config: Evolutionary configuration. Uses defaults if None.
        """
        self.evo_config = config or EvolutionaryConfig()
        super().__init__(self.evo_config)

    def initialize(
        self,
        seed_prompt: Prompt,
        population_size: int,
    ) -> Population:
        """Create initial population from seed prompt.

        The initial population includes the seed prompt and variants
        created through mutation.

        Args:
            seed_prompt: The starting prompt.
            population_size: Desired population size.

        Returns:
            Initial population.
        """
        individuals: list[Individual] = []

        # Include the original seed
        individuals.append(
            Individual(
                prompt=seed_prompt,
                generation=0,
                parent_ids=[],
            )
        )

        # Generate variants through mutation
        for _ in range(population_size - 1):
            mutated = self._mutate_prompt(seed_prompt)
            individuals.append(
                Individual(
                    prompt=mutated,
                    generation=0,
                    parent_ids=[seed_prompt.id],
                )
            )

        return Population(individuals=individuals, generation=0)

    def evolve(
        self,
        population: Population,
        fitness_scores: list[float],
    ) -> Population:
        """Generate the next generation through selection, crossover, and mutation.

        Args:
            population: Current population.
            fitness_scores: Fitness scores for each individual.

        Returns:
            New population for the next generation.
        """
        # Update fitness scores
        evaluated = []
        for ind, score in zip(population.individuals, fitness_scores, strict=True):
            evaluated.append(ind.with_fitness(score))

        # Sort by fitness (descending)
        evaluated.sort(key=lambda x: x.fitness or 0.0, reverse=True)

        new_individuals: list[Individual] = []
        next_gen = population.generation + 1

        # Elitism: preserve top individuals
        for elite in evaluated[: self.evo_config.elitism]:
            new_individuals.append(
                Individual(
                    prompt=elite.prompt,
                    fitness=None,  # Will be re-evaluated
                    generation=next_gen,
                    parent_ids=[elite.id],
                )
            )

        # Generate offspring
        while len(new_individuals) < len(population):
            # Tournament selection
            parent1 = self._tournament_select(evaluated)
            parent2 = self._tournament_select(evaluated)

            # Crossover
            if random.random() < self.evo_config.crossover_rate:
                child1_prompt, child2_prompt = self._crossover(
                    parent1.prompt, parent2.prompt
                )
            else:
                child1_prompt = parent1.prompt
                child2_prompt = parent2.prompt

            # Mutation
            if random.random() < self.evo_config.mutation_rate:
                child1_prompt = self._mutate_prompt(child1_prompt)
            if random.random() < self.evo_config.mutation_rate:
                child2_prompt = self._mutate_prompt(child2_prompt)

            # Add children
            new_individuals.append(
                Individual(
                    prompt=child1_prompt,
                    generation=next_gen,
                    parent_ids=[parent1.id, parent2.id],
                )
            )
            if len(new_individuals) < len(population):
                new_individuals.append(
                    Individual(
                        prompt=child2_prompt,
                        generation=next_gen,
                        parent_ids=[parent1.id, parent2.id],
                    )
                )

        return Population(individuals=new_individuals, generation=next_gen)

    def _tournament_select(self, evaluated: list[Individual]) -> Individual:
        """Select an individual using tournament selection.

        Args:
            evaluated: List of evaluated individuals.

        Returns:
            Selected individual.
        """
        tournament = random.sample(
            evaluated, min(self.evo_config.tournament_size, len(evaluated))
        )
        return max(tournament, key=lambda x: x.fitness or 0.0)

    def _crossover(self, parent1: Prompt, parent2: Prompt) -> tuple[Prompt, Prompt]:
        """Perform single-point crossover on prompt texts.

        Args:
            parent1: First parent prompt.
            parent2: Second parent prompt.

        Returns:
            Tuple of two offspring prompts.
        """
        text1 = parent1.text
        text2 = parent2.text

        # Split by sentences for more meaningful crossover
        import re

        sentences1 = re.split(r"(?<=[.!?])\s+", text1)
        sentences2 = re.split(r"(?<=[.!?])\s+", text2)

        if len(sentences1) > 1 and len(sentences2) > 1:
            # Single-point crossover at sentence level
            point1 = random.randint(1, len(sentences1) - 1)
            point2 = random.randint(1, len(sentences2) - 1)

            child1_text = " ".join(sentences1[:point1] + sentences2[point2:])
            child2_text = " ".join(sentences2[:point2] + sentences1[point1:])
        else:
            # Word-level crossover for short prompts
            words1 = text1.split()
            words2 = text2.split()

            if len(words1) > 2 and len(words2) > 2:
                point = random.randint(1, min(len(words1), len(words2)) - 1)
                child1_text = " ".join(words1[:point] + words2[point:])
                child2_text = " ".join(words2[:point] + words1[point:])
            else:
                # Too short, just swap
                child1_text = text2
                child2_text = text1

        return (
            parent1.with_text(child1_text),
            parent2.with_text(child2_text),
        )

    def _mutate_prompt(self, prompt: Prompt) -> Prompt:
        """Apply a random mutation to a prompt.

        Available mutations:
        - Rephrase a sentence
        - Add a constraint
        - Remove a constraint
        - Modify formatting instruction

        Args:
            prompt: The prompt to mutate.

        Returns:
            Mutated prompt.
        """
        mutations = [
            self._mutate_rephrase,
            self._mutate_add_constraint,
            self._mutate_remove_word,
            self._mutate_swap_words,
        ]

        mutation_fn = random.choice(mutations)
        new_text = mutation_fn(prompt.text)

        return prompt.with_text(new_text)

    def _mutate_rephrase(self, text: str) -> str:
        """Rephrase by substituting common instruction words."""
        substitutions = {
            "Classify": ["Categorize", "Label", "Determine", "Identify"],
            "Determine": ["Figure out", "Identify", "Establish", "Ascertain"],
            "Output": ["Return", "Respond with", "Provide", "Give"],
            "the following": ["this", "the given", "the provided"],
            "Please": ["", "Kindly", ""],
            "must": ["should", "need to", "have to"],
        }

        for original, replacements in substitutions.items():
            if original.lower() in text.lower():
                replacement = random.choice(replacements)
                # Case-preserving replacement
                import re

                pattern = re.compile(re.escape(original), re.IGNORECASE)
                text = pattern.sub(replacement, text, count=1)
                break

        return text

    def _mutate_add_constraint(self, text: str) -> str:
        """Add a constraint or clarification."""
        constraints = [
            " Be concise.",
            " Respond with only the answer.",
            " Do not explain your reasoning.",
            " Think step by step.",
            " Be precise and accurate.",
            " Format your response clearly.",
        ]

        # Only add if not already present
        constraint = random.choice(constraints)
        if constraint.strip().lower() not in text.lower():
            # Add at the end
            text = text.rstrip() + constraint

        return text

    def _mutate_remove_word(self, text: str) -> str:
        """Remove a random non-essential word."""
        removable = ["please", "kindly", "just", "simply", "actually", "basically"]

        for word in removable:
            if word in text.lower():
                import re

                pattern = re.compile(r"\b" + re.escape(word) + r"\b\s*", re.IGNORECASE)
                text = pattern.sub("", text, count=1)
                break

        return text.strip()

    def _mutate_swap_words(self, text: str) -> str:
        """Swap two adjacent words."""
        words = text.split()
        if len(words) > 3:
            idx = random.randint(1, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            text = " ".join(words)

        return text
