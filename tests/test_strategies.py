"""Tests for optimization strategies."""

from __future__ import annotations

import pytest

from promptfoundry.core.prompt import Prompt
from promptfoundry.core.population import Individual, Population
from promptfoundry.core.history import OptimizationHistory
from promptfoundry.strategies.evolutionary import (
    GeneticAlgorithmStrategy,
    EvolutionaryConfig,
)


class TestGeneticAlgorithmStrategy:
    """Tests for GeneticAlgorithmStrategy."""

    @pytest.fixture
    def strategy(self) -> GeneticAlgorithmStrategy:
        """Create a strategy with fixed seed for reproducibility."""
        config = EvolutionaryConfig(
            population_size=10,
            mutation_rate=0.3,
            crossover_rate=0.7,
            tournament_size=3,
            elitism=2,
            seed=42,
        )
        return GeneticAlgorithmStrategy(config)

    @pytest.fixture
    def seed_prompt(self) -> Prompt:
        """Create a seed prompt for testing."""
        return Prompt(text="Classify the sentiment of this text: {input}")

    def test_initialize_population(
        self, strategy: GeneticAlgorithmStrategy, seed_prompt: Prompt
    ) -> None:
        """Test population initialization."""
        population = strategy.initialize(seed_prompt, population_size=10)

        assert len(population) == 10
        assert population.generation == 0

        # First individual should be the seed
        assert population[0].prompt.text == seed_prompt.text

        # Others should be variants
        texts = {ind.prompt.text for ind in population}
        assert len(texts) > 1  # Not all identical

    def test_evolve_population(
        self, strategy: GeneticAlgorithmStrategy, seed_prompt: Prompt
    ) -> None:
        """Test population evolution."""
        population = strategy.initialize(seed_prompt, population_size=10)

        # Assign fitness scores
        fitness_scores = [0.5 + i * 0.05 for i in range(10)]

        new_population = strategy.evolve(population, fitness_scores)

        assert len(new_population) == 10
        assert new_population.generation == 1

    def test_elitism_preserves_best(
        self, strategy: GeneticAlgorithmStrategy, seed_prompt: Prompt
    ) -> None:
        """Test that elitism preserves top individuals."""
        population = strategy.initialize(seed_prompt, population_size=10)
        fitness_scores = list(range(10))  # 0-9

        new_population = strategy.evolve(population, fitness_scores)

        # Elitism = 2, so best 2 prompt texts should be preserved
        old_best_texts = {
            ind.prompt.text
            for ind, fit in sorted(
                zip(population.individuals, fitness_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:2]
        }
        new_texts = {ind.prompt.text for ind in new_population}

        assert old_best_texts.issubset(new_texts)

    def test_mutation_modifies_text(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Test that mutation modifies prompt text."""
        prompt = Prompt(text="Please classify the sentiment.")

        # Run mutation multiple times to ensure at least one change
        modified_texts = set()
        for _ in range(20):
            mutated = strategy._mutate_prompt(prompt)
            modified_texts.add(mutated.text)

        # Should have some variation
        assert len(modified_texts) > 1 or prompt.text not in modified_texts

    def test_mutation_avoids_noop_for_short_seed(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Test that weak short prompts still receive a meaningful mutation."""
        prompt = Prompt(text="Solve: {input}")

        mutated = strategy._mutate_prompt(prompt)

        assert mutated.text != prompt.text
        assert "{input}" in mutated.text

    def test_initialize_generates_unique_variants_for_short_seed(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Test that initialization does not collapse into duplicate prompts."""
        prompt = Prompt(text="Solve: {input}")

        population = strategy.initialize(prompt, population_size=6)

        texts = [individual.prompt.text for individual in population]
        assert len(set(texts)) == len(texts)

    def test_crossover_produces_offspring(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Test that crossover produces offspring."""
        parent1 = Prompt(text="First sentence. Second sentence. Third sentence.")
        parent2 = Prompt(text="Alpha part. Beta part. Gamma part.")

        child1, child2 = strategy._crossover(parent1, parent2)

        # Children should be different from parents
        parent_texts = {parent1.text, parent2.text}
        assert child1.text not in parent_texts or child2.text not in parent_texts

    def test_tournament_selection(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Test tournament selection tends to select fitter individuals."""
        prompt = Prompt(text="test")
        individuals = [
            Individual(prompt=prompt, fitness=float(i))
            for i in range(10)
        ]

        # Run many selections
        selected_fitness = []
        for _ in range(100):
            selected = strategy._tournament_select(individuals)
            selected_fitness.append(selected.fitness or 0.0)

        avg_selected = sum(selected_fitness) / len(selected_fitness)

        # Average selection should be above average fitness (4.5)
        assert avg_selected > 5.0  # Tournament bias toward higher fitness

    def test_should_terminate_max_generations(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Test termination on max generations."""
        history = OptimizationHistory()

        # Add generations
        from promptfoundry.core.history import GenerationRecord

        for i in range(100):
            history.generations.append(
                GenerationRecord(
                    generation=i,
                    best_fitness=0.8,
                    average_fitness=0.6,
                    best_prompt="test",
                    population_size=10,
                )
            )

        assert strategy.should_terminate(history, max_generations=100, patience=10)

    def test_should_terminate_no_improvement(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Test termination on no improvement."""
        history = OptimizationHistory()

        from promptfoundry.core.history import GenerationRecord

        # First generation with high fitness
        history.generations.append(
            GenerationRecord(
                generation=0, best_fitness=0.9, average_fitness=0.7,
                best_prompt="best", population_size=10
            )
        )

        # Following generations with lower fitness
        for i in range(1, 15):
            history.generations.append(
                GenerationRecord(
                    generation=i, best_fitness=0.7, average_fitness=0.5,
                    best_prompt="worse", population_size=10
                )
            )

        # Should terminate due to no improvement for 10 generations
        assert strategy.should_terminate(history, max_generations=100, patience=10)
