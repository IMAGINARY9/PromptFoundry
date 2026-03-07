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

    def test_evolve_small_population_does_not_freeze_with_elitism(
        self, seed_prompt: Prompt
    ) -> None:
        """Small populations should still create at least one new prompt."""
        strategy = GeneticAlgorithmStrategy(
            EvolutionaryConfig(
                population_size=2,
                mutation_rate=1.0,
                crossover_rate=0.0,
                tournament_size=2,
                elitism=2,
                seed=42,
            )
        )

        population = strategy.initialize(seed_prompt, population_size=2)
        original_texts = [individual.prompt.text for individual in population]

        new_population = strategy.evolve(population, [0.8, 0.7])
        new_texts = [individual.prompt.text for individual in new_population]

        assert len(new_population) == 2
        assert new_population.generation == 1
        assert any(text not in original_texts for text in new_texts)

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

    def test_mutation_operator_catalog_uses_semantic_transforms(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Mutation operators should favor semantic prompt changes over word shuffling."""
        names = {operator.name for operator in strategy._get_mutation_operators()}

        assert "promote_structured_layout" in names
        assert "add_answer_only_directive" in names
        assert "swap_words" not in names

    def test_operator_feedback_tracks_attempts_and_wins(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Successful mutations should be reflected in aggregate operator stats."""
        population = Population(
            individuals=[
                Individual(
                    prompt=Prompt(
                        text="Mutated prompt",
                        metadata={
                            "mutation_operator": "add_answer_only_directive",
                            "parent_baseline_fitness": 0.2,
                        },
                    ),
                    generation=1,
                ),
                Individual(prompt=Prompt(text="Unchanged"), generation=1),
            ],
            generation=1,
        )

        strategy.record_generation_feedback(population, [0.7, 0.3])
        stats = strategy.get_operator_stats()["add_answer_only_directive"]

        assert stats["attempts"] == 1.0
        assert stats["wins"] == 1.0
        assert stats["avg_delta"] == pytest.approx(0.5)

    def test_operator_feedback_updates_adaptive_weights(
        self, seed_prompt: Prompt
    ) -> None:
        """Adaptive weighting should reward operators that improve fitness."""
        strategy = GeneticAlgorithmStrategy(
            EvolutionaryConfig(
                population_size=4,
                mutation_rate=1.0,
                crossover_rate=0.0,
                tournament_size=2,
                elitism=1,
                seed=42,
                adaptive_mutation_weights=True,
            )
        )

        mutated = Prompt(
            text="Improved prompt",
            metadata={
                "mutation_operator": "add_answer_only_directive",
                "parent_baseline_fitness": 0.1,
            },
        )
        population = Population(
            individuals=[
                Individual(prompt=seed_prompt, generation=1),
                Individual(prompt=mutated, generation=1),
            ],
            generation=1,
        )

        before = strategy.get_operator_stats()["add_answer_only_directive"]["current_weight"]
        strategy.record_generation_feedback(population, [0.1, 0.9])
        after = strategy.get_operator_stats()["add_answer_only_directive"]["current_weight"]

        assert after > before

    def test_structured_layout_mutation_preserves_placeholder(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Structured mutations should preserve the input placeholder and add output guidance."""
        mutated = strategy._mutate_promote_structured_layout("Answer the question: {input}")

        assert "{input}" in mutated
        assert "Return only the final answer." in mutated or "Answer with only the final answer." in mutated
        assert "Input:" in mutated or "Question:" in mutated or "Task Input:" in mutated
        assert ": ." not in mutated

    def test_placeholder_is_preserved_during_mutation(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Mutations should not accidentally drop the input placeholder."""
        prompt = Prompt(text="Classify sentiment: {input}. Return one label.")

        for _ in range(20):
            mutated = strategy._mutate_prompt(prompt)
            assert "{input}" in mutated.text

    def test_numeric_mutation_adds_digits_only_guidance(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Arithmetic prompts should receive numeric-output guidance."""
        mutated = strategy._mutate_add_numeric_constraint("Answer the question: What is 7 minus 2?")

        assert mutated != "Answer the question: What is 7 minus 2?"
        assert "number" in mutated.lower() or "digits only" in mutated.lower() or "numeric" in mutated.lower()

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

    def test_crossover_preserves_placeholder_when_parents_have_it(
        self, strategy: GeneticAlgorithmStrategy
    ) -> None:
        """Crossover should retain the input placeholder for prompt templates."""
        parent1 = Prompt(text="Classify sentiment: {input}. Return one label.")
        parent2 = Prompt(text="Analyze review: {input}. Respond with only the final answer.")

        child1, child2 = strategy._crossover(parent1, parent2)

        assert "{input}" in child1.text
        assert "{input}" in child2.text

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
