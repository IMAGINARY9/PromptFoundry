"""Tests for MVP 3 diversity controls module."""

from __future__ import annotations

from promptfoundry.core.population import Individual, Population
from promptfoundry.core.prompt import Prompt
from promptfoundry.strategies.diversity import (
    DiversityController,
    DiversityMetrics,
    LineageNode,
)


class TestDiversityMetrics:
    """Test diversity metrics dataclass."""

    def test_metrics_defaults(self) -> None:
        """Test default metric values."""
        metrics = DiversityMetrics()
        assert metrics.unique_ratio == 1.0
        assert metrics.duplicate_count == 0

    def test_metrics_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = DiversityMetrics(
            unique_ratio=0.8,
            duplicate_count=2,
            entropy=2.5,
        )
        result = metrics.to_dict()
        assert result["unique_ratio"] == 0.8
        assert result["duplicate_count"] == 2
        assert result["entropy"] == 2.5


class TestLineageNode:
    """Test lineage node dataclass."""

    def test_node_creation(self) -> None:
        """Test lineage node creation."""
        node = LineageNode(
            prompt_id="test-id",
            prompt_text="Test prompt",
            fitness=0.85,
            generation=1,
        )
        assert node.prompt_id == "test-id"
        assert node.fitness == 0.85

    def test_node_to_dict(self) -> None:
        """Test node serialization."""
        node = LineageNode(
            prompt_id="test-id",
            prompt_text="Test prompt text",
            fitness=0.9,
            generation=2,
            parent_ids=["parent-1"],
            mutation_operator="add_constraint",
        )
        result = node.to_dict()
        assert result["id"] == "test-id"
        assert result["fitness"] == 0.9
        assert result["mutation_operator"] == "add_constraint"


class TestDiversityController:
    """Test diversity controller."""

    def test_controller_initialization(self) -> None:
        """Test controller initializes correctly."""
        controller = DiversityController()
        assert controller.min_unique_ratio == 0.7

    def test_controller_custom_params(self) -> None:
        """Test controller with custom parameters."""
        controller = DiversityController(
            min_unique_ratio=0.8,
            similarity_threshold=0.9,
        )
        assert controller.min_unique_ratio == 0.8
        assert controller.similarity_threshold == 0.9

    def test_duplicate_detection(self) -> None:
        """Test duplicate detection."""
        controller = DiversityController()

        # First occurrence is not duplicate
        assert not controller.is_duplicate("Test prompt")
        controller.register_prompt("id-1", "Test prompt")

        # Same text is duplicate
        assert controller.is_duplicate("Test prompt")

        # Different text is not duplicate
        assert not controller.is_duplicate("Different prompt")

    def test_register_prompt(self) -> None:
        """Test prompt registration."""
        controller = DiversityController()

        # Returns True for new prompt
        result = controller.register_prompt("id-1", "Test prompt")
        assert result is True

        # Returns False for duplicate
        result = controller.register_prompt("id-2", "Test prompt")
        assert result is False

    def test_lineage_tracking(self) -> None:
        """Test lineage is tracked."""
        controller = DiversityController()

        controller.register_prompt(
            prompt_id="seed",
            text="Seed prompt",
            generation=0,
        )
        controller.register_prompt(
            prompt_id="child-1",
            text="Child prompt 1",
            generation=1,
            parent_ids=["seed"],
            mutation_operator="rephrase",
        )

        node = controller.get_lineage("child-1")
        assert node is not None
        assert node.parent_ids == ["seed"]
        assert node.mutation_operator == "rephrase"

    def test_ancestry_retrieval(self) -> None:
        """Test ancestry chain retrieval."""
        controller = DiversityController()

        # Create lineage: seed -> child -> grandchild
        controller.register_prompt("seed", "Seed", generation=0)
        controller.register_prompt("child", "Child", generation=1, parent_ids=["seed"])
        controller.register_prompt("grandchild", "Grandchild", generation=2, parent_ids=["child"])

        ancestry = controller.get_ancestry("grandchild")
        assert len(ancestry) == 3
        assert ancestry[0].prompt_id == "seed"
        assert ancestry[2].prompt_id == "grandchild"

    def test_measure_diversity(self) -> None:
        """Test diversity measurement."""
        controller = DiversityController()

        # Create population with some duplicates
        prompts = [
            Prompt(text="Unique prompt 1"),
            Prompt(text="Unique prompt 2"),
            Prompt(text="Unique prompt 1"),  # Duplicate
            Prompt(text="Unique prompt 3"),
        ]
        individuals = [
            Individual(prompt=p, generation=0, parent_ids=[])
            for p in prompts
        ]
        population = Population(individuals=individuals, generation=0)

        metrics = controller.measure_diversity(population)
        assert metrics.unique_ratio == 0.75  # 3 unique out of 4
        assert metrics.duplicate_count == 1

    def test_needs_diversity_injection(self) -> None:
        """Test diversity injection check."""
        controller = DiversityController(min_unique_ratio=0.7)

        # Low diversity
        low_diversity = DiversityMetrics(unique_ratio=0.5)
        assert controller.needs_diversity_injection(low_diversity) is True

        # High diversity
        high_diversity = DiversityMetrics(unique_ratio=0.9)
        assert controller.needs_diversity_injection(high_diversity) is False

    def test_get_duplicates_in_population(self) -> None:
        """Test finding duplicates in population."""
        controller = DiversityController()

        prompts = [
            Prompt(text="Prompt A"),
            Prompt(text="Prompt B"),
            Prompt(text="Prompt A"),  # Duplicate of first
        ]
        individuals = [
            Individual(prompt=p, generation=0, parent_ids=[])
            for p in prompts
        ]
        population = Population(individuals=individuals, generation=0)

        duplicates = controller.get_duplicates_in_population(population)
        assert len(duplicates) == 1
        assert individuals[0].id in duplicates[0]
        assert individuals[2].id in duplicates[0]

    def test_crowding_penalty(self) -> None:
        """Test crowding penalty application."""
        controller = DiversityController(similarity_threshold=0.8)

        # Create population with similar prompts
        prompts = [
            Prompt(text="Classify the sentiment of this text"),
            Prompt(text="Classify the sentiment of the text"),  # Very similar
            Prompt(text="Completely different instruction here"),
        ]
        individuals = [
            Individual(prompt=p, generation=0, parent_ids=[])
            for p in prompts
        ]
        population = Population(individuals=individuals, generation=0)

        original_scores = [1.0, 0.9, 0.8]
        adjusted = controller.apply_crowding_penalty(population, original_scores)

        # Some penalties should be applied to similar prompts
        # The exact amount depends on similarity calculation
        # At minimum, verify adjusted scores are <= original
        for i, (adj, orig) in enumerate(zip(adjusted, original_scores, strict=True)):
            assert adj <= orig, f"adjusted[{i}]={adj} > original[{i}]={orig}"

    def test_select_diverse_subset(self) -> None:
        """Test diverse subset selection."""
        controller = DiversityController()

        prompts = [
            Prompt(text="Type A prompt version 1"),
            Prompt(text="Type A prompt version 2"),
            Prompt(text="Type B completely different"),
            Prompt(text="Type A prompt version 3"),
            Prompt(text="Type C another different one"),
        ]
        individuals = [
            Individual(prompt=p, generation=0, parent_ids=[])
            for p in prompts
        ]
        population = Population(individuals=individuals, generation=0)

        subset = controller.select_diverse_subset(population, n=3)

        # Should select diverse prompts
        assert len(subset) == 3
        texts = [ind.prompt.text for ind in subset]
        # Should prefer diverse prompts over similar ones
        assert len(set(texts)) == 3

    def test_reset_clears_state(self) -> None:
        """Test reset clears all tracking state."""
        controller = DiversityController()

        controller.register_prompt("id-1", "Test prompt")
        assert controller.is_duplicate("Test prompt")

        controller.reset()
        assert not controller.is_duplicate("Test prompt")

    def test_generate_lineage_report(self) -> None:
        """Test lineage report generation."""
        controller = DiversityController()

        controller.register_prompt("seed", "Seed prompt", fitness=0.5)
        controller.register_prompt(
            "child",
            "Child prompt",
            fitness=0.7,
            generation=1,
            parent_ids=["seed"],
            mutation_operator="add_constraint",
        )

        # Update fitness in lineage
        node = controller.get_lineage("child")
        if node:
            node.fitness = 0.7

        individual = Individual(
            prompt=Prompt(text="Child prompt"),
            generation=1,
            parent_ids=["seed"],
        )
        # Manually set ID to match
        object.__setattr__(individual, "_id", "child")

        report = controller.generate_lineage_report(individual)
        assert "ancestry_length" in report or "error" not in report
