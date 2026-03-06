"""Tests for core domain models."""

from __future__ import annotations

import pytest

from promptfoundry.core.prompt import Prompt, PromptTemplate
from promptfoundry.core.task import Example, Task
from promptfoundry.core.population import Individual, Population


class TestPrompt:
    """Tests for Prompt class."""

    def test_prompt_creation(self) -> None:
        """Test basic prompt creation."""
        prompt = Prompt(text="Hello world")
        assert prompt.text == "Hello world"
        assert len(prompt.id) == 8  # UUID truncated

    def test_prompt_immutability(self) -> None:
        """Test that prompts are immutable (frozen dataclass)."""
        prompt = Prompt(text="Test")
        with pytest.raises(Exception):  # FrozenInstanceError
            prompt.text = "Modified"  # type: ignore

    def test_prompt_with_text(self) -> None:
        """Test creating a new prompt with modified text."""
        original = Prompt(text="Original", metadata={"key": "value"})
        modified = original.with_text("Modified")

        assert modified.text == "Modified"
        assert original.text == "Original"  # Original unchanged
        assert modified.metadata["parent_id"] == original.id

    def test_prompt_str(self) -> None:
        """Test string representation."""
        prompt = Prompt(text="Hello")
        assert str(prompt) == "Hello"

    def test_prompt_len(self) -> None:
        """Test length calculation."""
        prompt = Prompt(text="Hello")
        assert len(prompt) == 5


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_template_variable_extraction(self) -> None:
        """Test automatic variable extraction."""
        template = PromptTemplate(template="Hello {name}, you are {age} years old.")
        assert template.variables == ["name", "age"]

    def test_template_render(self) -> None:
        """Test template rendering."""
        template = PromptTemplate(template="Hello {name}!")
        prompt = template.render(name="World")

        assert prompt.text == "Hello World!"
        assert prompt.metadata["template"] == "Hello {name}!"

    def test_template_render_with_defaults(self) -> None:
        """Test rendering with default values."""
        template = PromptTemplate(
            template="Format: {format}",
            default_values={"format": "JSON"},
        )
        prompt = template.render()
        assert prompt.text == "Format: JSON"

    def test_template_render_missing_variable(self) -> None:
        """Test error on missing required variable."""
        template = PromptTemplate(template="{a} and {b}")
        with pytest.raises(ValueError, match="Missing required variables"):
            template.render(a="A")

    def test_template_validate(self) -> None:
        """Test template validation."""
        valid = PromptTemplate(template="Hello {name}!")
        assert valid.validate() == []

        invalid = PromptTemplate(template="Hello {name")
        errors = invalid.validate()
        assert len(errors) > 0


class TestExample:
    """Tests for Example class."""

    def test_example_creation(self) -> None:
        """Test basic example creation."""
        example = Example(input="test input", expected_output="test output")
        assert example.input == "test input"
        assert example.expected_output == "test output"

    def test_example_with_metadata(self) -> None:
        """Test example with metadata."""
        example = Example(
            input="test",
            expected_output="test",
            metadata={"difficulty": "easy"},
        )
        assert example.metadata["difficulty"] == "easy"


class TestTask:
    """Tests for Task class."""

    def test_task_creation(self, sample_examples: list[Example]) -> None:
        """Test basic task creation."""
        task = Task(name="test", examples=sample_examples)
        assert task.name == "test"
        assert len(task) == 5

    def test_task_empty_examples(self) -> None:
        """Test error on empty examples."""
        with pytest.raises(ValueError, match="at least 1 example"):
            Task(name="test", examples=[])

    def test_task_to_dict(self, sample_task: Task) -> None:
        """Test task serialization."""
        data = sample_task.to_dict()
        assert data["name"] == "test_sentiment"
        assert len(data["examples"]) == 5

    def test_task_split(self, sample_examples: list[Example]) -> None:
        """Test train/validation split."""
        task = Task(name="test", examples=sample_examples)
        train, val = task.split(validation_ratio=0.4, seed=42)

        assert len(train) + len(val) == 5
        assert len(val) == 2  # 40% of 5 = 2


class TestIndividual:
    """Tests for Individual class."""

    def test_individual_creation(self, sample_prompt: Prompt) -> None:
        """Test basic individual creation."""
        ind = Individual(prompt=sample_prompt)
        assert ind.prompt == sample_prompt
        assert ind.fitness is None
        assert ind.generation == 0

    def test_individual_with_fitness(self, sample_prompt: Prompt) -> None:
        """Test creating individual with fitness."""
        ind = Individual(prompt=sample_prompt)
        scored = ind.with_fitness(0.85)

        assert scored.fitness == 0.85
        assert ind.fitness is None  # Original unchanged

    def test_individual_comparison(self, sample_prompt: Prompt) -> None:
        """Test individual comparison by fitness."""
        ind1 = Individual(prompt=sample_prompt, fitness=0.5)
        ind2 = Individual(prompt=sample_prompt, fitness=0.8)
        ind3 = Individual(prompt=sample_prompt, fitness=None)

        assert ind1 < ind2  # Lower fitness is "less than"
        assert ind3 < ind1  # None fitness is worst

    def test_individual_to_dict(self, sample_individual: Individual) -> None:
        """Test individual serialization."""
        data = sample_individual.to_dict()
        assert "id" in data
        assert "prompt_text" in data
        assert data["fitness"] == 0.8


class TestPopulation:
    """Tests for Population class."""

    def test_population_creation(self, sample_population: Population) -> None:
        """Test basic population creation."""
        assert len(sample_population) == 5
        assert sample_population.generation == 1

    def test_population_best(self, sample_population: Population) -> None:
        """Test finding best individual."""
        best = sample_population.best
        assert best is not None
        assert best.fitness == 0.9  # Highest fitness in fixture

    def test_population_average_fitness(self, sample_population: Population) -> None:
        """Test average fitness calculation."""
        avg = sample_population.average_fitness
        assert avg is not None
        assert 0.5 < avg < 1.0

    def test_population_sorted(self, sample_population: Population) -> None:
        """Test sorting by fitness."""
        sorted_inds = sample_population.sorted_by_fitness()
        fitnesses = [ind.fitness for ind in sorted_inds]

        # Should be descending
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_population_iteration(self, sample_population: Population) -> None:
        """Test iteration over population."""
        count = sum(1 for _ in sample_population)
        assert count == 5

    def test_population_to_dict(self, sample_population: Population) -> None:
        """Test population serialization."""
        data = sample_population.to_dict()
        assert data["generation"] == 1
        assert data["size"] == 5
        assert len(data["individuals"]) == 5
