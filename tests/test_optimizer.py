"""Tests for Optimizer orchestration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from promptfoundry.core.optimizer import Optimizer, OptimizerConfig
from promptfoundry.core.population import Individual, Population
from promptfoundry.core.prompt import Prompt
from promptfoundry.core.task import Example, Task


class MockStrategy:
    """Mock optimization strategy for testing."""

    def __init__(self) -> None:
        self.initialize_called = False
        self.evolve_count = 0

    def initialize(self, seed_prompt: Prompt, population_size: int) -> Population:
        self.initialize_called = True
        individuals = [
            Individual(prompt=seed_prompt, generation=0)
            for _ in range(population_size)
        ]
        return Population(individuals=individuals, generation=0)

    def evolve(
        self, population: Population, fitness_scores: list[float]
    ) -> Population:
        self.evolve_count += 1
        # Return same population with incremented generation
        individuals = [
            Individual(prompt=ind.prompt, generation=population.generation + 1)
            for ind in population.individuals
        ]
        return Population(individuals=individuals, generation=population.generation + 1)


class MockEvaluator:
    """Mock evaluator for testing."""

    def __init__(self, fixed_score: float = 0.8) -> None:
        self.fixed_score = fixed_score
        self.evaluate_count = 0

    def evaluate(
        self, predicted: str, expected: str, metadata: dict[str, Any] | None = None
    ) -> float:
        self.evaluate_count += 1
        return self.fixed_score

    def evaluate_batch(self, predictions: list[str], expected: list[str]) -> list[float]:
        return [self.evaluate(p, e) for p, e in zip(predictions, expected, strict=True)]

    def aggregate(self, scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "test response") -> None:
        self.response = response
        self.complete_count = 0

    async def complete(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        self.complete_count += 1
        return self.response

    async def complete_batch(
        self, prompts: list[str], system_prompt: str | None = None, **kwargs: Any
    ) -> list[str]:
        return [await self.complete(p) for p in prompts]


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OptimizerConfig()
        assert config.max_generations == 50
        assert config.population_size == 10
        assert config.patience == 10
        assert config.batch_size == 5
        assert config.max_concurrency == 1

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = OptimizerConfig(
            max_generations=20,
            population_size=5,
            patience=3,
        )
        assert config.max_generations == 20
        assert config.patience == 3


class TestOptimizer:
    """Tests for Optimizer."""

    @pytest.fixture
    def seed_prompt(self) -> Prompt:
        """Create a seed prompt for testing."""
        return Prompt(text="Classify the sentiment: {input}")

    @pytest.fixture
    def task(self) -> Task:
        """Create a task for testing."""
        return Task(
            name="test_task",
            examples=[
                Example(input="I love this!", expected_output="positive"),
                Example(input="I hate this!", expected_output="negative"),
            ],
        )

    @pytest.fixture
    def optimizer(self) -> Optimizer:
        """Create an optimizer for testing."""
        return Optimizer(
            strategy=MockStrategy(),
            evaluator=MockEvaluator(fixed_score=0.8),
            llm_client=MockLLMClient(),
            config=OptimizerConfig(
                max_generations=3,
                population_size=2,
                patience=5,
            ),
        )

    @pytest.mark.asyncio
    async def test_optimize_runs(
        self, optimizer: Optimizer, seed_prompt: Prompt, task: Task
    ) -> None:
        """Test that optimization runs to completion."""
        result = await optimizer.optimize(seed_prompt, task)

        assert result is not None
        assert result.total_generations > 0
        assert result.best_score > 0
        assert result.best_prompt is not None

    @pytest.mark.asyncio
    async def test_optimize_tracks_history(
        self, optimizer: Optimizer, seed_prompt: Prompt, task: Task
    ) -> None:
        """Test that optimization records history."""
        result = await optimizer.optimize(seed_prompt, task)

        assert result.history is not None
        assert len(result.history.generations) > 0

    @pytest.mark.asyncio
    async def test_result_to_dict_always_includes_history_generations(
        self, optimizer: Optimizer, seed_prompt: Prompt, task: Task
    ) -> None:
        """Test serialized results always contain nested per-generation history."""
        result = await optimizer.optimize(seed_prompt, task)

        data = result.to_dict()

        assert "history" in data
        assert "generations" in data["history"]
        assert len(data["history"]["generations"]) == result.total_generations
        assert data["seed_fitness"] == pytest.approx(
            result.history.generations[0].best_fitness
        )

    @pytest.mark.asyncio
    async def test_result_to_dict_is_json_serializable(
        self, optimizer: Optimizer, seed_prompt: Prompt, task: Task
    ) -> None:
        """Test serialized optimization results can be written directly to JSON."""
        result = await optimizer.optimize(seed_prompt, task)

        payload = result.to_dict()

        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        assert decoded["history"]["generations"]

    @pytest.mark.asyncio
    async def test_optimize_calls_strategy(
        self, seed_prompt: Prompt, task: Task
    ) -> None:
        """Test that strategy methods are called."""
        strategy = MockStrategy()
        optimizer = Optimizer(
            strategy=strategy,
            evaluator=MockEvaluator(),
            llm_client=MockLLMClient(),
            config=OptimizerConfig(max_generations=2, population_size=2),
        )

        await optimizer.optimize(seed_prompt, task)

        assert strategy.initialize_called
        assert strategy.evolve_count > 0

    @pytest.mark.asyncio
    async def test_optimize_early_stopping(
        self, seed_prompt: Prompt, task: Task
    ) -> None:
        """Test early stopping on perfect score."""
        optimizer = Optimizer(
            strategy=MockStrategy(),
            evaluator=MockEvaluator(fixed_score=1.0),  # Perfect score
            llm_client=MockLLMClient(),
            config=OptimizerConfig(max_generations=100, population_size=2),
        )

        result = await optimizer.optimize(seed_prompt, task)

        # Should stop early due to perfect score
        assert result.total_generations < 100

    @pytest.mark.asyncio
    async def test_callback_invoked(
        self, optimizer: Optimizer, seed_prompt: Prompt, task: Task
    ) -> None:
        """Test that progress callbacks are invoked."""
        callback_calls: list[tuple[int, float, float, str]] = []

        def callback(gen: int, best: float, avg: float, prompt: str) -> None:
            callback_calls.append((gen, best, avg, prompt))

        optimizer.add_callback(callback)
        await optimizer.optimize(seed_prompt, task)

        assert len(callback_calls) > 0
        # Check first callback has expected structure
        gen, best, avg, prompt = callback_calls[0]
        assert gen == 0
        assert isinstance(best, float)
        assert isinstance(avg, float)
        assert isinstance(prompt, str)

    def test_format_prompt_with_placeholder(self) -> None:
        """Test prompt formatting with {input} placeholder."""
        optimizer = Optimizer(
            strategy=MockStrategy(),
            evaluator=MockEvaluator(),
            llm_client=MockLLMClient(),
        )

        prompt = Prompt(text="Classify: {input}")
        example = Example(input="test input", expected_output="positive")

        formatted = optimizer._format_prompt(prompt, example)
        assert formatted == "Classify: test input"

    def test_format_prompt_without_placeholder(self) -> None:
        """Test prompt formatting without placeholder."""
        optimizer = Optimizer(
            strategy=MockStrategy(),
            evaluator=MockEvaluator(),
            llm_client=MockLLMClient(),
        )

        prompt = Prompt(text="Classify the following")
        example = Example(input="test input", expected_output="positive")

        formatted = optimizer._format_prompt(prompt, example)
        assert "test input" in formatted
        assert "Input:" in formatted
