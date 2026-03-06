"""Pytest configuration and fixtures for PromptFoundry tests."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from promptfoundry.core.prompt import Prompt, PromptTemplate
from promptfoundry.core.task import Example, Task
from promptfoundry.core.population import Individual, Population
from promptfoundry.llm.base import BaseLLMClient


# =============================================================================
# Fixtures: Core Domain
# =============================================================================


@pytest.fixture
def sample_prompt() -> Prompt:
    """Create a sample prompt for testing."""
    return Prompt(
        text="Classify the sentiment of the following text: {input}",
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_template() -> PromptTemplate:
    """Create a sample prompt template for testing."""
    return PromptTemplate(
        template="Classify the sentiment: {input}. Respond with: {format}",
        default_values={"format": "positive, negative, or neutral"},
    )


@pytest.fixture
def sample_examples() -> list[Example]:
    """Create sample examples for testing."""
    return [
        Example(input="I love this!", expected_output="positive"),
        Example(input="This is terrible.", expected_output="negative"),
        Example(input="It's okay.", expected_output="neutral"),
        Example(input="Best ever!", expected_output="positive"),
        Example(input="Awful product.", expected_output="negative"),
    ]


@pytest.fixture
def sample_task(sample_examples: list[Example]) -> Task:
    """Create a sample task for testing."""
    return Task(
        name="test_sentiment",
        examples=sample_examples,
        system_prompt="You are a sentiment classifier.",
    )


@pytest.fixture
def sample_individual(sample_prompt: Prompt) -> Individual:
    """Create a sample individual for testing."""
    return Individual(
        prompt=sample_prompt,
        fitness=0.8,
        generation=1,
    )


@pytest.fixture
def sample_population(sample_prompt: Prompt) -> Population:
    """Create a sample population for testing."""
    individuals = [
        Individual(prompt=sample_prompt.with_text(f"Variant {i}: {sample_prompt.text}"), fitness=0.5 + i * 0.1)
        for i in range(5)
    ]
    return Population(individuals=individuals, generation=1)


# =============================================================================
# Fixtures: Mock LLM Client
# =============================================================================


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        """Initialize with optional pre-defined responses."""
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []
        self.default_response = "mock response"

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return mock response."""
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            **kwargs,
        })
        return self.responses.get(prompt, self.default_response)

    async def health_check(self) -> bool:
        """Always return healthy."""
        return True


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client for testing."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_with_responses() -> MockLLMClient:
    """Create a mock LLM client with sentiment responses."""
    return MockLLMClient(
        responses={
            "I love this!": "positive",
            "This is terrible.": "negative",
            "It's okay.": "neutral",
        }
    )


# =============================================================================
# Fixtures: Temporary Files
# =============================================================================


@pytest.fixture
def temp_task_file(tmp_path: Any, sample_examples: list[Example]) -> Any:
    """Create a temporary task YAML file."""
    import yaml

    task_data = {
        "name": "temp_test_task",
        "examples": [
            {"input": ex.input, "output": ex.expected_output}
            for ex in sample_examples
        ],
    }

    task_file = tmp_path / "task.yaml"
    with task_file.open("w") as f:
        yaml.dump(task_data, f)

    return task_file


@pytest.fixture
def temp_config_file(tmp_path: Any) -> Any:
    """Create a temporary config YAML file."""
    import yaml

    config_data = {
        "optimization": {
            "strategy": "evolutionary",
            "max_generations": 10,
            "population_size": 5,
        },
        "llm": {
            "base_url": "http://localhost:5000/v1",
            "api_key": "test",
        },
    }

    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    return config_file
