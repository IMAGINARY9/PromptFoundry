"""Task and example models.

This module defines the Task and Example classes that represent
optimization task definitions and their associated data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Example:
    """Single input-output example for task evaluation.

    Attributes:
        input: The input text to provide to the LLM.
        expected_output: The expected/desired output.
        metadata: Additional metadata (e.g., category, difficulty).
    """

    input: str
    expected_output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a brief representation."""
        input_preview = self.input[:50] + "..." if len(self.input) > 50 else self.input
        return f"Example(input={input_preview!r})"


@dataclass
class Task:
    """Optimization task definition.

    A task consists of a name, optional system prompt, and a collection
    of input-output examples used for training and evaluation.

    Attributes:
        name: Human-readable task name.
        examples: List of training examples.
        system_prompt: Optional system prompt for the LLM.
        validation_examples: Optional separate validation set.
        metadata: Additional task metadata.
    """

    name: str
    examples: list[Example]
    system_prompt: str | None = None
    validation_examples: list[Example] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task configuration."""
        if len(self.examples) < 1:
            raise ValueError("Task must have at least 1 example")

    def __len__(self) -> int:
        """Return the number of training examples."""
        return len(self.examples)

    @classmethod
    def from_file(cls, path: str | Path) -> Task:
        """Load a task from a YAML file.

        Args:
            path: Path to the YAML task file.

        Returns:
            A Task instance loaded from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid task file format: {path}")

        # Parse examples
        examples = []
        for ex in data.get("examples", []):
            examples.append(
                Example(
                    input=ex["input"],
                    expected_output=ex["output"],
                    metadata=ex.get("metadata", {}),
                )
            )

        # Parse validation examples if present
        validation_examples = None
        if "validation_examples" in data:
            validation_examples = []
            for ex in data["validation_examples"]:
                validation_examples.append(
                    Example(
                        input=ex["input"],
                        expected_output=ex["output"],
                        metadata=ex.get("metadata", {}),
                    )
                )

        return cls(
            name=data.get("name", path.stem),
            examples=examples,
            system_prompt=data.get("system_prompt"),
            validation_examples=validation_examples,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert task to a dictionary representation.

        Returns:
            Dictionary suitable for YAML serialization.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "examples": [
                {
                    "input": ex.input,
                    "output": ex.expected_output,
                    **({"metadata": ex.metadata} if ex.metadata else {}),
                }
                for ex in self.examples
            ],
        }

        if self.system_prompt:
            result["system_prompt"] = self.system_prompt

        if self.validation_examples:
            result["validation_examples"] = [
                {
                    "input": ex.input,
                    "output": ex.expected_output,
                    **({"metadata": ex.metadata} if ex.metadata else {}),
                }
                for ex in self.validation_examples
            ]

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def save(self, path: str | Path) -> None:
        """Save task to a YAML file.

        Args:
            path: Path to save the task file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def split(
        self, validation_ratio: float = 0.2, seed: int | None = None
    ) -> tuple[list[Example], list[Example]]:
        """Split examples into training and validation sets.

        Args:
            validation_ratio: Fraction of examples for validation (0.0-1.0).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (training_examples, validation_examples).
        """
        import random

        if not 0.0 <= validation_ratio <= 1.0:
            raise ValueError("validation_ratio must be between 0.0 and 1.0")

        examples = list(self.examples)
        if seed is not None:
            random.seed(seed)
        random.shuffle(examples)

        split_idx = int(len(examples) * (1 - validation_ratio))
        return examples[:split_idx], examples[split_idx:]
