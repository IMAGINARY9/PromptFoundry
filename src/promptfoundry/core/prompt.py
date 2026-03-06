"""Prompt and template models.

This module defines the core Prompt and PromptTemplate classes used
throughout the optimization process.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Prompt:
    """Immutable prompt representation.

    Attributes:
        text: The prompt text content.
        id: Unique identifier for this prompt instance.
        metadata: Additional metadata (e.g., source, generation).
    """

    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return the prompt text."""
        return self.text

    def __len__(self) -> int:
        """Return the length of the prompt text."""
        return len(self.text)

    def with_text(self, text: str) -> Prompt:
        """Create a new Prompt with modified text, preserving metadata."""
        return Prompt(
            text=text,
            metadata={**self.metadata, "parent_id": self.id},
        )


@dataclass
class PromptTemplate:
    """Prompt with variable placeholders.

    Templates use {variable_name} syntax for placeholders.

    Attributes:
        template: The template string with placeholders.
        variables: List of variable names found in the template.
        default_values: Default values for variables.
    """

    template: str
    variables: list[str] = field(default_factory=list)
    default_values: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract variables from template if not provided."""
        if not self.variables:
            # Extract {variable_name} patterns
            self.variables = re.findall(r"\{(\w+)\}", self.template)

    def render(self, **kwargs: str) -> Prompt:
        """Render template with variables.

        Args:
            **kwargs: Variable values to substitute.

        Returns:
            A Prompt with all variables substituted.

        Raises:
            ValueError: If required variables are missing.
        """
        # Merge defaults with provided values
        values = {**self.default_values, **kwargs}

        # Check for missing required variables
        missing = set(self.variables) - set(values.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        text = self.template
        for var, value in values.items():
            text = text.replace(f"{{{var}}}", value)

        return Prompt(
            text=text,
            metadata={"template": self.template, "variables": kwargs},
        )

    def validate(self) -> list[str]:
        """Validate template syntax.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Check for unbalanced braces
        open_count = self.template.count("{")
        close_count = self.template.count("}")
        if open_count != close_count:
            errors.append(f"Unbalanced braces: {open_count} '{{' vs {close_count} '}}'")

        # Check for empty variable names
        if re.search(r"\{\s*\}", self.template):
            errors.append("Empty variable name found: {}")

        return errors
