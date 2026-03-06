"""Runtime configuration with profile presets.

This module provides a unified configuration system for optimization runtime,
supporting preset profiles and YAML/CLI override precedence.

Profile Hierarchy (lowest to highest priority):
1. Default profile values
2. YAML config file
3. Explicit profile selection (--profile)
4. CLI argument overrides
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import yaml


class RuntimeProfile(str, Enum):
    """Predefined runtime profiles for different deployment scenarios.

    - SLOW_LOCAL: Single-threaded local model, conservative settings
    - BALANCED: Moderate concurrency, good for medium-speed backends
    - THROUGHPUT: High concurrency for fast API backends
    """

    SLOW_LOCAL = "slow-local"
    BALANCED = "balanced"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration for optimization.

    Attributes:
        profile: The runtime profile being used.
        max_generations: Maximum optimization generations.
        population_size: Number of prompts per generation.
        patience: Generations without improvement before early stop.
        max_concurrency: Maximum parallel LLM requests.
        timeout_per_request: Timeout in seconds for each LLM call.
        batch_size: Number of prompts to evaluate per batch.
        checkpoint_frequency: Save checkpoint every N generations (0 to disable).
        runtime_budget_seconds: Maximum total runtime (0 for unlimited).
        seed: Random seed for reproducibility (None for random).
    """

    profile: RuntimeProfile = RuntimeProfile.BALANCED
    max_generations: int = 50
    population_size: int = 10
    patience: int = 10
    max_concurrency: int = 4
    timeout_per_request: float = 60.0
    batch_size: int = 5
    checkpoint_frequency: int = 5
    runtime_budget_seconds: float = 0.0
    seed: int | None = None

    # Profile presets as class variables
    _PROFILES: ClassVar[dict[RuntimeProfile, dict[str, Any]]] = {
        RuntimeProfile.SLOW_LOCAL: {
            "max_generations": 20,
            "population_size": 3,
            "patience": 8,
            "max_concurrency": 1,
            "timeout_per_request": 120.0,
            "batch_size": 1,
            "checkpoint_frequency": 2,
            "runtime_budget_seconds": 0.0,
        },
        RuntimeProfile.BALANCED: {
            "max_generations": 50,
            "population_size": 8,
            "patience": 10,
            "max_concurrency": 4,
            "timeout_per_request": 60.0,
            "batch_size": 4,
            "checkpoint_frequency": 5,
            "runtime_budget_seconds": 0.0,
        },
        RuntimeProfile.THROUGHPUT: {
            "max_generations": 100,
            "population_size": 20,
            "patience": 15,
            "max_concurrency": 16,
            "timeout_per_request": 30.0,
            "batch_size": 10,
            "checkpoint_frequency": 10,
            "runtime_budget_seconds": 0.0,
        },
    }

    @classmethod
    def from_profile(cls, profile: RuntimeProfile | str) -> RuntimeConfig:
        """Create configuration from a named profile.

        Args:
            profile: Profile name or enum value.

        Returns:
            RuntimeConfig with profile defaults.

        Raises:
            ValueError: If profile name is unknown.
        """
        if isinstance(profile, str):
            try:
                profile = RuntimeProfile(profile)
            except ValueError:
                valid = [p.value for p in RuntimeProfile if p != RuntimeProfile.CUSTOM]
                raise ValueError(f"Unknown profile: {profile}. Valid profiles: {valid}") from None

        if profile == RuntimeProfile.CUSTOM:
            return cls(profile=RuntimeProfile.CUSTOM)

        preset = cls._PROFILES.get(profile, {})
        return cls(profile=profile, **preset)

    @classmethod
    def from_yaml(cls, path: Path | str) -> RuntimeConfig:
        """Load configuration from a YAML file.

        The YAML should have an 'optimization' section with settings.

        Args:
            path: Path to YAML config file.

        Returns:
            RuntimeConfig with YAML values.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeConfig:
        """Create configuration from a dictionary.

        Supports both flat dictionaries and nested 'optimization' sections.

        Args:
            data: Configuration dictionary.

        Returns:
            RuntimeConfig instance.
        """
        # Support nested 'optimization' section from full config
        if "optimization" in data:
            opt_data = data["optimization"]
        else:
            opt_data = data

        # Start with profile if specified
        profile_name = opt_data.get("profile", "balanced")
        try:
            config = cls.from_profile(profile_name)
        except ValueError:
            config = cls(profile=RuntimeProfile.CUSTOM)

        # Override with explicit values
        overrides = {}
        field_map = {
            "max_generations": "max_generations",
            "population_size": "population_size",
            "patience": "patience",
            "max_concurrency": "max_concurrency",
            "timeout_per_request": "timeout_per_request",
            "timeout": "timeout_per_request",  # alias
            "batch_size": "batch_size",
            "checkpoint_frequency": "checkpoint_frequency",
            "runtime_budget_seconds": "runtime_budget_seconds",
            "runtime_budget": "runtime_budget_seconds",  # alias
            "seed": "seed",
        }

        for yaml_key, field_name in field_map.items():
            if yaml_key in opt_data and opt_data[yaml_key] is not None:
                overrides[field_name] = opt_data[yaml_key]

        if overrides:
            # If we have explicit overrides, mark as custom unless profile was explicit
            if "profile" not in opt_data and overrides:
                overrides["profile"] = RuntimeProfile.CUSTOM
            return replace(config, **overrides)

        return config

    def with_overrides(self, **kwargs: Any) -> RuntimeConfig:
        """Create a new config with specific values overridden.

        Args:
            **kwargs: Fields to override.

        Returns:
            New RuntimeConfig with overrides applied.
        """
        # Filter out None values (CLI passes None for unset options)
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        if not filtered:
            return self

        # Mark as custom if any overrides are made
        if "profile" not in filtered:
            filtered["profile"] = RuntimeProfile.CUSTOM

        return replace(self, **filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of config.
        """
        return {
            "profile": self.profile.value,
            "max_generations": self.max_generations,
            "population_size": self.population_size,
            "patience": self.patience,
            "max_concurrency": self.max_concurrency,
            "timeout_per_request": self.timeout_per_request,
            "batch_size": self.batch_size,
            "checkpoint_frequency": self.checkpoint_frequency,
            "runtime_budget_seconds": self.runtime_budget_seconds,
            "seed": self.seed,
        }

    def describe(self) -> str:
        """Return a human-readable description of this configuration.

        Returns:
            Multi-line description string.
        """
        lines = [
            f"Runtime Profile: {self.profile.value}",
            f"  Max Generations: {self.max_generations}",
            f"  Population Size: {self.population_size}",
            f"  Patience: {self.patience}",
            f"  Max Concurrency: {self.max_concurrency}",
            f"  Timeout per Request: {self.timeout_per_request}s",
            f"  Batch Size: {self.batch_size}",
        ]
        if self.runtime_budget_seconds > 0:
            lines.append(f"  Runtime Budget: {self.runtime_budget_seconds}s")
        if self.seed is not None:
            lines.append(f"  Seed: {self.seed}")
        return "\n".join(lines)


def get_available_profiles() -> list[str]:
    """Return list of available profile names.

    Returns:
        List of valid profile name strings.
    """
    return [p.value for p in RuntimeProfile if p != RuntimeProfile.CUSTOM]


def get_profile_description(profile: RuntimeProfile | str) -> str:
    """Get a description of what a profile is optimized for.

    Args:
        profile: Profile name or enum.

    Returns:
        Human-readable description.
    """
    if isinstance(profile, str):
        try:
            profile = RuntimeProfile(profile)
        except ValueError:
            return f"Unknown profile: {profile}"

    descriptions = {
        RuntimeProfile.SLOW_LOCAL: (
            "Optimized for single-threaded local models (e.g., text-generation-webui). "
            "Small population, no concurrency, longer timeouts."
        ),
        RuntimeProfile.BALANCED: (
            "Balanced settings for moderate-speed backends. "
            "Medium population with some concurrency."
        ),
        RuntimeProfile.THROUGHPUT: (
            "Optimized for fast API backends with high rate limits. "
            "Large population, high concurrency, shorter timeouts."
        ),
        RuntimeProfile.CUSTOM: "Custom configuration with user-specified values.",
    }
    return descriptions.get(profile, "Unknown profile")
