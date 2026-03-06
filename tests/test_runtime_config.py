"""Tests for RuntimeConfig and profile system.

Tests cover:
- Profile creation and preset values
- YAML loading
- Config precedence (profile -> config file -> CLI overrides)
- RuntimeConfig immutability
"""

import pytest
from pathlib import Path
import tempfile

from promptfoundry.core.config import (
    RuntimeConfig,
    RuntimeProfile,
    get_available_profiles,
    get_profile_description,
)


class TestRuntimeProfile:
    """Tests for RuntimeProfile enum."""

    def test_available_profiles(self) -> None:
        """Test that expected profiles are available."""
        profiles = get_available_profiles()
        assert "slow-local" in profiles
        assert "balanced" in profiles
        assert "throughput" in profiles
        assert "custom" not in profiles  # custom is not a preset

    def test_profile_descriptions(self) -> None:
        """Test that all profiles have descriptions."""
        for profile in RuntimeProfile:
            desc = get_profile_description(profile)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestRuntimeConfigFromProfile:
    """Tests for creating config from profiles."""

    def test_slow_local_profile(self) -> None:
        """Test slow-local profile has expected settings."""
        config = RuntimeConfig.from_profile("slow-local")
        
        assert config.profile == RuntimeProfile.SLOW_LOCAL
        assert config.population_size == 3
        assert config.max_concurrency == 1
        assert config.batch_size == 1
        assert config.timeout_per_request == 120.0

    def test_balanced_profile(self) -> None:
        """Test balanced profile has expected settings."""
        config = RuntimeConfig.from_profile("balanced")
        
        assert config.profile == RuntimeProfile.BALANCED
        assert config.population_size == 8
        assert config.max_concurrency == 4

    def test_throughput_profile(self) -> None:
        """Test throughput profile has expected settings."""
        config = RuntimeConfig.from_profile("throughput")
        
        assert config.profile == RuntimeProfile.THROUGHPUT
        assert config.population_size == 20
        assert config.max_concurrency == 16
        assert config.max_generations == 100

    def test_profile_from_enum(self) -> None:
        """Test creating profile from enum value."""
        config = RuntimeConfig.from_profile(RuntimeProfile.SLOW_LOCAL)
        assert config.profile == RuntimeProfile.SLOW_LOCAL

    def test_unknown_profile_raises(self) -> None:
        """Test that unknown profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            RuntimeConfig.from_profile("nonexistent")


class TestRuntimeConfigFromDict:
    """Tests for creating config from dictionaries."""

    def test_from_flat_dict(self) -> None:
        """Test creating config from flat dict."""
        data = {
            "max_generations": 100,
            "population_size": 15,
            "patience": 20,
        }
        config = RuntimeConfig.from_dict(data)
        
        assert config.max_generations == 100
        assert config.population_size == 15
        assert config.patience == 20

    def test_from_nested_dict(self) -> None:
        """Test creating config from nested dict with 'optimization' key."""
        data = {
            "optimization": {
                "max_generations": 50,
                "max_concurrency": 8,
            },
            "llm": {"base_url": "http://example.com"},
        }
        config = RuntimeConfig.from_dict(data)
        
        assert config.max_generations == 50
        assert config.max_concurrency == 8

    def test_from_dict_with_profile(self) -> None:
        """Test that profile in dict is respected."""
        data = {
            "optimization": {
                "profile": "slow-local",
                "max_generations": 30,  # override profile default
            }
        }
        config = RuntimeConfig.from_dict(data)
        
        # Profile should be slow-local but overridden values should apply  
        assert config.max_generations == 30
        # Non-overridden values come from profile
        assert config.max_concurrency == 1

    def test_alias_support(self) -> None:
        """Test that timeout and runtime_budget aliases work."""
        data = {
            "timeout": 90.0,
            "runtime_budget": 3600.0,
        }
        config = RuntimeConfig.from_dict(data)
        
        assert config.timeout_per_request == 90.0
        assert config.runtime_budget_seconds == 3600.0


class TestRuntimeConfigFromYaml:
    """Tests for loading config from YAML files."""

    def test_from_yaml_file(self) -> None:
        """Test loading config from YAML file."""
        yaml_content = """
optimization:
  profile: slow-local
  max_generations: 25
  population_size: 5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = RuntimeConfig.from_yaml(f.name)
            
            assert config.max_generations == 25
            assert config.population_size == 5
            # slow-local defaults for non-overridden values
            assert config.max_concurrency == 1

    def test_from_yaml_empty_file(self) -> None:
        """Test loading from empty YAML file returns defaults."""
        yaml_content = ""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = RuntimeConfig.from_yaml(f.name)
            
            # Should use balanced defaults
            assert config.profile == RuntimeProfile.BALANCED


class TestRuntimeConfigOverrides:
    """Tests for config override functionality."""

    def test_with_overrides_basic(self) -> None:
        """Test that with_overrides creates new config."""
        config = RuntimeConfig.from_profile("balanced")
        new_config = config.with_overrides(max_generations=200)
        
        # New config should have override
        assert new_config.max_generations == 200
        # Original should be unchanged (frozen dataclass)
        assert config.max_generations == 50

    def test_with_overrides_none_values(self) -> None:
        """Test that None values are filtered out."""
        config = RuntimeConfig.from_profile("balanced")
        new_config = config.with_overrides(
            max_generations=None,
            population_size=15,
        )
        
        # None should not override
        assert new_config.max_generations == config.max_generations
        # Non-None should override
        assert new_config.population_size == 15

    def test_with_overrides_marks_custom(self) -> None:
        """Test that overrides mark profile as custom."""
        config = RuntimeConfig.from_profile("balanced")
        new_config = config.with_overrides(max_generations=200)
        
        assert new_config.profile == RuntimeProfile.CUSTOM

    def test_no_overrides_returns_same(self) -> None:
        """Test that no overrides returns same config."""
        config = RuntimeConfig.from_profile("balanced")
        same_config = config.with_overrides()
        
        assert same_config is config


class TestRuntimeConfigPrecedence:
    """Tests for config precedence rules."""

    def test_cli_overrides_yaml(self) -> None:
        """Test CLI args override YAML config."""
        # Simulate: YAML sets values, then CLI overrides some
        yaml_config = RuntimeConfig.from_dict({
            "optimization": {
                "profile": "balanced",
                "max_generations": 50,
                "population_size": 10,
            }
        })
        
        # CLI override
        final_config = yaml_config.with_overrides(max_generations=100)
        
        assert final_config.max_generations == 100  # CLI override
        assert final_config.population_size == 10  # From YAML

    def test_yaml_overrides_profile(self) -> None:
        """Test YAML values override profile defaults."""
        config = RuntimeConfig.from_dict({
            "optimization": {
                "profile": "slow-local",
                "max_concurrency": 4,  # Override slow-local default of 1
            }
        })
        
        assert config.max_concurrency == 4

    def test_full_precedence_chain(self) -> None:
        """Test complete precedence: profile -> yaml -> cli."""
        # 1. Start with profile
        base_config = RuntimeConfig.from_profile("slow-local")
        assert base_config.population_size == 3
        
        # 2. Apply YAML overrides (simulated)
        yaml_overrides = {"population_size": 6, "patience": 15}
        yaml_config = base_config.with_overrides(**yaml_overrides)
        assert yaml_config.population_size == 6
        assert yaml_config.patience == 15
        
        # 3. Apply CLI overrides
        cli_config = yaml_config.with_overrides(population_size=10)
        assert cli_config.population_size == 10  # CLI wins
        assert cli_config.patience == 15  # From YAML
        assert cli_config.max_concurrency == 1  # From profile


class TestRuntimeConfigSerialization:
    """Tests for config serialization."""

    def test_to_dict(self) -> None:
        """Test converting config to dict."""
        config = RuntimeConfig.from_profile("balanced")
        data = config.to_dict()
        
        assert data["profile"] == "balanced"
        assert data["max_generations"] == 50
        assert data["population_size"] == 8
        assert "seed" in data

    def test_describe(self) -> None:
        """Test human-readable description."""
        config = RuntimeConfig.from_profile("slow-local")
        desc = config.describe()
        
        assert "slow-local" in desc
        assert "Population Size: 3" in desc
        assert "Max Concurrency: 1" in desc


class TestRuntimeConfigImmutability:
    """Tests for frozen dataclass behavior."""

    def test_cannot_modify_fields(self) -> None:
        """Test that config fields cannot be modified."""
        config = RuntimeConfig.from_profile("balanced")
        
        with pytest.raises(AttributeError):
            config.max_generations = 100  # type: ignore

    def test_with_overrides_returns_new_instance(self) -> None:
        """Test that with_overrides returns a new instance."""
        config = RuntimeConfig.from_profile("balanced")
        new_config = config.with_overrides(population_size=20)
        
        assert config is not new_config
        assert id(config) != id(new_config)
