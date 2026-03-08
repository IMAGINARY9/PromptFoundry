"""Tests for CLI helper behavior."""

from __future__ import annotations

from pathlib import Path

import yaml

from promptfoundry.cli import _apply_runtime_llm_overrides, _load_task
from promptfoundry.core.config import RuntimeConfig, RuntimeProfile
from promptfoundry.llm import LLMConfig


class TestApplyRuntimeLlmOverrides:
    """Tests for runtime-derived LLM settings."""

    def test_uses_runtime_timeout_when_llm_timeout_not_explicit(self) -> None:
        """Runtime timeout should flow into the LLM config by default."""
        llm_config = LLMConfig(timeout=30.0)
        runtime_config = RuntimeConfig.from_profile(RuntimeProfile.SLOW_LOCAL)

        updated = _apply_runtime_llm_overrides(llm_config, runtime_config, {})

        assert updated.timeout == 120.0

    def test_preserves_explicit_llm_timeout(self) -> None:
        """Explicit LLM timeout should win over runtime profile defaults."""
        llm_config = LLMConfig(timeout=45.0)
        runtime_config = RuntimeConfig.from_profile(RuntimeProfile.SLOW_LOCAL)

        updated = _apply_runtime_llm_overrides(
            llm_config,
            runtime_config,
            {"timeout": 45.0},
        )

        assert updated.timeout == 45.0


class TestLoadTask:
    """Tests for task loading from CLI YAML files."""

    def test_load_task_supports_validation_examples_and_expected_schema(
        self, tmp_path: Path
    ) -> None:
        """CLI loading should preserve validation examples from bundled task files."""
        task_file = tmp_path / "task.yaml"
        task_data = {
            "name": "demo_task",
            "system_prompt": "You are helpful.",
            "evaluator": "exact_match",
            "evaluator_config": {"normalize_output": False},
            "examples": [{"input": "a", "expected": "b"}],
            "validation_examples": [{"input": "c", "expected": "d"}],
        }
        task_file.write_text(yaml.safe_dump(task_data), encoding="utf-8")

        task, evaluator_type, evaluator_config = _load_task(task_file)

        assert task.examples[0].expected_output == "b"
        assert task.validation_examples is not None
        assert task.validation_examples[0].expected_output == "d"
        assert evaluator_type == "exact_match"
        assert evaluator_config["normalize_output"] is False
