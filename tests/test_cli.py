"""Tests for CLI helper behavior."""

from __future__ import annotations

from promptfoundry.cli import _apply_runtime_llm_overrides
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